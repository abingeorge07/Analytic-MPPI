"""Gymnasium (Box2D) backend for the MPPI testbed -- LunarLanderContinuous.

The MPPI controllers were written against MuJoCo, whose backend offers two
things Box2D does not: batched parallel rollouts and native state injection.
This backend recovers both with a thin shim so the *same* controllers can drive
a real Gymnasium environment:

  * State injection -- LunarLander's dynamic state is three Box2D bodies (the
    lander + two legs). `get_state`/`set_state` snapshot/restore their
    kinematics (position, angle, linear & angular velocity) -> an `nstate = 18`
    vector. The leg revolute-joint internal state is not captured (recomputed by
    the solver on the next step); fine for the short horizons MPPI uses.

  * Batched rollout -- `rollout` replays each of the B samples *serially* in a
    dedicated SCRATCH env, leaving the real env untouched (mirroring
    MujocoBackend's per-thread `MjData` separation). The scratch env is reset
    with the *same seed* as the real env so its randomized terrain/helipad match.

LunarLander engine impulses carry a small per-step `np_random` dispersion, so a
rollout is not a pure function of (state, controls). We reseed the scratch env's
RNG to a fixed seed before each sample (common random numbers) so that dispersion
is identical across samples -- variance-reduction that makes the MPPI cost
comparison reflect the controls, not the noise.
"""
from __future__ import annotations

import numpy as np

FPS = 50          # gymnasium LunarLander integrates at 1/FPS s per step
_NBODY = 3        # lander + 2 legs
_BODY_DOF = 6     # x, y, angle, vx, vy, omega


class GymBackend:
    """Backend wrapping `LunarLanderContinuous-v3` with the MujocoBackend interface.

    Exposes `nstate`, `nu`, `dt`, `u_min`, `u_max`, `nobs`, `nsensordata` and the
    methods `reset`, `get_state`, `set_state`, `step`, `observe`, `rollout` that the
    controllers (and the closed-loop drivers) rely on.
    """

    def __init__(
        self,
        env_id: str = "LunarLanderContinuous-v3",
        *,
        seed: int = 0,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ):
        import gymnasium as gym   # imported lazily so the core package stays MuJoCo-only

        self.env_id = env_id
        self._seeding = __import__("gymnasium.utils.seeding", fromlist=["np_random"]).np_random
        # real env (advanced by step); scratch env (used only inside rollout)
        self.env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
        self._scratch = gym.make(env_id, max_episode_steps=max_episode_steps)
        self._scratch_u = self._scratch.unwrapped       # step the unwrapped env in rollout (no TimeLimit)

        self.nu = int(self.env.action_space.shape[0])    # 2 (continuous)
        self.u_min = np.asarray(self.env.action_space.low, dtype=np.float64).reshape(self.nu)
        self.u_max = np.asarray(self.env.action_space.high, dtype=np.float64).reshape(self.nu)
        self.nobs = int(self.env.observation_space.shape[0])   # 8
        self.nstate = _NBODY * _BODY_DOF                       # 18 (Box2D kinematics snapshot)
        self.nsensordata = 3                                   # [terminated, crashed, landed]
        self.dt = 1.0 / FPS

        self.seed = int(seed)
        self._last_obs = np.zeros(self.nobs, dtype=np.float64)
        self._last_terminated = False
        self.reset(self.seed)

    # ---- episode reset (scratch terrain kept identical via shared seed) ----

    def reset(self, seed: int | None = None) -> np.ndarray:
        s = self.seed if seed is None else int(seed)
        self.seed = s
        obs0, _ = self.env.reset(seed=s)
        self._scratch.reset(seed=s)                 # same seed -> identical terrain/helipad
        self._last_obs = np.asarray(obs0, dtype=np.float64)
        self._last_terminated = False
        return self.get_state()

    # ---- Box2D kinematics snapshot / restore ----

    @staticmethod
    def _bodies(env_unwrapped):
        return (env_unwrapped.lander, env_unwrapped.legs[0], env_unwrapped.legs[1])

    def _snapshot(self, env_unwrapped) -> np.ndarray:
        out = np.empty(self.nstate, dtype=np.float64)
        for k, b in enumerate(self._bodies(env_unwrapped)):
            p, v = b.position, b.linearVelocity
            out[k * 6:(k + 1) * 6] = (p.x, p.y, b.angle, v.x, v.y, b.angularVelocity)
        return out

    def _restore(self, env_unwrapped, state: np.ndarray) -> None:
        for k, b in enumerate(self._bodies(env_unwrapped)):
            x, y, ang, vx, vy, om = state[k * 6:(k + 1) * 6]
            b.position = (float(x), float(y))
            b.angle = float(ang)
            b.linearVelocity = (float(vx), float(vy))
            b.angularVelocity = float(om)
            b.awake = True
        env_unwrapped.legs[0].ground_contact = False
        env_unwrapped.legs[1].ground_contact = False
        env_unwrapped.game_over = False

    def get_state(self) -> np.ndarray:
        return self._snapshot(self.env.unwrapped)

    def set_state(self, state: np.ndarray) -> None:
        self._restore(self.env.unwrapped, np.asarray(state, dtype=np.float64))

    def observe(self) -> np.ndarray:
        """Current 8-dim observation of the real env (for logging/plots)."""
        return self._last_obs.copy()

    # ---- single real step ----

    def step(self, control: np.ndarray) -> np.ndarray:
        obs, _reward, terminated, _truncated, _info = self.env.step(
            np.asarray(control, dtype=np.float64)
        )
        self._last_obs = np.asarray(obs, dtype=np.float64)
        self._last_terminated = bool(terminated)
        return self.get_state()

    @property
    def terminated(self) -> bool:
        """Did the last real `step` end the episode (crash or landing)?"""
        return self._last_terminated

    def outcome(self) -> str:
        """Classify the real env's current terminal state: 'crash' / 'land' / 'running'."""
        if not self._last_terminated:
            return "running"
        crashed = bool(self.env.unwrapped.game_over or abs(self._last_obs[0]) >= 1.0)
        return "crash" if crashed else "land"

    # ---- batched (serial) rollout in the scratch env ----

    def rollout(self, initial_states: np.ndarray, controls: np.ndarray):
        """Replay B control sequences from `initial_states` (each (nstate,)).

        initial_states: (B, nstate)   controls: (B, H, nu)
        returns: states (B, H, nobs) [observations], sensordata (B, H, 3)
                 with per-step flags [terminated, crashed, landed].
        After a sample terminates, its remaining steps are frozen at the
        terminal observation/flags (so the terminal outcome is readable at [-1]).
        """
        initial_states = np.asarray(initial_states, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        B, H, _ = controls.shape
        states = np.zeros((B, H, self.nobs), dtype=np.float64)
        sens = np.zeros((B, H, self.nsensordata), dtype=np.float64)

        scu = self._scratch_u
        for b in range(B):
            # Fresh world each sample (same seed -> identical terrain) clears the
            # Box2D solver/contact warm-start caches, so samples don't leak state
            # into one another; then restore the 3 bodies to the current state.
            self._scratch.reset(seed=self.seed)
            self._restore(scu, initial_states[b])
            scu.np_random, _ = self._seeding(self.seed)     # common random numbers across samples
            last_obs = None
            last_sens = np.zeros(self.nsensordata, dtype=np.float64)
            for h in range(H):
                if last_obs is None:
                    obs, _r, term, trunc, _i = scu.step(controls[b, h])
                    obs = np.asarray(obs, dtype=np.float64)
                    crashed = bool(scu.game_over or abs(obs[0]) >= 1.0)
                    landed = bool(term and not crashed)
                    cur_sens = np.array([float(term), float(crashed), float(landed)])
                    states[b, h] = obs
                    sens[b, h] = cur_sens
                    if term or trunc:
                        last_obs, last_sens = obs, cur_sens
                else:                                        # frozen after termination
                    states[b, h] = last_obs
                    sens[b, h] = last_sens
        return states, sens
