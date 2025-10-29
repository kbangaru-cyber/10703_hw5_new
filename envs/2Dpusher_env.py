import numpy as np
import copy
import cv2
import Box2D
from Box2D.b2 import (circleShape, fixtureDef, polygonShape)

import gym
from gym import spaces
try:
    from gym.utils import seeding
except Exception:
    # Optional fallback if you ever swap to Gymnasium later.
    from gymnasium.utils import seeding  # type: ignore

from utils.opencv_draw import OpencvDrawFuncs

MIN_COORD = 0
MAX_COORD = 5
PUSHER_START = np.array([1.0, 1.0])
BOX_START   = np.array([2.0, 2.0])
FORCE_MULT  = 1
RAD         = 0.2
SIDE_GAP_MULT = 2
BOX_RAD     = 0.2
GOAL_RAD    = 0.5
MAX_STEPS   = 40
FPS         = 4


class Pusher2d(gym.Env):
    """
    Gym-style env:
      - reset() -> obs
      - step(a) -> (obs, reward, done, info)
      - step_state(s_next_full) -> (obs, reward, done, info)  [used for synthetic transitions]
    State (10D): [pusher_xy(2), box_xy(2), pusher_vel_xy(2), box_vel_xy(2), goal_xy(2)]
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, control_noise: float = 0.0):
        self.control_noise = control_noise
        self.seed()
        self.world = Box2D.b2World(gravity=(0, 0))
        self.pusher = None
        self.box = None

        # Actions: x-vel, y-vel in [-1, 1]
        self.action_space = spaces.Box(low=-np.ones(2, dtype=np.float32),
                                       high= np.ones(2, dtype=np.float32),
                                       dtype=np.float32)
        # Observations: 10D (see _get_obs). Bounds are coarse but sufficient.
        self.observation_space = spaces.Box(low = np.ones(10, dtype=np.float32) * MIN_COORD,
                                            high= np.ones(10, dtype=np.float32) * MAX_COORD,
                                            dtype=np.float32)

        self.reset()
        self.drawer = OpencvDrawFuncs(w=240, h=180, ppm=40)
        self.drawer.install()

    # ---------------- Seeding ----------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ---------------- Helpers ----------------
    def random_place(self):
        """Return [x, y] away from the initial box position."""
        return [
            self.np_random.uniform(BOX_START[0] + BOX_RAD + GOAL_RAD, MAX_COORD - RAD * SIDE_GAP_MULT),
            self.np_random.uniform(BOX_START[1] + BOX_RAD + GOAL_RAD, MAX_COORD - RAD * SIDE_GAP_MULT),
        ]

    def _destroy(self):
        """Remove Box2D bodies if created."""
        if not self.box:
            return
        self.world.DestroyBody(self.box)
        self.world.DestroyBody(self.pusher)
        self.box = None
        self.pusher = None

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        """Return the first observation of the episode (Gym API)."""
        if seed is not None:
            self.seed(seed)

        self._destroy()

        self.pusher = self.world.CreateDynamicBody(
            position=PUSHER_START.tolist(),
            fixtures=fixtureDef(
                shape=circleShape(radius=RAD, pos=(0, 0)),
                density=1.0
            )
        )
        self.box = self.world.CreateDynamicBody(
            position=BOX_START.tolist(),
            fixtures=fixtureDef(
                shape=circleShape(radius=BOX_RAD, pos=(0, 0)),
                density=1.0
            )
        )

        self.goal_pos = self.random_place()
        self.elapsed_steps = 0
        return self._get_obs()

    def get_done_reward(self, state):
        """Compute termination, reward, info based on current state."""
        pusher_position = state[:2]
        box_position    = state[2:4]
        obj_coords = np.concatenate([pusher_position, box_position])

        done, reward, info = False, -1.0, {"done": None}

        # Out of bounds
        if np.min(obj_coords) < MIN_COORD or np.max(obj_coords) > MAX_COORD:
            reward = -1.0 * (MAX_STEPS - self.elapsed_steps + 2)
            done = True
            info["done"] = "unstable simulation"

        # Max steps
        elif self.elapsed_steps >= MAX_STEPS:
            done = True
            info["done"] = "max_steps_reached"

        # Goal reached
        elif np.linalg.norm(np.array(self.box.position.tuple) - self.goal_pos) < (RAD + GOAL_RAD):
            done = True
            reward = 0.0
            info["done"] = "goal reached"

        return done, reward, info

    def step_state(self, new_state):
        """
        Set the full next state (10D) and return a Gym 4-tuple.
        Used by model-based synthetic transitions.
        """
        self.elapsed_steps += 1
        self.set_state(new_state)
        done, reward, info = self.get_done_reward(self._get_obs())
        obs = self._get_obs()
        return obs, float(reward), bool(done), info

    def step(self, action, render: bool = False):
        """Standard Gym step -> (obs, reward, done, info)."""
        if render:
            self.drawer.clear_screen()
            self.drawer.draw_world(self.world)
            self.drawer._draw_dot(self.goal_pos)

        action = np.clip(action, -1, 1).astype(np.float32)
        if self.control_noise > 0.0:
            action += self.np_random.normal(0.0, scale=self.control_noise, size=action.shape)

        self.elapsed_steps += 1
        # Apply "velocity command" to pusher and step physics
        self.pusher._b2Body__SetLinearVelocity((float(FORCE_MULT * action[0]),
                                                float(FORCE_MULT * action[1])))
        self.box._b2Body__SetActive(True)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        if render:
            cv2.imshow("world", self.drawer.screen)
            cv2.waitKey(20)

        done, reward, info = self.get_done_reward(self._get_obs())
        obs = self._get_obs()
        return obs, float(reward), bool(done), info

    # ---------------- State utilities ----------------
    def _get_obs(self):
        """Assemble current full state (10D)."""
        return np.concatenate([
            self.pusher.position.tuple,
            self.box.position.tuple,
            self.pusher.linearVelocity.tuple,
            self.box.linearVelocity.tuple,
            self.goal_pos,
        ]).astype(np.float32)

    def set_state(self, state, elapsed_steps=None):
        """Set the full state. If state is 10D, last 2 dims are the goal."""
        if not isinstance(state, list):
            state = state.tolist()
        if elapsed_steps is not None:
            self.elapsed_steps = elapsed_steps

        self.pusher.position       = state[:2]
        self.box.position          = state[2:4]
        self.pusher.linearVelocity = state[4:6]
        self.box.linearVelocity    = state[6:8]
        if len(state) == 10:
            self.goal_pos = state[8:10]

    def get_state(self):
        return copy.copy(self._get_obs())

    def get_nxt_state(self, state, action):
        """
        Simulate one step from a *given* state/action (without side effects),
        and return the next state's FIRST 8 dims (dynamics part).
        """
        original_state = self.get_state()
        original_elapsed_steps = self.elapsed_steps

        self.set_state(state)
        nxt_state, _, _, _ = self.step(action)
        nxt_state = nxt_state[:8]

        # Restore
        self.set_state(original_state)
        self.elapsed_steps = original_elapsed_steps
        return nxt_state
