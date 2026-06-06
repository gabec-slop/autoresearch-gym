from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


class PixelObservationWrapper(gym.Wrapper):
    """Expose low-resolution named task feeds as policy observations."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        policy_feeds: tuple[str, ...] = ("wrist",),
        image_size: tuple[int, int] = (84, 84),
        frame_stack: int = 3,
        include_proprio: bool = True,
    ) -> None:
        super().__init__(env)
        self.policy_feeds = tuple(policy_feeds)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.frame_stack = int(frame_stack)
        self.include_proprio = bool(include_proprio)
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self._frames: dict[str, deque[np.ndarray]] = {name: deque(maxlen=self.frame_stack) for name in self.policy_feeds}
        self._proprio_dim = int(self._read_proprio().size) if self.include_proprio else 0
        channels = 3 * len(self.policy_feeds) * self.frame_stack
        height, width = self.image_size
        spaces_dict: dict[str, spaces.Space[Any]] = {
            "pixels": spaces.Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8),
        }
        if self.include_proprio:
            spaces_dict["proprio"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self._proprio_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(spaces_dict)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _obs, info = self.env.reset(seed=seed, options=options)
        feeds = self._capture_policy_feeds()
        for name, frame in feeds.items():
            self._frames[name].clear()
            for _ in range(self.frame_stack):
                self._frames[name].append(frame)
        return self._observation(), info

    def step(self, action: np.ndarray):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        feeds = self._capture_policy_feeds()
        for name, frame in feeds.items():
            self._frames[name].append(frame)
        return self._observation(), float(reward), terminated, truncated, info

    def _capture_policy_feeds(self) -> dict[str, np.ndarray]:
        provider = getattr(self.env, "unwrapped", self.env)
        feed_fn = getattr(provider, "sample_feeds", None)
        if not callable(feed_fn):
            raise AttributeError("PixelObservationWrapper requires env.sample_feeds()")
        raw_feeds = feed_fn()
        frames: dict[str, np.ndarray] = {}
        for name in self.policy_feeds:
            if name not in raw_feeds:
                raise KeyError(f"policy feed {name!r} is not available")
            frames[name] = _resize_rgb(np.asarray(raw_feeds[name], dtype=np.uint8), self.image_size)
        return frames

    def _observation(self) -> dict[str, np.ndarray]:
        pixel_parts: list[np.ndarray] = []
        for name in self.policy_feeds:
            frames = list(self._frames[name])
            if len(frames) != self.frame_stack:
                raise RuntimeError(f"feed {name!r} stack is incomplete")
            for frame in frames:
                pixel_parts.append(np.transpose(frame, (2, 0, 1)))
        obs: dict[str, np.ndarray] = {
            "pixels": np.concatenate(pixel_parts, axis=0).astype(np.uint8, copy=False),
        }
        if self.include_proprio:
            obs["proprio"] = self._read_proprio()
        return obs

    def _read_proprio(self) -> np.ndarray:
        provider = getattr(self.env, "unwrapped", self.env)
        proprio_fn = getattr(provider, "proprio_observation", None)
        if callable(proprio_fn):
            return np.asarray(proprio_fn(), dtype=np.float32).reshape(-1)
        return np.zeros(0, dtype=np.float32)


def _resize_rgb(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8))
    height, width = image_size
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)
