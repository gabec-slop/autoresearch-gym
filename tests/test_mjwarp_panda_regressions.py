from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch


def test_panda_curriculum_uses_competence_not_raw_vector_clock() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable as seed

    class FlatVectorEnv:
        num_envs = 4

        def step(self, actions):
            del actions
            raw = np.zeros(self.num_envs, dtype=np.float32)
            infos = {
                "ee_to_cube_distance": np.full(self.num_envs, 0.58, dtype=np.float32),
                "cube_to_goal_distance": np.full(self.num_envs, 0.20, dtype=np.float32),
                "ee_to_cube_progress": np.zeros(self.num_envs, dtype=np.float32),
                "cube_to_goal_progress": np.zeros(self.num_envs, dtype=np.float32),
                "initial_ee_to_cube_distance": np.full(self.num_envs, 0.22, dtype=np.float32),
                "initial_cube_to_goal_distance": np.full(self.num_envs, 0.20, dtype=np.float32),
                "cube_lift_height": np.zeros(self.num_envs, dtype=np.float32),
                "near_cube": np.zeros(self.num_envs, dtype=bool),
                "gripper_closed_near_cube": np.zeros(self.num_envs, dtype=bool),
                "lifted": np.zeros(self.num_envs, dtype=bool),
                "lifted_ever": np.zeros(self.num_envs, dtype=bool),
                "placed_success": np.zeros(self.num_envs, dtype=bool),
            }
            return np.zeros((self.num_envs, 8), dtype=np.float32), raw, np.zeros(self.num_envs, dtype=bool), infos

    state = seed.CurriculumState()
    _, rewards, _, infos = seed._step_vector_env(
        FlatVectorEnv(),
        np.zeros((4, 2), dtype=np.float32),
        global_step=630_000,
        curriculum_state=state,
    )

    assert state.phase == "approach"
    assert np.all(infos["curriculum_phase_index"] == 0.0)
    assert np.all(rewards > 0.0)


def test_panda_curriculum_advances_only_after_diagnostic_competence() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable as seed

    state = seed.CurriculumState(window=2)
    state.update_from_infos({"near_cube": np.array([True, True, False, False]), "lifted_ever": np.zeros(4, dtype=bool)})
    assert state.phase == "grasp_lift"

    state.update_from_infos({"near_cube": np.ones(4, dtype=bool), "lifted_ever": np.array([True, True, False, False])})
    assert state.phase == "place"


def test_panda_ppo_advantage_normalization_low_variance_guard() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable as seed

    constant = torch.ones(8)
    assert torch.allclose(seed._normalize_advantages(constant), torch.zeros_like(constant))

    tiny_variance = torch.tensor([1.0, 1.0, 1.0, 1.000001])
    normalized = seed._normalize_advantages(tiny_variance)
    assert float(normalized.abs().max()) < 1e-4


def test_panda_ppo_logprob_matches_raw_stored_action() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable as seed

    agent = seed.Agent(3, 1)
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.actor_logstd.fill_(0.0)
    obs = torch.zeros((1, 3))
    raw_action = torch.tensor([[2.0]])

    stored_action, old_logprob, _, _ = agent.get_action_and_value(obs, raw_action)
    env_action = torch.clamp(stored_action, -1.0, 1.0)
    _, replay_logprob, _, _ = agent.get_action_and_value(obs, stored_action)

    assert stored_action.item() == pytest.approx(2.0)
    assert env_action.item() == pytest.approx(1.0)
    assert replay_logprob.item() == pytest.approx(old_logprob.item())


def test_panda_env_grasp_width_and_prelift_dense_reward_have_signal() -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    assert bool(panda_env._is_grasp_width(0.04))
    assert not bool(panda_env._is_grasp_width(0.0))
    assert not bool(panda_env._is_grasp_width(0.08))

    env = panda_env.AutoresearchMujocoPandaPickAndPlaceEnv.__new__(panda_env.AutoresearchMujocoPandaPickAndPlaceEnv)
    env.reward_type = "dense"
    env.success_requires_lift = True
    achieved = np.array([0.0, 0.0, panda_env.CUBE_Z], dtype=np.float32)
    desired = np.array([0.2, 0.0, panda_env.CUBE_Z], dtype=np.float32)
    far = float(env.compute_reward(achieved, desired, {"lifted_ever": False, "ee_to_cube_distance": 0.50}))
    near = float(env.compute_reward(achieved, desired, {"lifted_ever": False, "ee_to_cube_distance": 0.03}))

    assert near > far
    assert far > -1.0


def test_panda_scene_injects_pinch_site_into_hand_body(tmp_path: Path) -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    panda_xml = tmp_path / "panda.xml"
    panda_xml.write_text(
        """
        <mujoco model="panda">
          <compiler/>
          <worldbody>
            <body name="link0" pos="0 0 0">
              <body name="hand" pos="0 0 0"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )

    scene = panda_env._write_scene_xml(panda_xml)
    root = ET.parse(scene).getroot()
    pinch = root.find(".//body[@name='hand']/site[@name='pinch']")

    assert pinch is not None
    assert [float(value) for value in pinch.get("pos", "").split()] == pytest.approx(
        panda_env.PINCH_OFFSET_FROM_HAND.tolist()
    )


def test_panda_warp_obs_uses_pinch_offset_instead_of_cube_proxy() -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    class ArrayField:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def numpy(self):
            return self.value

    class WarpData:
        pass

    vec = panda_env.MujocoWarpPandaPickAndPlaceVectorEnv.__new__(panda_env.MujocoWarpPandaPickAndPlaceVectorEnv)
    vec.cube_qpos_adr = 0
    vec.cube_qvel_adr = 0
    vec.robot_qpos_adrs = np.asarray([], dtype=np.int32)
    vec.robot_qvel_adrs = np.asarray([], dtype=np.int32)
    vec.model = type("Model", (), {"nu": 2})()
    vec.ee_site_id = -1
    vec.ee_body_id = 0
    vec.goals = np.asarray([[0.2, 0.0, panda_env.CUBE_Z]], dtype=np.float32)
    vec.last_actions = np.zeros((1, 2), dtype=np.float32)
    vec.warp_data = WarpData()
    vec.warp_data.xpos = ArrayField([[[0.1, 0.2, 0.3]]])
    vec.warp_data.xmat = ArrayField([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]])
    qpos = np.asarray([[0.0, 0.0, panda_env.CUBE_Z, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    qvel = np.zeros((1, 3), dtype=np.float32)

    obs = vec._obs_from_arrays(qpos, qvel)

    assert obs[0, :3] == pytest.approx((np.asarray([0.1, 0.2, 0.3]) + panda_env.PINCH_OFFSET_FROM_HAND).tolist())
    assert obs[0, :3] != pytest.approx([0.0, 0.0, panda_env.CUBE_Z + 0.12])


def test_oracle_fallback_uses_observed_pinch_without_extra_offset() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import oracle_policy

    obs = np.zeros((1, 43), dtype=np.float32)
    obs[:, 0:3] = np.asarray([[0.05, -0.02, 0.12]], dtype=np.float32)
    obs[:, 3:6] = np.asarray([[0.05, -0.02, 0.02]], dtype=np.float32)
    obs[:, 6:9] = np.asarray([[0.10, 0.04, 0.02]], dtype=np.float32)

    implicit_state = oracle_policy.ScriptedPickPlaceOracle(1)
    explicit_state = oracle_policy.ScriptedPickPlaceOracle(1)
    implicit = implicit_state.actions(obs, noise_scale=0.0)
    explicit = explicit_state.actions(obs, noise_scale=0.0, control_pos=obs[:, 0:3])

    assert implicit[:, :3] == pytest.approx(explicit[:, :3])

    advance_state = oracle_policy.ScriptedPickPlaceOracle(1)
    advance_state.phase[:] = 3
    advance_state.advance(
        obs,
        np.asarray([False], dtype=bool),
        {"lifted_ever": np.asarray([True]), "cube_at_goal": np.asarray([False]), "placed_success": np.asarray([False])},
    )

    assert advance_state.grasp_offset[0] == pytest.approx((obs[0, 0:3] - obs[0, 3:6]).tolist())
