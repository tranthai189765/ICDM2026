"""
Unit tests for dmorl/trainer.py

DMORLTrainer is instantiated with object.__new__ to bypass the heavy
PADPPTrainer.__init__ (no game setup, no model download, no data loading).
Only the attributes used by the tested methods are populated.
"""
import pytest
import torch
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from tests.conftest import make_trainer


# ─────────────────────────────────────────────────────────────────────────────
# post_dialogue_refinement
# ─────────────────────────────────────────────────────────────────────────────

class TestPostDialogueRefinement:
    def test_calls_controller_with_correct_args(self):
        trainer = make_trainer(use_hints=True)
        ctrl = MagicMock()
        ctrl.refine_after_dialogue.return_value = ["hint1"]
        trainer.dmorl_controller = ctrl

        history = [{"role": "user", "content": "hi"}]
        trainer.post_dialogue_refinement(history, "success")

        ctrl.refine_after_dialogue.assert_called_once_with(history, "success")

    def test_does_not_raise_when_no_controller(self):
        trainer = make_trainer(use_hints=True)
        trainer.dmorl_controller = None
        trainer.post_dialogue_refinement([], "success")  # must not raise

    def test_skips_when_use_hints_is_false(self):
        trainer = make_trainer(use_hints=False)
        ctrl = MagicMock()
        trainer.dmorl_controller = ctrl
        trainer.post_dialogue_refinement([], "failure")
        ctrl.refine_after_dialogue.assert_not_called()

    def test_outcome_passed_verbatim(self):
        trainer = make_trainer(use_hints=True)
        ctrl = MagicMock()
        ctrl.refine_after_dialogue.return_value = []
        trainer.dmorl_controller = ctrl
        trainer.post_dialogue_refinement([], "partial")
        _, outcome = ctrl.refine_after_dialogue.call_args[0]
        assert outcome == "partial"


# ─────────────────────────────────────────────────────────────────────────────
# train_basic_skills
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainBasicSkills:
    def test_returns_silently_when_no_controller(self):
        trainer = make_trainer()
        trainer.dmorl_controller = None
        trainer.train_basic_skills([], device=None, simulators=[], action_mapping={})

    def test_skips_rlt_when_basic_skills_empty(self):
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.basic_skills = []
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_basic_skills([], device=None, simulators=[], action_mapping={})
        mock_rlt.assert_not_called()

    def test_calls_rlt_once_with_all_basic_skills(self):
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.basic_skills = [
            {"name": "S1", "weight_vector": [1.0, 0.0]},
            {"name": "S2", "weight_vector": [0.5, 0.5]},
            {"name": "S3", "weight_vector": [0.0, 1.0]},
        ]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_basic_skills(["case"], device=None, simulators=["sim"], action_mapping={})

        assert mock_rlt.call_count == 1
        kwargs = mock_rlt.call_args[1]
        assert kwargs["phase"] == "1a"
        assert kwargs["p_skill"] == 1.0
        assert kwargs["skill_names"] == ["S1", "S2", "S3"]
        np.testing.assert_allclose(
            kwargs["skill_weights"],
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            atol=1e-6,
        )

    def test_rlt_call_uses_basic_skill_weight_bank(self):
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.basic_skills = [
            {"name": "S1", "weight_vector": [1.0, 0.0]},
            {"name": "S2", "weight_vector": [0.0, 1.0]},
        ]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_basic_skills([], device=None, simulators=[], action_mapping={})

        kwargs = mock_rlt.call_args[1]
        assert "fixed_weight" not in kwargs
        np.testing.assert_allclose(kwargs["skill_weights"], [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)
        assert kwargs["skill_names"] == ["S1", "S2"]

    def test_restores_num_train_rl_epochs_after_completion(self):
        trainer = make_trainer()
        original = trainer.model_config.num_train_rl_epochs
        ctrl = MagicMock()
        ctrl.skill_library.basic_skills = [{"name": "S", "weight_vector": [0.5, 0.5]}]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt"):
            trainer.train_basic_skills([], device=None, simulators=[], action_mapping={})

        assert trainer.model_config.num_train_rl_epochs == original


# ─────────────────────────────────────────────────────────────────────────────
# train_advanced_skills
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainAdvancedSkills:
    def test_returns_silently_when_no_controller(self):
        trainer = make_trainer()
        trainer.dmorl_controller = None
        trainer.train_advanced_skills([], device=None, simulators=[], action_mapping={})

    def test_skips_rlt_when_advanced_skills_empty(self):
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.advanced_skills = []
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_advanced_skills([], device=None, simulators=[], action_mapping={})
        mock_rlt.assert_not_called()

    def test_calls_rlt_exactly_once(self):
        """Phase-1b trains all advanced skills in a single _run_curriculum_rlt call."""
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.advanced_skills = [
            {"name": "A1", "weight_vector": [0.6, 0.4]},
            {"name": "A2", "weight_vector": [0.3, 0.7]},
        ]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_advanced_skills([], device=None, simulators=[], action_mapping={})

        assert mock_rlt.call_count == 1

    def test_rlt_called_with_p_skill_0_6(self):
        trainer = make_trainer()
        ctrl = MagicMock()
        ctrl.skill_library.advanced_skills = [{"name": "A", "weight_vector": [0.5, 0.5]}]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt") as mock_rlt:
            trainer.train_advanced_skills([], device=None, simulators=[], action_mapping={})

        kwargs = mock_rlt.call_args[1]
        assert "p_skill" in kwargs, "_run_curriculum_rlt must receive p_skill"
        assert abs(kwargs["p_skill"] - 0.6) < 1e-6

    def test_restores_num_train_rl_epochs_after_completion(self):
        trainer = make_trainer()
        original = trainer.model_config.num_train_rl_epochs
        ctrl = MagicMock()
        ctrl.skill_library.advanced_skills = [{"name": "A", "weight_vector": [0.5, 0.5]}]
        trainer.dmorl_controller = ctrl

        with patch.object(trainer, "_run_curriculum_rlt"):
            trainer.train_advanced_skills([], device=None, simulators=[], action_mapping={})

        assert trainer.model_config.num_train_rl_epochs == original


# ─────────────────────────────────────────────────────────────────────────────
# predict_dynamic — horizon / LLM-query logic
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictDynamic:
    """
    Tests the when-to-query-LLM logic of predict_dynamic.
    The heavy predict() call is mocked so no model forward-pass is needed.
    """

    def _patch_predict(self, trainer):
        return patch.object(trainer, "predict", return_value=("action", 0.0, None))

    def test_queries_llm_at_step_zero(self):
        trainer = make_trainer(use_dynamic=True, horizon=3)
        ctrl = MagicMock()
        ctrl.get_dynamic_weight.return_value = [0.6, 0.4]
        trainer.dmorl_controller = ctrl

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer):
            trainer.predict_dynamic(state, [], {}, step_in_episode=0)

        ctrl.get_dynamic_weight.assert_called_once()

    def test_no_llm_call_at_non_horizon_steps(self):
        trainer = make_trainer(use_dynamic=True, horizon=3)
        ctrl = MagicMock()
        ctrl.get_dynamic_weight.return_value = [0.5, 0.5]
        trainer.dmorl_controller = ctrl

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer):
            trainer.predict_dynamic(state, [], {}, step_in_episode=1)
            trainer.predict_dynamic(state, [], {}, step_in_episode=2)

        ctrl.get_dynamic_weight.assert_not_called()

    def test_queries_llm_again_at_multiples_of_horizon(self):
        trainer = make_trainer(use_dynamic=True, horizon=3)
        ctrl = MagicMock()
        ctrl.get_dynamic_weight.return_value = [0.5, 0.5]
        trainer.dmorl_controller = ctrl

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer):
            for step in [0, 1, 2, 3, 4, 5, 6]:
                trainer.predict_dynamic(state, [], {}, step_in_episode=step)

        # Steps 0, 3, 6 are multiples of horizon=3 → 3 calls
        assert ctrl.get_dynamic_weight.call_count == 3

    def test_no_llm_when_controller_is_none(self):
        trainer = make_trainer(use_dynamic=True, horizon=3)
        trainer.dmorl_controller = None

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer) as mock_pred:
            trainer.predict_dynamic(state, [], {}, step_in_episode=0)

        mock_pred.assert_called_once()  # predict() still called (falls through to standard path)

    def test_no_llm_when_use_dynamic_false(self):
        trainer = make_trainer(use_dynamic=False, horizon=3)
        ctrl = MagicMock()
        ctrl.get_dynamic_weight.return_value = [0.5, 0.5]
        trainer.dmorl_controller = ctrl

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer):
            trainer.predict_dynamic(state, [], {}, step_in_episode=0)

        ctrl.get_dynamic_weight.assert_not_called()

    def test_llm_weight_written_back_to_state(self):
        """After a horizon step, state['w'] must be updated with the LLM weight."""
        trainer = make_trainer(use_dynamic=True, horizon=3)
        ctrl = MagicMock()
        ctrl.get_dynamic_weight.return_value = [0.7, 0.3]
        trainer.dmorl_controller = ctrl

        state = {"w": [0.5, 0.5]}
        with self._patch_predict(trainer):
            trainer.predict_dynamic(state, [], {}, step_in_episode=0)

        np.testing.assert_allclose(state["w"], [0.7, 0.3], atol=1e-6)
