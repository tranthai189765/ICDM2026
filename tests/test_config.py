"""
Unit tests for dmorl/config.py

Pure data-class tests — no mocks, no network access, runs in milliseconds.
"""
import pytest

from dmorl.config import (
    DMORLConfig,
    DMORLConfigForRecommendation,
    DMORLConfigForNegotiation,
    DMORLConfigForEmotionalSupport,
)
from padpp.config import PADPPConfig


# ─────────────────────────────────────────────────────────────────────────────
# Inheritance
# ─────────────────────────────────────────────────────────────────────────────

class TestInheritance:
    def test_dmorl_config_is_subclass_of_padpp_config(self):
        assert issubclass(DMORLConfig, PADPPConfig)

    def test_scenario_configs_are_subclasses_of_dmorl_config(self):
        for cls in [
            DMORLConfigForRecommendation,
            DMORLConfigForNegotiation,
            DMORLConfigForEmotionalSupport,
        ]:
            assert issubclass(cls, DMORLConfig), f"{cls.__name__} must extend DMORLConfig"


# ─────────────────────────────────────────────────────────────────────────────
# DMORLConfig class-level defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestDMORLConfigDefaults:
    def test_n_basic_skills(self):
        assert DMORLConfig.n_basic_skills == 5

    def test_n_advanced_skills(self):
        assert DMORLConfig.n_advanced_skills == 5

    def test_n_skill_train_epochs(self):
        assert DMORLConfig.n_skill_train_epochs == 15

    def test_n_advanced_train_epochs(self):
        assert DMORLConfig.n_advanced_train_epochs == 20

    def test_use_dynamic_weight(self):
        assert DMORLConfig.use_dynamic_weight is True

    def test_dynamic_weight_horizon(self):
        assert DMORLConfig.dynamic_weight_horizon == 3

    def test_use_hints(self):
        assert DMORLConfig.use_hints is True

    def test_run_curriculum_true(self):
        assert DMORLConfig.run_curriculum is True

    def test_force_rediscover_false(self):
        assert DMORLConfig.force_rediscover_skills is False

    def test_inherits_padpp_mlp_hidden_size(self):
        assert DMORLConfig.mlp_hidden_size == 128

    def test_inherits_padpp_freeze_plm(self):
        assert DMORLConfig.freeze_plm is True


# ─────────────────────────────────────────────────────────────────────────────
# DMORLConfig instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestDMORLConfigInit:
    def test_empty_params_preserves_class_defaults(self):
        cfg = DMORLConfig({})
        assert cfg.n_basic_skills == 5
        assert cfg.use_dynamic_weight is True

    def test_params_override_class_defaults(self):
        cfg = DMORLConfig({"n_basic_skills": 10, "dynamic_weight_horizon": 5})
        assert cfg.n_basic_skills == 10
        assert cfg.dynamic_weight_horizon == 5

    def test_arbitrary_params_are_stored(self):
        cfg = DMORLConfig({"custom_key": "custom_value"})
        assert cfg.custom_key == "custom_value"

    def test_instantiation_does_not_raise(self):
        DMORLConfig({})  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-specific configs
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationConfig:
    def test_combined_action_is_false(self):
        assert DMORLConfigForRecommendation.combined_action is False

    def test_obj_to_weight_has_uniform_key(self):
        assert "uniform" in DMORLConfigForRecommendation.obj_to_weight
        assert DMORLConfigForRecommendation.obj_to_weight["uniform"] is None

    def test_non_uniform_weights_have_dim_2(self):
        for key, val in DMORLConfigForRecommendation.obj_to_weight.items():
            if val is not None:
                assert len(val) == 2, f"obj_to_weight['{key}'] should have 2 elements"

    def test_corner_weights_are_valid_probability_vectors(self):
        for key, val in DMORLConfigForRecommendation.obj_to_weight.items():
            if val is not None:
                assert abs(sum(val) - 1.0) < 1e-6
                assert all(x >= 0.0 for x in val)


class TestNegotiationConfig:
    def test_combined_action_is_true(self):
        assert DMORLConfigForNegotiation.combined_action is True

    def test_n_topics(self):
        assert DMORLConfigForNegotiation.n_topics == 5

    def test_obj_to_weight_has_uniform_key(self):
        assert "uniform" in DMORLConfigForNegotiation.obj_to_weight
        assert DMORLConfigForNegotiation.obj_to_weight["uniform"] is None

    def test_non_uniform_weights_have_dim_3(self):
        for key, val in DMORLConfigForNegotiation.obj_to_weight.items():
            if val is not None:
                assert len(val) == 3, f"obj_to_weight['{key}'] should have 3 elements"

    def test_corner_weights_are_valid(self):
        for key, val in DMORLConfigForNegotiation.obj_to_weight.items():
            if val is not None:
                assert abs(sum(val) - 1.0) < 1e-6


class TestEmotionalSupportConfig:
    def test_combined_action_is_false(self):
        assert DMORLConfigForEmotionalSupport.combined_action is False

    def test_obj_to_weight_has_uniform_key(self):
        assert "uniform" in DMORLConfigForEmotionalSupport.obj_to_weight
        assert DMORLConfigForEmotionalSupport.obj_to_weight["uniform"] is None

    def test_non_uniform_weights_have_dim_3(self):
        for key, val in DMORLConfigForEmotionalSupport.obj_to_weight.items():
            if val is not None:
                assert len(val) == 3, f"obj_to_weight['{key}'] should have 3 elements"

    def test_corner_weights_are_valid(self):
        for key, val in DMORLConfigForEmotionalSupport.obj_to_weight.items():
            if val is not None:
                assert abs(sum(val) - 1.0) < 1e-6
                assert all(x >= 0.0 for x in val)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-scenario sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioSanity:
    ALL = [
        DMORLConfigForRecommendation,
        DMORLConfigForNegotiation,
        DMORLConfigForEmotionalSupport,
    ]

    def test_all_have_obj_to_weight_dict(self):
        for cls in self.ALL:
            assert hasattr(cls, "obj_to_weight")
            assert isinstance(cls.obj_to_weight, dict)

    def test_all_have_skills_and_hints_files(self):
        for cls in self.ALL:
            assert hasattr(cls, "skills_file")
            assert hasattr(cls, "hints_file")
            assert cls.skills_file.endswith(".json")
            assert cls.hints_file.endswith(".json")

    def test_all_inherit_run_curriculum(self):
        for cls in self.ALL:
            assert cls.run_curriculum is True
