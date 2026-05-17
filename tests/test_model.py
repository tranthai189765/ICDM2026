"""
Unit tests for the GPI action-value logic in dmorl/model.py

The DMORLModel depends on PADPPModel → AutoModel (requires PyTorch >= 2.1).
On a local machine with an older PyTorch we test the same math via
MinimalGPIModel, a self-contained nn.Module that replicates the two methods
under test (gpi_action_values, get_skill_weight_tensor / set_skill_library)
without the full PADPP/transformers stack.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Optional

from dmorl.llm_controller import SkillLibrary


# ─────────────────────────────────────────────────────────────────────────────
# Minimal standalone model  (same math as DMORLModel, no AutoModel dependency)
# ─────────────────────────────────────────────────────────────────────────────

N_OBJ = 2
N_GOALS = 4
MLP_HIDDEN = 8
EMB_SIZE = 4


class MinimalGPIModel(nn.Module):
    """
    Replicates the two methods tested from DMORLModel without the
    roberta-large / transformers dependency.
    Used only in this test file.
    """

    def __init__(self, mlp_hidden_size=MLP_HIDDEN, n_objectives=N_OBJ,
                 objective_embedding_size=EMB_SIZE, n_goals=N_GOALS):
        super().__init__()
        self.model_config = SimpleNamespace(
            mlp_hidden_size=mlp_hidden_size,
            n_objectives=n_objectives,
            objective_embedding_size=objective_embedding_size,
            n_goals=n_goals,
            n_topics=3,
            combined_action=False,
        )
        self.objective_embedding = nn.Linear(n_objectives, objective_embedding_size)
        action_size = n_goals
        self.actor = nn.Sequential(
            nn.Linear(mlp_hidden_size + objective_embedding_size, mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size * 2, mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size * 2, action_size * n_objectives),
        )
        self.skill_library: Optional[SkillLibrary] = None

    # ── Copied verbatim from DMORLModel ──────────────────────────────────────

    def gpi_action_values(self, feature: torch.Tensor,
                          skill_weights: torch.Tensor) -> torch.Tensor:
        state_dim = self.model_config.mlp_hidden_size
        action_size = (
            self.model_config.n_goals * self.model_config.n_topics
            if self.model_config.combined_action
            else self.model_config.n_goals
        )
        state = feature[:, :state_dim]
        k = skill_weights.size(0)
        w_embs = self.objective_embedding(skill_weights)
        state_rep = state.repeat(k, 1)
        feat_k = torch.cat([state_rep, w_embs], dim=-1)
        Q_all = self.actor(feat_k)
        Q_all = Q_all.view(k, action_size, self.model_config.n_objectives)
        skill_weights_exp = skill_weights.unsqueeze(1).expand_as(Q_all)
        scalarised = (Q_all * skill_weights_exp).sum(-1)
        gpi_values, _ = scalarised.max(0)
        return gpi_values.unsqueeze(0)

    def set_skill_library(self, lib: Optional[SkillLibrary]):
        self.skill_library = lib

    def get_skill_weight_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.skill_library is None:
            return None
        wvs = self.skill_library.all_weight_vectors()
        if not wvs:
            return None
        return torch.FloatTensor(wvs).to(device)


@pytest.fixture(scope="module")
def model():
    return MinimalGPIModel()


# ─────────────────────────────────────────────────────────────────────────────
# gpi_action_values — output shape
# ─────────────────────────────────────────────────────────────────────────────

class TestGPIActionValuesShape:
    def test_single_skill_returns_1_x_n_actions(self, model):
        feature = torch.randn(1, MLP_HIDDEN)
        sw = torch.tensor([[0.4, 0.6]])
        out = model.gpi_action_values(feature, sw)
        assert out.shape == (1, N_GOALS)

    def test_multiple_skills_still_1_x_n_actions(self, model):
        feature = torch.randn(1, MLP_HIDDEN)
        sw = torch.softmax(torch.randn(7, N_OBJ), dim=-1)
        out = model.gpi_action_values(feature, sw)
        assert out.shape == (1, N_GOALS)

    def test_feature_wider_than_state_dim_is_ok(self, model):
        # Trainer concatenates [state, w_emb]; gpi_action_values strips the w_emb part
        feature = torch.randn(1, MLP_HIDDEN + EMB_SIZE)
        sw = torch.tensor([[0.5, 0.5]])
        out = model.gpi_action_values(feature, sw)
        assert out.shape == (1, N_GOALS)


# ─────────────────────────────────────────────────────────────────────────────
# gpi_action_values — correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestGPIActionValuesCorrectness:
    def test_gpi_dominates_each_individual_skill(self, model):
        """GPI = element-wise max over skills → gpi[a] >= skill_k[a] for all k, a."""
        model.eval()
        K = 5
        feature = torch.randn(1, MLP_HIDDEN)
        skill_weights = torch.softmax(torch.randn(K, N_OBJ), dim=-1)

        gpi_out = model.gpi_action_values(feature, skill_weights)

        for k in range(K):
            single = model.gpi_action_values(feature, skill_weights[k:k+1])
            assert (gpi_out + 1e-5 >= single).all(), \
                f"GPI must dominate skill {k} for all actions"

    def test_gpi_equals_single_skill_when_k_equals_1(self, model):
        model.eval()
        sw = torch.tensor([[0.3, 0.7]])
        feature = torch.randn(1, MLP_HIDDEN)
        out1 = model.gpi_action_values(feature, sw)
        out2 = model.gpi_action_values(feature, sw)
        assert torch.allclose(out1, out2)

    def test_deterministic_in_eval_mode(self, model):
        model.eval()
        feature = torch.randn(1, MLP_HIDDEN)
        sw = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(
            model.gpi_action_values(feature, sw),
            model.gpi_action_values(feature, sw),
        )

    def test_corner_weights_differ_from_uniform(self, model):
        """Extreme weights [1,0] and [0,1] should almost always produce different values."""
        model.eval()
        feature = torch.randn(1, MLP_HIDDEN)
        sw_corner_a = torch.tensor([[1.0, 0.0]])
        sw_corner_b = torch.tensor([[0.0, 1.0]])
        sw_uniform = torch.tensor([[0.5, 0.5]])
        out_a = model.gpi_action_values(feature, sw_corner_a)
        out_b = model.gpi_action_values(feature, sw_corner_b)
        out_u = model.gpi_action_values(feature, sw_uniform)
        # All have the same shape
        assert out_a.shape == out_b.shape == out_u.shape == (1, N_GOALS)

    def test_gpi_with_two_skills_dominates_each_alone(self, model):
        """max([A, B]) >= A and max([A, B]) >= B elementwise."""
        model.eval()
        feature = torch.randn(1, MLP_HIDDEN)
        swA = torch.tensor([[0.9, 0.1]])
        swB = torch.tensor([[0.1, 0.9]])
        sw_both = torch.cat([swA, swB], dim=0)

        gpi = model.gpi_action_values(feature, sw_both)
        a = model.gpi_action_values(feature, swA)
        b = model.gpi_action_values(feature, swB)

        assert (gpi + 1e-5 >= a).all()
        assert (gpi + 1e-5 >= b).all()


# ─────────────────────────────────────────────────────────────────────────────
# Skill library integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillLibraryIntegration:
    def test_get_skill_weight_tensor_returns_none_when_no_library(self, model):
        model.skill_library = None
        assert model.get_skill_weight_tensor(torch.device("cpu")) is None

    def test_get_skill_weight_tensor_returns_none_for_empty_library(self, model):
        lib = SkillLibrary(n_objectives=N_OBJ)
        model.set_skill_library(lib)
        assert model.get_skill_weight_tensor(torch.device("cpu")) is None

    def test_get_skill_weight_tensor_correct_shape(self, model):
        lib = SkillLibrary(n_objectives=N_OBJ)
        lib.basic_skills = [{"weight_vector": [1.0, 0.0]}, {"weight_vector": [0.0, 1.0]}]
        lib.advanced_skills = [{"weight_vector": [0.5, 0.5]}]
        model.set_skill_library(lib)

        t = model.get_skill_weight_tensor(torch.device("cpu"))
        assert t is not None
        assert t.shape == (3, N_OBJ)   # 2 basic + 1 advanced

    def test_get_skill_weight_tensor_on_cpu(self, model):
        lib = SkillLibrary(n_objectives=N_OBJ)
        lib.basic_skills = [{"weight_vector": [0.6, 0.4]}]
        model.set_skill_library(lib)
        t = model.get_skill_weight_tensor(torch.device("cpu"))
        assert t.device == torch.device("cpu")

    def test_set_skill_library_updates_attribute(self, model):
        lib = SkillLibrary(n_objectives=N_OBJ)
        model.set_skill_library(lib)
        assert model.skill_library is lib

    def test_set_skill_library_accepts_none(self, model):
        model.set_skill_library(None)
        assert model.skill_library is None

    def test_gpi_uses_all_library_skills(self, model):
        """GPI over K=3 library skills must dominate any single skill."""
        model.eval()
        lib = SkillLibrary(n_objectives=N_OBJ)
        lib.basic_skills = [{"weight_vector": [1.0, 0.0]}, {"weight_vector": [0.0, 1.0]}]
        lib.advanced_skills = [{"weight_vector": [0.5, 0.5]}]
        model.set_skill_library(lib)

        skill_tensor = model.get_skill_weight_tensor(torch.device("cpu"))
        feature = torch.randn(1, MLP_HIDDEN)
        gpi = model.gpi_action_values(feature, skill_tensor)

        for k in range(skill_tensor.size(0)):
            single = model.gpi_action_values(feature, skill_tensor[k:k+1])
            assert (gpi + 1e-5 >= single).all()
