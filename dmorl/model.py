"""
DMORL Model
Extends PADPPModel with GPI across the full skill library
(basic + advanced skills discovered by LLM).
"""

import torch
import torch.nn as nn
from typing import List, Optional

from padpp.model import PADPPModel
from dmorl.llm_controller import SkillLibrary


class DMORLModel(PADPPModel):
    """
    Drop-in replacement for PADPPModel that is aware of the skill library.
    The underlying architecture is identical; the skill library is only used
    during inference (predict_with_skills_gpi) to expand the GPI search space
    with semantically meaningful weight vectors.
    """

    def __init__(self, model_config, skill_library: Optional[SkillLibrary] = None, **kwargs):
        super().__init__(model_config, **kwargs)
        self.skill_library = skill_library  # injected after skill discovery

    def set_skill_library(self, skill_library: SkillLibrary):
        self.skill_library = skill_library

    def gpi_action_values(self, feature: torch.Tensor,
                          skill_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute GPI-improved Q-values by taking the element-wise max over
        all skill weight vectors.

        Args:
            feature: [1, feature_dim] state + current-w embedding
            skill_weights: [K, n_objectives] tensor of skill weight vectors

        Returns:
            logits: [1, n_actions] GPI-improved scalarised Q-values
        """
        state_dim = self.model_config.mlp_hidden_size
        # state part of the feature (strip off the current w embedding)
        state = feature[:, :state_dim]  # [1, state_dim]

        action_size = (
            self.model_config.n_goals * self.model_config.n_topics
            if self.model_config.combined_action
            else self.model_config.n_goals
        )

        # Compute Q(s, a, w_k) for every skill weight w_k
        # skill_weights: [K, n_obj]
        k = skill_weights.size(0)
        w_embs = self.objective_embedding(skill_weights)    # [K, emb_size]
        state_rep = state.repeat(k, 1)                      # [K, state_dim]
        feat_k = torch.cat([state_rep, w_embs], dim=-1)    # [K, state_dim+emb_size]

        Q_all = self.actor(feat_k)                          # [K, n_actions*n_obj]
        Q_all = Q_all.view(k, action_size,
                           self.model_config.n_objectives)  # [K, n_actions, n_obj]

        # Scalarise with each skill's own weight
        skill_weights_exp = skill_weights.unsqueeze(1).expand_as(Q_all)  # [K, n_actions, n_obj]
        scalarised = (Q_all * skill_weights_exp).sum(-1)                   # [K, n_actions]

        # GPI: for each action, take the max over all K skills
        gpi_values, _ = scalarised.max(0)                  # [n_actions]
        return gpi_values.unsqueeze(0)                     # [1, n_actions]

    def get_skill_weight_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return [K, n_objectives] tensor from skill library, or None."""
        if self.skill_library is None:
            return None
        wvs = self.skill_library.all_weight_vectors()
        if not wvs:
            return None
        return torch.FloatTensor(wvs).to(device)
