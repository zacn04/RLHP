from __future__ import annotations

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


class RenormalizingMaskableMultiInputActorCriticPolicy(MaskableMultiInputActorCriticPolicy):
    """Policy that renormalizes action probabilities to avoid numerical issues."""

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskableCategorical:
        dist = super()._get_action_dist_from_latent(latent_pi)
        p = dist.distribution.probs
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        dist.distribution.probs = p
        return dist
