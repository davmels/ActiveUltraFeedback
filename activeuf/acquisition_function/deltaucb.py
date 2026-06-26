import torch

from activeuf.acquisition_function.base import (
    BaseAcquisitionFunction,
    quota_constrained_topk,
)


class DeltaUCB(BaseAcquisitionFunction):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__()
        self.beta = beta

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        K: int = None,
        domain_labels: list | None = None,
        quotas: dict | None = None,
        rng=None,
    ) -> list[list[int, int]] | tuple[list[int], list[list[int, int]]]:
        n_prompts, n_completions = upper_bounds.shape

        # Shape: (n_prompts, n_completions, n_completions)
        upper_confidence_bounds = upper_bounds.unsqueeze(2) - lower_bounds.unsqueeze(1)

        # Mask out diagonal to avoid choosing the same completion twice
        diag_mask = torch.eye(
            n_completions, device=upper_confidence_bounds.device, dtype=torch.bool
        ).unsqueeze(0)
        upper_confidence_bounds.masked_fill_(diag_mask, -torch.inf)

        # Flatten the n_completions x n_completions matrices
        upper_confidence_bounds_flattened = upper_confidence_bounds.view(n_prompts, -1)

        # Completion selection: argmax pair per prompt (computed for all prompts)
        max_confidence_gap_flat_idx = upper_confidence_bounds_flattened.argmax(dim=1)
        first_idxs = max_confidence_gap_flat_idx // n_completions
        second_idxs = max_confidence_gap_flat_idx % n_completions
        completion_pairs = list(zip(first_idxs.tolist(), second_idxs.tolist()))

        # Prompt selection: score by max pairwise (UCB - LCB) gap
        if K is not None and K < n_prompts:
            prompt_scores = upper_confidence_bounds_flattened.max(dim=1).values
            if domain_labels is not None and quotas is not None:
                # Quota-constrained top-K: cap each domain at quotas[domain] on a first
                # ranked pass, then random-fill any shortfall from the skipped prompts.
                selected_prompt_indices = quota_constrained_topk(
                    prompt_scores, K, domain_labels, quotas, rng=rng
                )
            else:
                selected_prompt_indices = prompt_scores.topk(K).indices.sort().values.tolist()
            selected_pairs = [completion_pairs[i] for i in selected_prompt_indices]
            return selected_prompt_indices, selected_pairs

        return completion_pairs
