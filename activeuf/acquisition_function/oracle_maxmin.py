import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class OracleMaxMin(BaseAcquisitionFunction):
    """Select best and worst completions per prompt using reward scores directly.

    For prompt selection with K, selects prompts with the largest score gap first.
    This serves as an oracle baseline showing the upper bound of what maxmin-style
    acquisition can achieve with perfect knowledge of the scores.
    """

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        K: int = None,
    ) -> list[list[int, int]] | tuple[list[int], list[list[int, int]]]:
        n_prompts, n_completions = rewards.shape

        valid_mask = rewards.isfinite()
        rewards_for_min = rewards.clone()
        rewards_for_min[~valid_mask] = float("inf")

        best_idxs = rewards.argmax(dim=1)
        worst_idxs = rewards_for_min.argmin(dim=1)

        for i in range(n_prompts):
            if best_idxs[i] == worst_idxs[i]:
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) >= 2:
                    best_idxs[i] = valid_indices[0]
                    worst_idxs[i] = valid_indices[1]

        gaps = rewards.max(dim=1).values - rewards_for_min.min(dim=1).values

        completion_pairs = list(zip(best_idxs.tolist(), worst_idxs.tolist()))

        if K is not None and K < n_prompts:
            selected_prompt_indices = gaps.topk(K).indices.sort().values.tolist()
            selected_pairs = [completion_pairs[i] for i in selected_prompt_indices]
            return selected_prompt_indices, selected_pairs

        return completion_pairs
