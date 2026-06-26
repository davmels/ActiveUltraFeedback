from abc import ABC, abstractmethod
import random as _random

import torch


def quota_constrained_topk(scores, K, domain_labels, quotas, rng=None):
    """Select K indices from `scores` (higher is better) honoring per-domain quotas.

    Walk the prompts in descending score order, taking a prompt only while its domain
    is still under quota (``quotas[domain]``, default 0); over-quota prompts are skipped.
    If fewer than K prompts are taken after the full pass, random-fill the shortfall from
    the skipped (over-quota) prompts.

    Args:
        scores: 1-D tensor of per-prompt scores (length n).
        K: number of prompts to select.
        domain_labels: list of length n mapping prompt index -> domain key.
        quotas: dict domain key -> max prompts allowed on the first pass.
        rng: object with a .sample(population, k) method (defaults to the `random` module,
            so the global RNG state checkpointed by the loop is reused).

    Returns:
        list[int] of selected indices (length min(K, n)), in selection order
        (ranked picks first, then any random fill).
    """
    rng = rng or _random
    ranking = torch.argsort(scores, descending=True).tolist()

    counts = {}
    selected = []
    skipped = []
    for idx in ranking:
        d = domain_labels[idx]
        if counts.get(d, 0) < quotas.get(d, 0):
            selected.append(idx)
            counts[d] = counts.get(d, 0) + 1
            if len(selected) >= K:
                break
        else:
            skipped.append(idx)

    if len(selected) < K and skipped:
        need = K - len(selected)
        selected.extend(rng.sample(skipped, min(need, len(skipped))))
    return selected


class BaseAcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[list[int, int]] | tuple[list[int], list[list[int, int]]]:
        """
        Given information on the completions for a batch of prompts, selects
        the indices for the two completions per prompt that should be annotated
        by the oracle.

        Args:
            Blank, because it can vary across the child classes.
            K (int, optional): If provided, select only K prompts from the
                batch before choosing completion pairs. Returns a tuple of
                (selected_prompt_indices, completion_pairs).

        Returns:
            list[list[int, int]]: The selected completion indices per prompt
                (when K is None).
            OR
            tuple[list[int], list[list[int, int]]]: A tuple of
                (selected_prompt_indices, completion_pairs) when K is set.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
