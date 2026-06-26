import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class DoubleReverseThompsonSampling(BaseAcquisitionFunction):
    def __init__(self, max_iterations: int = 30, beta: int = 1, **kwargs):
        super().__init__()
        self.max_iterations = max_iterations
        self.beta = beta

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        K: int = None,
    ) -> list[list[int, int]] | tuple[list[int], list[list[int, int]]]:
        """
        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
            K: if set, select K prompts with largest sampled reward spread
        """
        # even if upper and lower bounds are not symmetric, our current implementation is such, that we can assume they are symmetric.
        std_deviation = (upper_bounds - lower_bounds) / 2

        # Prompt selection: sample from posterior, score by max - min
        if K is not None and K < len(rewards):
            z = torch.from_numpy(np.random.uniform(-1, 1, size=rewards.shape)).float()
            sampled_rewards = rewards + self.beta * z * std_deviation
            prompt_scores = sampled_rewards.max(dim=1).values - sampled_rewards.min(dim=1).values
            selected_prompt_indices = prompt_scores.topk(K).indices.sort().values.tolist()

            rewards = rewards[selected_prompt_indices]
            std_deviation = std_deviation[selected_prompt_indices]
        else:
            selected_prompt_indices = None

        selected_ids_batch = []
        for i in range(len(rewards)):
            # step 1 - selecting first response
            response_1 = self.dts_optimize(rewards[i], std_deviation[i])

            # step 2 - selecting second response
            response_2 = response_1
            iterations = 0
            while response_1 == response_2:
                if iterations == self.max_iterations:
                    response_2 = np.random.randint(0, len(rewards[i]))
                else:
                    response_2 = self.dts_reverse_optimize(rewards[i], std_deviation[i])
                    iterations += 1

            selected_ids_batch.append((response_1, response_2))

        if selected_prompt_indices is not None:
            return selected_prompt_indices, selected_ids_batch
        return selected_ids_batch

    def dts_optimize(self, reward_list, std_deviation_list):
        r_epistemic_index = []
        for j in range(len(reward_list)):
            z = np.random.uniform(-1, 1)

            r_x_y_epistemic_index = (
                reward_list[j] + self.beta * z * std_deviation_list[j]
            )
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax([idx.cpu() for idx in r_epistemic_index])

    def dts_reverse_optimize(self, reward_list, std_deviation_list):
        # negating all the reward estimates and passing them to dts_optimize
        negated_reward_list = [-r for r in reward_list]
        return self.dts_optimize(negated_reward_list, std_deviation_list)
