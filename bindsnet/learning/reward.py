from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.nn import Module


class AbstractReward(ABC):
    # language=rst
    """
    Abstract base class for reward computation.
    """
    # def __init__(self):
    #     super.__init__(AbstractReward, self)

    @abstractmethod
    def compute(self, **kwargs) -> None:
        # language=rst
        """
        Computes/modifies reward.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per episode.
        """
        pass


class MovingAvgRPE(AbstractReward):
    # language=rst
    """
    Computes reward prediction error (RPE) based on an exponential moving average (EMA) of past rewards.
    """

    def __init__(self, **kwargs) -> None:
        # language=rst
        """
        Constructor for EMA reward prediction error.
        """
        self.reward_predict = torch.tensor(0.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(0.0)  # Predicted reward per episode.
        self.rewards_predict_episode = (
            []
        )  # List of predicted rewards per episode (used for plotting).

    def compute(self, **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the reward prediction error using EMA.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Current reward.
        :return: Reward prediction error.
        """
        # Get keyword arguments.
        reward = kwargs["reward"]

        return reward - self.reward_predict

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Updates the EMAs. Called once per episode.

        Keyword arguments:

        :param Union[float, torch.Tensor] accumulated_reward: Reward accumulated over one episode.
        :param int steps: Steps in that episode.
        :param float ema_window: Width of the averaging window.
        """
        # Get keyword arguments.
        accumulated_reward = kwargs["accumulated_reward"]
        steps = torch.tensor(kwargs["steps"]).float()
        ema_window = torch.tensor(kwargs.get("ema_window", 10.0))

        # Compute average reward per step.
        reward = accumulated_reward / steps

        # Update EMAs.
        self.reward_predict = (
            1 - 1 / ema_window
        ) * self.reward_predict + 1 / ema_window * reward
        self.reward_predict_episode = (
            1 - 1 / ema_window
        ) * self.reward_predict_episode + 1 / ema_window * accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())


class AutoEncoderLoss(AbstractReward):
    """Simply regresses voltage distribution of output neurons to be equal 
    to input intensity in input layer.
    """
    def __init__(self, **kwargs):
        self.epsilon = 0.2
        self.device = kwargs.get("device", torch.device("cpu"))
        self.network = kwargs.get("network", None)
    
    def compute(self, **kwargs) -> Dict:
        # Determine input intensity distribution
        input_distr = kwargs["input"]
        input_distr = input_distr.sum([0, 1, 2, 3]).float()
        input_distr = input_distr / input_distr.max()
        
        # Determine output voltage distribution
        output_distr = self.network.output.v
        output_distr = output_distr + output_distr.min()
        output_distr = output_distr / output_distr.max()
        
        # Determine rewards based on comparison of distributions
        elem_reward = (output_distr < input_distr + self.epsilon) \
            & (output_distr > input_distr - self.epsilon)
        elem_reward = elem_reward.float()
        elem_reward[elem_reward == 0] = -1
        kwargs["element_wise_reward"] = elem_reward
        kwargs["reward"] = elem_reward.sum()
        return kwargs
    
    def update(self, **kwargs) -> None:
        pass


class SNNCrossEntropyLoss(AbstractReward):
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", torch.device("cpu"))
        self.network = kwargs.get("network", None)
        self.prev_out_spikes = None
        self.t_thresh = 0
        self.t_window = 50
        self.target_rate = 10
        self.epsilon = 2  # Maximum allowed deviation from target value

    def compute(self, **kwargs):
        t = kwargs["t"]
        if self.prev_out_spikes is None:
            self.prev_out_spikes = torch.zeros(self.network.output.s.shape)

        # Set reward to 0 if time is below threshold
        if t < self.t_thresh:
            kwargs["element_wise_reward"] = torch.zeros_like(self.network.output.s).float()
            kwargs["reward"] = 0

        # Determine reward
        elif t >= self.t_thresh:
            label = kwargs.get("label", None)
            output_spikes = self.network.monitors["output_spikes"].get("s")
            window = output_spikes[:, max(0, t-self.t_window):t]
            out_sum = window.sum(1).float().unsqueeze(0)
            elem_reward = torch.zeros_like(out_sum)

            elem_reward = label * self.target_rate - out_sum
            elem_reward[elem_reward>=1] = 1
            elem_reward[elem_reward<=-1] = -1

            # Assign reward to kwargs
            kwargs["element_wise_reward"] = elem_reward
            kwargs["reward"] = elem_reward.sum()
        return kwargs

    def update(self, **kwargs):
        pass

    def reset_(self):
        self.prev_out_spikes = None
