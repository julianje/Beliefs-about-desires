# -*- coding: utf-8 -*-

"""
Simple generative model of an agent who maximizes expected utilities in 2AFC tasks.
"""

import numpy as np


class Agent(object):

    def __init__(self, CostBeliefA, RewardBeliefA, CostBeliefB, RewardBeliefB):
        """
        Agent class.

        Create an agent with beliefs about her costs and rewards.
        """
        self.Costs = [CostBeliefA, CostBeliefB]
        self.Rewards = [RewardBeliefA, RewardBeliefB]

    def ChoiceProb(self, choice):
        UtilityA = self.Rewards[0].ExpectedValue(
        ) - self.Costs[0].ExpectedValue()
        UtilityB = self.Rewards[1].ExpectedValue(
        ) - self.Costs[1].ExpectedValue()
        if UtilityA > UtilityB:
            return 1 if choice == 0 else 0
        if UtilityA < UtilityB:
            return 1 if choice == 1 else 0
        return 0.5

    def MakeChoice(self):
        UtilityA = self.Rewards[0].ExpectedValue(
        ) - self.Costs[0].ExpectedValue()
        UtilityB = self.Rewards[1].ExpectedValue(
        ) - self.Costs[1].ExpectedValue()
        return np.argmax([UtilityA, UtilityB])

    def ResampleBeliefs(self):
        """
        Generate a random belief distribution for each dimension
        """
        self.Costs[0].ResetProbabilities()
        self.Costs[1].ResetProbabilities()
        self.Rewards[0].ResetProbabilities()
        self.Rewards[1].ResetProbabilities()

    def Normalize(self):
        """
        Normalize all probability distributions
        """
        self.Costs[0].Normalize()
        self.Costs[1].Normalize()
        self.Rewards[0].Normalize()
        self.Rewards[1].Normalize()

    def UpdateBeliefs(self, NewBCostA, NewBRewardA, NewBCostB, NewBRewardB):
        """
        Push new belief vectors onto agent's beliefs
        """
        self.Costs[0].Probabilities = NewBCostA
        self.Costs[1].Probabilities = NewBCostB
        self.Rewards[0].Probabilities = NewBRewardA
        self.Rewards[1].Probabilities = NewBRewardB

    def Display(self, Full=True):
        """
        Print object attributes.

        Args:
            Full (bool): When set to False, function only prints attribute names. Otherwise, it also prints its values.

        Returns:
            standard output summary
        """
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property
