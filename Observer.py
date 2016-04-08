# -*- coding: utf-8 -*-

"""
Simple observer.
"""

import copy
from Agent import *


class Observer(object):

    def __init__(self, CostPriorA, RewardPriorA, CostPriorB, RewardPriorB):
        self.CostPriors = [CostPriorA, CostPriorB]
        self.RewardPriors = [RewardPriorA, RewardPriorB]
        # Create a model of an agent with the priors.
        self.CostPriorA = CostPriorA
        self.CostPriorB = CostPriorB
        self.RewardPriorA = RewardPriorA
        self.RewardPriorB = RewardPriorB

    def BuildPosterior(self, Likelihood):
        """
        Take an agent object with the likelihoods and integrate the observer's priors.
        """
        Likelihood.Costs[0].Integrate(CostPriorA)
        Likelihood.Costs[1].Integrate(CostPriorB)
        Likelihood.Rewards[0].Integrate(RewardPriorA)
        Likelihood.Rewards[1].Integrate(RewardPriorB)
        Likelihood.Normalize()

    def ObserveAction(self, choice, samples=10000):
        """
        See an observed choice and infer the agents' costs and rewards
        prior to making the choice
        """
        # Generate a bunch of agents!
        Agents = []
        probs = []
        for i in range(samples):
            Agents.append(Agent(copy.deepcopy(self.CostPriorA), copy.deepcopy(self.RewardPriorA),
                                copy.deepcopy(self.CostPriorB), copy.deepcopy(self.RewardPriorB)))
            Agents[i].ResampleBeliefs()
            probs.append(Agents[i].ChoiceProb(choice))
        # Now create an agent with the likelihood!
        # To use as a skeleton.
        LAgent = Agent(self.CostPriorA, self.RewardPriorA,
                       self.CostPriorB, self.RewardPriorB)
        # You need four sets of probabilities since it's a 2AFC and each option
        # has costs and rewards.
        P_CostA = [0] * self.CostPriorA.HypothesisSpaceSize()
        P_CostB = [0] * self.CostPriorB.HypothesisSpaceSize()
        P_RewardA = [0] * self.RewardPriorA.HypothesisSpaceSize()
        P_RewardB = [0] * self.RewardPriorB.HypothesisSpaceSize()
        for i in range(samples):
            # Get sample weighted by it's probability.
            E_CostA = [j * probs[i] for j in Agents[i].Costs[0].Probabilities]
            E_RewardA = [j * probs[i]
                         for j in Agents[i].Rewards[0].Probabilities]
            E_CostB = [j * probs[i] for j in Agents[i].Costs[1].Probabilities]
            E_RewardB = [j * probs[i]
                         for j in Agents[i].Rewards[1].Probabilities]
            # Add it to the likelihoods
            P_CostA = [P_CostA[j] + E_CostA[j] for j in range(len(E_CostA))]
            P_CostB = [P_CostB[j] + E_CostB[j] for j in range(len(E_CostB))]
            P_RewardA = [P_RewardA[j] + E_RewardA[j]
                         for j in range(len(E_RewardA))]
            P_RewardB = [P_RewardB[j] + E_RewardB[j]
                         for j in range(len(E_RewardB))]
        # Now update LAgent object
        LAgent.UpdateBeliefs(P_CostA, P_RewardA, P_CostB, P_RewardB)
        LAgent.Normalize()
        return LAgent
