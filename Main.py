from Belief import *
from Agent import *
from Observer import *

# Experiments 1 and 2

# Two options and neither have a cost.
Costs = [0]
Rewards = [-3, -2, -1, 1, 2, 3]
PC = [1]
PR = [1, 1, 1, 1, 1, 1]

# Create observer for naive agent
ObsA = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [0, 0, 0, 0])

# Create observer for knowledgeable agent.
ObsB = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [0, 0, 1, 1])

[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsA.ObserveAction(1)
[LAgent_Knowledgeable, Samples_Knowledgeable, Probabilities_Knowledgeable] = ObsB.ObserveAction(1)

# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Two options with costs
Costs = [0, 1, 2]
Rewards = [1, 2, 3]
P = [1, 1, 1]

Cost_BeliefsA = Belief(Costs, P)
Reward_BeliefsA = Belief(Rewards, P)

Cost_BeliefsB = Belief(Costs, P)
Reward_BeliefsB = Belief(Rewards, P)

Obs = Observer(Cost_BeliefsA, Reward_BeliefsA, Cost_BeliefsB, Reward_BeliefsB)

LAgent = Obs.ObserveAction(1)

####

from Belief import *
from Agent import *
from Observer import *
