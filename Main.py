from Belief import *
from Agent import *
from Observer import *

# Two options and neither have a cost.
Costs = [0]
Rewards = [1, 2, 3]
PC = [1]
PR = [1, 1, 1]

Cost_BeliefsA = Belief(Costs, PC)
Reward_BeliefsA = Belief(Rewards, PR)

Cost_BeliefsB = Belief(Costs, PC)
Reward_BeliefsB = Belief(Rewards, PR)

Obs = Observer(Cost_BeliefsA, Reward_BeliefsA, Cost_BeliefsB, Reward_BeliefsB)

LAgent = Obs.ObserveAction(1)

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

