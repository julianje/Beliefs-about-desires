from Belief import *
from Agent import *
from Observer import *

# Experiment 5

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
                Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [0, 1, 0, 1])

[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsA.ObserveAction(0)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsB.ObserveAction(0)

# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Experiment 5: Take the probability of each sample producing a choice change.
p_Naive = sum([Samples_Naive[i].ChoiceChange() * Probabilities_Naive[i] for i in range(len(Samples_Naive))])
p_Knowledgeable = sum([Samples_Knowledgeable[i].ChoiceChange() * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])

# Normalize
p_Naive /= sum(Probabilities_Naive) * 1.0
p_Knowledgeable /= sum(Probabilities_Knowledgeable) * 1.0

# Probability that knowledgeable agent kept choice and ignorant changed mind
p_correct = p_Naive * (1 - p_Knowledgeable)
# Probability that ignorant agent kept her choice
p_incorrect = p_Knowledgeable * (1 - p_Naive)

# Because some options aren't possible (both didn't change their minds of keep their choices) renormalize probabilities
Norm = p_correct + p_incorrect
p_correct /= Norm
p_incorrect /= Norm
# Model prediction: 1

# Experiment 6
# See code for Expts 1 and 2 to see how model predictions in 5 and 6 are equivalent.
