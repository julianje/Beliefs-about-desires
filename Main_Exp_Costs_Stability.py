from Belief import *
from Agent import *
from Observer import *

# Experiment 7a

# Two options and neither have a cost.
Costs = [0, 1, 2]
Rewards = [-3, -2, -1, 1, 2, 3]
PC = [1, 1, 1]
PR = [1, 1, 1, 1, 1, 1]

# Create observer for naive agent who know what she likes.
# Fix the cost values and let rewards fluctuate across agents.
Obs_Naive = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                     Belief(Costs, PC, 2), Belief(Rewards, PR, 1), [0, 1, 0, 1], [1, 0, 1, 0])

# Create observer for knowledgeable agent who knows both costs and rewards.
# Fix costs and let rewards fluctuate
Obs_Knowledgeable = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                             Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [1, 1, 1, 1], [1, 0, 1, 0])

[LAgent_Naive, Samples_Naive, Probabilities_Naive] = Obs_Naive.ObserveAction(1)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = Obs_Knowledgeable.ObserveAction(1)

# Take the probability of each sample producing a choice change.
p_Naive = sum([Samples_Naive[i].ChoiceChange() * Probabilities_Naive[i]
               for i in range(len(Samples_Naive))])
p_Knowledgeable = sum([Samples_Knowledgeable[i].ChoiceChange(
) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])

# Normalize
p_Naive /= sum(Probabilities_Naive) * 1.0
p_Knowledgeable /= sum(Probabilities_Knowledgeable) * 1.0

# Probability that knowledgeable agent kept choice and ignorant changed mind
p_correct = p_Naive * (1 - p_Knowledgeable)
# Probability that ignorant agent kept her choice
p_incorrect = p_Knowledgeable * (1 - p_Naive)

# Because some options aren't possible (both didn't change their minds of
# keep their choices) renormalize probabilities
Norm = p_correct + p_incorrect
p_correct /= Norm
p_incorrect /= Norm
# Model prediction: 1

# Experiment 7b

# Two options and neither have a cost.
Costs = [0, 1, 2]
Rewards = [-3, -2, -1, 1, 2, 3]
PC = [1, 1, 1]
PR = [1, 1, 1, 1, 1, 1]

# Create observer for naive agent who know what she likes.
# Fix the cost values and let rewards fluctuate across agents.
Obs_Naive = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                     Belief(Costs, PC, 2), Belief(Rewards, PR, 1), [0, 1, 0, 1], [1, 0, 1, 0])

# Create observer for knowledgeable agent who knows both costs and rewards.
# Fix costs and let rewards fluctuate
Obs_Knowledgeable = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                             Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [1, 1, 1, 1], [1, 0, 1, 0])

[LAgent_Naive, Samples_Naive, Probabilities_Naive] = Obs_Naive.ObserveAction(0)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = Obs_Knowledgeable.ObserveAction(0)

# Take the probability of each sample producing a choice change.
p_Naive = sum([Samples_Naive[i].ChoiceChange() * Probabilities_Naive[i]
               for i in range(len(Samples_Naive))])
p_Knowledgeable = sum([Samples_Knowledgeable[i].ChoiceChange(
) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])

# Normalize
p_Naive /= sum(Probabilities_Naive) * 1.0
p_Knowledgeable /= sum(Probabilities_Knowledgeable) * 1.0

# Probability that knowledgeable agent kept choice and ignorant changed mind
p_correct = p_Naive * (1 - p_Knowledgeable)
# Probability that ignorant agent kept her choice
p_incorrect = p_Knowledgeable * (1 - p_Naive)

# Because some options aren't possible (both didn't change their minds of
# keep their choices) renormalize probabilities
Norm = p_correct + p_incorrect
if Norm > 0:
    p_correct /= Norm
    p_incorrect /= Norm
else:
    # Means generative model couldn't find any instances where
    # either agent would change her mind if maximizing expected utilities
    p_correct = 0.5
    p_incorrect = 0.5
# Model prediction: 0.5
