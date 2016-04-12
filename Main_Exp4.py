from Belief import *
from Agent import *
from Observer import *

###############
# Experiment 4a
###############

# In Experiment 4s the reward don't matter as they're observable and matched
Costs = [0, 1, 2]
Rewards = [3]
PC = [1, 1, 1]
PR = [1]

# Create observer for a knowledgeable agent
# Second vector tells code to hold costs constant, but to let the reward's
# true values vary
ObsA = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 3),
                Belief(Costs, PC, 2), Belief(Rewards, PR, 3), [1, 1, 1, 1], [1, 1, 1, 1])

# Create observer for an agent who is ignorant about the costs, but leave
# costs fixed.
ObsB = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 3),
                Belief(Costs, PC, 2), Belief(Rewards, PR, 3), [0, 1, 0, 1], [1, 1, 1, 1])


# Hypothesis one: Knowledgeable agent chose 0 (good option) and ignorant
# agent chose 1.
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsA.ObserveAction(0)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsB.ObserveAction(1)

# Here the probability of action is simply the number of probability hits
# of the samples
p_Knowledgeable = sum(Probabilities_Knowledgeable) * \
    1.0 / len(Probabilities_Knowledgeable)
p_Naive = sum(Probabilities_Naive) * 1.0 / len(Probabilities_Naive)

p_H1 = p_Knowledgeable * p_Naive

# Hypothesis two: Knowledgeable agent chose 1 and ignorant agent chose 0.
# THIS CODE WILL THROW OUT A BUNCH OF WARNINGS BECAUSE THE KNOWLEDGEABLE AGENT IS
# NOT ACTING RATIONALLY. IGNORE THE WARNINGS, THE CODE STILL WORKS.
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsA.ObserveAction(1)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsB.ObserveAction(2)
# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Here the probability of action is simply the number of probability hits
# of the samples
p_Knowledgeable = sum(Probabilities_Knowledgeable) * \
    1.0 / len(Probabilities_Knowledgeable)
p_Naive = sum(Probabilities_Naive) * 1.0 / len(Probabilities_Naive)

p_H2 = p_Knowledgeable * p_Naive

# Normalize probabilities
Norm = p_H1 + p_H2
p_H1 /= Norm
p_H2 /= Norm
# Prediction = 1

###############
# Experiment 4b
###############


# In Experiment 4s the reward for the apple doesn't matter since the other door was reward 0,
# and both option have cost 0
Costs = [0, 1, 2]
Rewards = [0, 3]
PC = [1, 1, 1]
PR = [1, 1]

# Create observer for a knowledgeable agent
Obs_Knowledgeable = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 0),
                             Belief(Costs, PC, 0), Belief(Rewards, PR, 3), [1, 1, 1, 1], [1, 1, 1, 1])

# Create observer for an agent who is ignorant about the costs, but leave
# costs fixed.
Obs_Naive = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 0),
                     Belief(Costs, PC, 0), Belief(Rewards, PR, 3), [0, 0, 0, 0], [1, 1, 1, 1])


# Hypothesis one: Knowledgeable agent chose 0 (poor option) and ignorant
# agent chose 1 (good option).
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = Obs_Knowledgeable.ObserveAction(0)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = Obs_Naive.ObserveAction(1)

# Here the probability of action is simply the number of probability hits
# of the samples.
p_Knowledgeable = sum(Probabilities_Knowledgeable) * \
    1.0 / len(Probabilities_Knowledgeable)
p_Naive = sum(Probabilities_Naive) * 1.0 / len(Probabilities_Naive)

p_H1 = p_Knowledgeable * p_Naive

# Hypothesis two: Knowledgeable agent chose 1 and ignorant agent chose 0.
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = Obs_Knowledgeable.ObserveAction(1)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = Obs_Naive.ObserveAction(0)

# Here the probability of action is simply the number of probability hits
# of the samples
p_Knowledgeable = sum(Probabilities_Knowledgeable) * \
    1.0 / len(Probabilities_Knowledgeable)
p_Naive = sum(Probabilities_Naive) * 1.0 / len(Probabilities_Naive)

p_H2 = p_Knowledgeable * p_Naive

# Normalize probabilities
Norm = p_H1 + p_H2
p_H1 /= Norm
p_H2 /= Norm
# Prediction = 1
