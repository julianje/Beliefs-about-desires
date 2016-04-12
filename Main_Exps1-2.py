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
                Belief(Costs, PC, 0), Belief(Rewards, PR, 1), [0, 1, 0, 1])

[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsA.ObserveAction(0)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsB.ObserveAction(0)

# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Experiment 1: What is each agent's probability of getting a low reward?
# Both agents chose option 0 so we just need to get the probability that
# TrueValue for option 0 is negative.
p_Naive = sum([(1 if Samples_Naive[i].Rewards[0].TrueValue < 0 else 0)
               * Probabilities_Naive[i] for i in range(len(Samples_Naive))])
p_Knowledgeable = sum([(1 if Samples_Knowledgeable[i].Rewards[0].TrueValue < 0 else 0)
                       * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
# Normalize over samples with positive probability
p_Naive /= sum(Probabilities_Naive) * 1.0
p_Knowledgeable /= sum(Probabilities_Knowledgeable) * 1.0
# Probability that knowledgeable said "Yum" and naive said "Yuck"
p_correct = p_Naive * (1 - p_Knowledgeable)
# Probability that ignorant said "Yum" and knowledgeable said "Yuck"
p_incorrect = p_Knowledgeable * (1 - p_Naive)
# Because some options aren't possible (both didn't say yum and both
# didn't say yuck) renormalize probabilities
Norm = p_correct + p_incorrect
p_correct /= Norm
p_incorrect /= Norm
# Model prediction: 0.7521700164388274

# Experiment 2: Who was knowledgeable given that one got a positive utility and one got a negative one.
# For each agent compute the probability that each agent gets a positive
# utility and the probability that they get a negative utility conditioned
# on their choice.
# Should get the same model prediction as in Experiment 1.
p_Naive_Negative = sum([(1 if Samples_Naive[i].Rewards[0].TrueValue < 0 else 0)
                        * Probabilities_Naive[i] for i in range(len(Samples_Naive))])
p_Knowledgeable_Negative = sum([(1 if Samples_Knowledgeable[i].Rewards[
                               0].TrueValue < 0 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Positive = sum([(1 if Samples_Naive[i].Rewards[0].TrueValue > 0 else 0)
                        * Probabilities_Naive[i] for i in range(len(Samples_Naive))])
p_Knowledgeable_Positive = sum([(1 if Samples_Knowledgeable[i].Rewards[
                               0].TrueValue > 0 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
# Normalize probabilities of each agent.
NormNaive = p_Naive_Negative + p_Naive_Positive
NormKnowledge = p_Knowledgeable_Negative + p_Knowledgeable_Positive
p_Naive_Negative /= NormNaive * 1.0
p_Naive_Positive /= NormNaive * 1.0
p_Knowledgeable_Negative /= NormKnowledge * 1.0
p_Knowledgeable_Positive /= NormKnowledge * 1.0
# Now compute the probability that the one who said yum was the
# knowledgeable one and the one who said yuck was ignorant.
p_correct = p_Knowledgeable_Positive * p_Naive_Negative
# And the probability of the incorrect answer
p_incorrect = p_Naive_Positive * p_Knowledgeable_Negative
# Normalize over these two options
Norm = p_correct + p_incorrect
p_correct /= Norm
p_incorrect /= Norm
# Model prediction: 0.7521700164388275
