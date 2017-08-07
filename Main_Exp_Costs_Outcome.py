from Belief import *
from Agent import *
from Observer import *

# Given cost knowledge, who has a strong preference for tomatoes?

# Two options with cost.
Costs = [0, 1, 2]
Rewards = [1, 2, 3]
PC = [1, 1, 1]
PR = [1, 1, 1]

# Create observer for an agent who knows rewards but not costs.
# Second vector tells code to hold costs constant, but to let the reward's
# true values vary
ObsA = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                Belief(Costs, PC, 2), Belief(Rewards, PR, 1), [0, 1, 0, 1], [1, 0, 1, 0])

# Create observer for an agent who knows costs and rewards.
ObsB = Observer(Belief(Costs, PC, 0), Belief(Rewards, PR, 1),
                Belief(Costs, PC, 2), Belief(Rewards, PR, 1), [1, 1, 1, 1], [1, 0, 1, 0])

###############
# Experiment 3a
###############

# Both agents choose option 1 (the one with a cost)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsA.ObserveAction(1)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsB.ObserveAction(1)
# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Get probability that each agent has each positive reward.
p_Knowledgeable_Preference_1 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 1 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_1 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 1 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

p_Knowledgeable_Preference_2 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 2 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_2 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 2 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

p_Knowledgeable_Preference_3 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 3 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_3 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 3 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

# Build distributions of positive values
Knowledgeable_reward = [p_Knowledgeable_Preference_1,
                        p_Knowledgeable_Preference_2, p_Knowledgeable_Preference_3]
Naive_reward = [p_Naive_Preference_1,
                p_Naive_Preference_2, p_Naive_Preference_3]
# Normalize
Know_Norm = sum(Knowledgeable_reward) * 1.0
Naive_Norm = sum(Naive_reward) * 1.0
# Note again that these are probability distributions over reward values
Knowledgeable_reward = [i / Know_Norm for i in Knowledgeable_reward]
Naive_reward = [i / Naive_Norm for i in Naive_reward]
# Get probability that Knowledgeable has a higher reward than the ignorant
# agent

p = 0
for i in range(len(Knowledgeable_reward)):
    # Sum the probabilities of smaller rewards
    NaiveProbs = 0
    # Probability that Naive agent has a lower reward
    for j in range(i):
        NaiveProbs += Naive_reward[j]
    # Split probability that agents have equal rewards (since it predicts
    # chance performance)
    NaiveProbs += Naive_reward[i] / 2.0
    # Add this probability to p, weighted by the probability that
    # Knowledgeable agent surpases all these probs
    p += NaiveProbs * Knowledgeable_reward[i]

p
# Model prediction
# 0.7375299281723863

###############
# Experiment 3b
###############

# Identical to code above, tu agents now choose 0 (low-cost)

# Both agents choose option 1 (the one with a cost)
[LAgent_Naive, Samples_Naive, Probabilities_Naive] = ObsA.ObserveAction(0)
[LAgent_Knowledgeable, Samples_Knowledgeable,
    Probabilities_Knowledgeable] = ObsB.ObserveAction(0)
# Integrate likelihoods with priors.
ObsA.BuildPosterior(LAgent_Naive)
ObsB.BuildPosterior(LAgent_Knowledgeable)

# Get probability that each agent has each positive reward.
p_Knowledgeable_Preference_1 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 1 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_1 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 1 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

p_Knowledgeable_Preference_2 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 2 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_2 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 2 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

p_Knowledgeable_Preference_3 = sum([(1 if Samples_Knowledgeable[i].Rewards[
                                   1].TrueValue == 3 else 0) * Probabilities_Knowledgeable[i] for i in range(len(Samples_Knowledgeable))])
p_Naive_Preference_3 = sum([(1 if Samples_Naive[i].Rewards[1].TrueValue == 3 else 0)
                            * Probabilities_Naive[i] for i in range(len(Samples_Knowledgeable))])

# Build distributions of positive values
Knowledgeable_reward = [p_Knowledgeable_Preference_1,
                        p_Knowledgeable_Preference_2, p_Knowledgeable_Preference_3]
Naive_reward = [p_Naive_Preference_1,
                p_Naive_Preference_2, p_Naive_Preference_3]
# Normalize
Know_Norm = sum(Knowledgeable_reward) * 1.0
Naive_Norm = sum(Naive_reward) * 1.0
# Note again that these are probability distributions over reward values
Knowledgeable_reward = [i / Know_Norm for i in Knowledgeable_reward]
Naive_reward = [i / Naive_Norm for i in Naive_reward]
# Get probability that Knowledgeable has a higher reward than the ignorant
# agent

p = 0
for i in range(len(Knowledgeable_reward)):
    # Sum the probabilities of smaller rewards
    NaiveProbs = 0
    # Probability that Naive agent has a lower reward
    for j in range(i):
        NaiveProbs += Naive_reward[j]
    # Split probability that agents have equal rewards (since it predicts
    # chance performance)
    NaiveProbs += Naive_reward[i] / 2.0
    # Add this probability to p, weighted by the probability that
    # Knowledgeable agent surpases all these probs
    p += NaiveProbs * Knowledgeable_reward[i]

p
# Model prediction
# 0.6010598418787185

