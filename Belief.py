# -*- coding: utf-8 -*-

"""
Simple generative model of an agent who maximizes expected utilities in 2AFC tasks.
"""

import random
import numpy as np


class Belief(object):

    def __init__(self, HypothesisSpace, Probabilities):
        """
        Belief class.

        Create an agent with beliefs about her costs and rewards.
        """
        if len(HypothesisSpace) != len(Probabilities):
            print "ERROR: Hypothesis space and probability vectors must match in length"
            return None
        self.HypothesisSpace = HypothesisSpace
        self.Probabilities = Probabilities
        self.Normalize()

    def Normalize(self):
        Norm = sum(self.Probabilities)
        if not Norm > 0:
            print "Can't normalize. Probabilities don't add up to a positive value"
            return None
        self.Probabilities = [i*1.0/Norm for i in self.Probabilities]

    def Integrate(self, ProbabilityVector):
        """
        Multiply belief by a new probablity vector.
        This let's you add the prior or likelihood.
        """
        if len(self.Probabilities) != len(ProbabilityVector):
            print "Integration vector doesn't have the right dimension."
            return None
        self.Probabilities = [self.Probabilities[i]*ProbabilityVector[i] for i in range(len(ProbabilityVector))]

    def HypothesisSpaceSize(self):
        return len(self.HypothesisSpace)

    def ExpectedValue(self):
        """
        Return expected value
        """
        return sum([self.HypothesisSpace[i]*self.Probabilities[i] for i in range(len(self.HypothesisSpace))])

    def ResetProbabilities(self):
        """
        Set a random belief distribution
        """
        self.Probabilities = [random.random() for i in range(len(self.Probabilities))]
        self.Normalize()

    def Display(self, Full=True):
        """
        Print object attributes.

        .. Warning::

           This function is for internal use only.

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
