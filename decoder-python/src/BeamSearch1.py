"""
Given a beam width B, search best B results according to a comparator
"""

import numpy as np
import copy
import math


class BeamSearch:
    def __init__(self, B):
        self.B = B

    def BeamSearch(self, data: np.array)->list:
        '''
        Implement of Beam Search algorithm
        :param data: numpy array of softmax probability of each step
        :return: a string chosen of highest conditional probability
        '''
        sequence_prob = [[[], 1.0]]
        for row in data:
            candidate = []
            for item in sequence_prob:
                sequence, prob = item
                for index, prob in enumerate(row):
                    candidate.append([sequence+[index], prob*(-math.log(prob))])
            ordered_candidate = sorted(candidate, key=lambda tup: tup[1])
            sequence_prob = ordered_candidate[:self.B]
        return sequence_prob






