from __future__ import division
import numpy as np
import kenlm



class BeamEntry:
    """
    A single Beam: Class which is used to store character label sequences(path) and Beam probability
    """
    def __init__(self):
        self.total_score = 0 ## sum of no_blank_prob_score and blank_prob_score
        #self.lm_score = 0 ## score of language model
        self.no_blank_score = 0 # prob score on the condition that this Beam ends with a no-blank character
        self.blank_score = 0 # prob_score on the condition that this Beam ends with a blank symbol
        self.labels = () # list of all characters which have been compressed by removing duplicates and blank
        #self.LM = False # whether use LM to calculate score


class BeamState:
    """
    Beams container:Class which Stores all candidate BeamEntry objects and provides sort and normalization interface
    """
    def __init__(self):
        self.entries = {}

    def normalize(self):
        """
        Normalize prob_score of each BeamEntry in entries by number of label in each BeamEntry.labels
        :return: None
        """
        for (K, V) in self.entries.items():
            label_number = len(self.entries[K].labels)
            self.entries[K].lm_score **= (1.0/(label_number if label_number else 1.0))

    def sort(self):
        """
        Sort all BeamEntry according to overall possibility which is equal to BeamEntry.lm_score * BeamEntry.total_score
        :return: list(BeamEntry.labels)
        """
        beams = [V for(_, V) in self.entries.items()]
        sorted_beams = sorted(beams, key=lambda x: x.total_score, reverse=True)
        return [x.labels for x in sorted_beams]


class CtcDecoder:
    def __init__(self):
        self.map_path = r"../data/ocr_char_set.txt"
        #self.model = kenlm.Model(r"../data/train.bin")
        self.map = {}
        with open(self.map_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                result = line.split()
                try:
                    self.map[result[0]] = result[1]
                except IndexError:
                    self.map[result[0]] = ''

    def apply_lm(self, child_beam: BeamEntry, parent_beam: BeamEntry):
        """
        Calculate the LM score of children beam taken from parent beam and the bigram possibility of last two characters
        :param child_beam: beam of current time step
        :param parent_beam: beam of previous time step
        :return:None
        """
        lm_weights = 0.01
        if not child_beam.LM:
            char_1 = self.map[parent_beam.labels[-1]]
            char_2 = self.map[child_beam.labels[-1]]
            bi_gram = char_1 + char_2
            conditional_prob = (self.model.score(bi_gram) - self.model.score(char_1))*lm_weights
            child_beam.lm_score = parent_beam.lm_score + conditional_prob
            child_beam.LM = True

    def add_beam(self, state: BeamState, labels: list):
        """
        Add a new BeamEntry object to BeamState object
        :param labels: labels of a Beam to be added to BeamState
        :param state: Existing BeamState objectjj
        :return: None
        """
        if labels not in state.entries:
            state.entries[labels] = BeamEntry()

    def ctc_beam_search_decoder(self, prob_matrix: np.array, beam_width=2):
        """
        Detailed implement of prefix beam search decoder for ctc model
        :param: prob_matrix: Output result of CTC model,a numpy array of shape(t, len(word_map))
        :param: beam_width: Number of beams kept each iteration
        :return: A most suitable Beam at the end time step of sequence of prob matrix :list of characters
        """
        time_steps, dict_length = prob_matrix.shape
        blank_idx = dict_length-1

        # Initialize BeamState
        last = BeamState()
        labels = ()
        last.entries[labels] = BeamEntry()
        last.entries[labels].total_score = 1
        last.entries[labels].blank_score = 1

        for i in range(time_steps):
            curr = BeamState()
            best_labels = last.sort()[0:beam_width]
            for labels in best_labels:
                no_blank_score = 0
                if labels:
                    no_blank_score = last.entries[labels].no_blank_score*prob_matrix[i, labels[-1]]

                blank_score = last.entries[labels].total_score*prob_matrix[i, blank_idx]
                self.add_beam(curr, labels)

                curr.entries[labels].labels = labels
                curr.entries[labels].no_blank_score += no_blank_score
                curr.entries[labels].blank_score += blank_score
                curr.entries[labels].total_score += (no_blank_score + blank_score)
                #print(labels)
                #curr.entries[labels].lm_score = last.entries[labels].lm_score
                #curr.entries[labels].LM = True

                for j in range(dict_length-1):
                    new_labels = labels + (j, )
                    if labels and labels[-1] ==j:
                        no_blank_score = prob_matrix[i, j]*last.entries[labels].blank_score
                    else:
                        no_blank_score = prob_matrix[i, j]*last.entries[labels].total_score
                    self.add_beam(curr, new_labels)
                    #print(new_labels)
                    curr.entries[new_labels].labels = new_labels
                    curr.entries[new_labels].no_blank_score += no_blank_score
                    curr.entries[new_labels].total_score += no_blank_score
                    #self.apply_lm(curr.entries[labels], curr.entries[new_labels])

            last = curr
        #last.normalize()
        final_labels_set = last.sort()[0:beam_width]
        res_set = []
        for final_labels in final_labels_set:
            res_str = ''
            for loc in final_labels:
                res_str += self.map[str(loc)]
            res_set.append(res_str)
        return res_set


def testBeamSearch():
    "test decoder"
    mat= np.load(r"/home/augustus/Documents/CTC-Decoder/data/test_nps/38.npy")
    print('Test beam search')
    decoder = CtcDecoder()
    actual = decoder.ctc_beam_search_decoder(mat,1)
    print('Actual: "' + actual + '"')


if __name__ == '__main__':
    testBeamSearch()


