from __future__ import division
import numpy as np
import kenlm
import math



class BeamEntry:
    """
    A single Beam: Class which is used to store character label sequences(path) and Beam probability
    """
    def __init__(self):
        self.total_score = 0.0 ## sum of no_blank_prob_score and blank_prob_score
        self.lm_score = 1.0 ## score of language model
        self.no_blank_score = 0.0 # prob score on the condition that this Beam ends with a no-blank character
        self.blank_score = 0.0 # prob_score on the condition that this Beam ends with a blank symbol
        self.labels = () # list of all characters which have been compressed by removing duplicates and blank
        self.LM = False # whether use LM to calculate score


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
        sorted_beams = sorted(beams, key=lambda x: x.total_score*x.lm_score, reverse=True)
        return [x.labels for x in sorted_beams]

    def sort_1(self):
        """
        Sort all BeamEntry according to overall possibility which is equal to BeamEntry.lm_score * BeamEntry.total_score
        :return: list(BeamEntry.labels)
        """
        beams = [V for(_, V) in self.entries.items()]
        sorted_prob = sorted(beams, key=lambda x: x.total_score, reverse=True)[:20]
        sorted_beams = sorted(sorted_prob, key=lambda x: x.lm_score, reverse=True)
        return [x.labels for x in sorted_beams]

    def sort_2(self):
        """
        Sort all BeamEntry according to overall possibility which is equal to BeamEntry.lm_score * BeamEntry.total_score
        :return: list(BeamEntry.labels)
        """
        beams = [V for(_, V) in self.entries.items()]
        sorted_lm = sorted(beams, key=lambda x: x.lm_score, reverse=True)[:20]
        sorted_beams = sorted(sorted_lm, key=lambda x: x.total_score, reverse=True)
        return [x.labels for x in sorted_beams]

    def sort_3(self):
        """
        Sort all BeamEntry according to overall possibility which is equal to BeamEntry.lm_score * BeamEntry.total_score
        :return: list(BeamEntry.labels)
        """
        beams = [V for(_, V) in self.entries.items()]
        sorted_prob = sorted(beams, key=lambda x: x.total_score, reverse=True)[:20]
        sorted_beams = sorted(sorted_prob, key=lambda x: x.total_score*x.lm_score, reverse=True)
        return [x.labels for x in sorted_beams]


    def sort_last(self):
        """
        Sort all BeamEntry according to overall possibility which is equal to BeamEntry.lm_score * BeamEntry.total_score
        :return: list(BeamEntry.labels)
        """
        beams = [V for(_, V) in self.entries.items()]
        sorted_beams_total = sorted(beams, key=lambda x: x.total_score, reverse=True)[0:10]
        sorted_beams_lm = sorted(sorted_beams_total, key=lambda x: x.lm_score, reverse=True)
        return [x.labels for x in sorted_beams_lm]


class CtcDecoder:
    def __init__(self):
        self.map_path = r"/home/augustus/Documents/decoder/decoder-python/data/ocr_char_set.txt"
        self.model = kenlm.Model(r"../data/mixed.bin")
        self.map = {}
        with open(self.map_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                result = line.split()
                try:
                    self.map[result[0]] = result[1]
                except IndexError:
                    self.map[result[0]] = ''

    def apply_lm(self, parent_beam: BeamEntry, child_beam: BeamEntry):
        """
        Calculate the LM score of children beam taken from parent beam and the bigram possibility of last two characters
        :param child_beam: beam of current time step
        :param parent_beam: beam of previous time step
        :return:None
        """
        lm_weights = 0.01
        if not child_beam.LM:
            char_1 = self.map[str(parent_beam.labels[-1]) if parent_beam.labels else list(self.map.keys())[-1]]
            char_2 = self.map[str(child_beam.labels[-1])]
            bi_gram = char_1 + char_2
            conditional_prob = math.pow(10, (self.model.score(bi_gram) - self.model.score(char_1)))**lm_weights
            print("unigram score " + char_1 + str(self.model.score(char_1)))
            print("bigram score " + char_2 + str(self.model.score(bi_gram)))
            child_beam.lm_score = parent_beam.lm_score*conditional_prob
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
            best_labels = last.sort_3()[0:beam_width]
            for labels in best_labels:
                no_blank_score = 0.0
                if labels:
                    no_blank_score = last.entries[labels].no_blank_score*prob_matrix[i, labels[-1]]
                    # print("if (!labels.empty()) no_blank_score" + str(no_blank_score))

                blank_score = last.entries[labels].total_score*prob_matrix[i, blank_idx]
                # print("blank_score " + str(blank_score))
                self.add_beam(curr, labels)
                # print("copy add beam")
                curr.entries[labels].labels = labels
                curr.entries[labels].no_blank_score += no_blank_score
                # print("curr.entries[labels].no_blank_score " + str(curr.entries[labels].no_blank_score))
                curr.entries[labels].blank_score += blank_score
                # print("curr.entries[labels].blank_score " + str(curr.entries[labels].blank_score))
                curr.entries[labels].total_score += (no_blank_score + blank_score)
                # print("curr.entries[labels].total_score " + str(curr.entries[labels].total_score) )
                curr.entries[labels].lm_score = last.entries[labels].lm_score
                # print("curr.entries[labels].lm_score " + str(curr.entries[labels].lm_score))
                curr.entries[labels].LM = True

                for j in range(dict_length-1):
                    new_labels = labels + (j, )
                    if labels and labels[-1] ==j:
                        no_blank_score = prob_matrix[i, j]*last.entries[labels].blank_score
                        # print("if (!labels.empty() && *(labels.rbegin()) == j) no_blank_score "+ str(no_blank_score))
                    else:
                        no_blank_score = prob_matrix[i, j]*last.entries[labels].total_score
                        # print("else no_blank_score" + str(no_blank_score))
                    self.add_beam(curr, new_labels)
                    curr.entries[new_labels].labels = new_labels
                    curr.entries[new_labels].no_blank_score += no_blank_score
                    # print("extend curr.entries[new_labels].no_blank_score" + str(curr.entries[new_labels].no_blank_score))
                    curr.entries[new_labels].total_score += no_blank_score
                    # print("exdtend curr.entries[new_labels].total_score" + str(curr.entries[new_labels].total_score))
                    self.apply_lm(curr.entries[labels], curr.entries[new_labels])
                    # print(new_labels)

            last = curr
        last.normalize()
        final_labels = last.sort()[0:beam_width]
        # final_entries = last.sort_last()[0:beam_width]
        res_set = []
        for labels in final_labels:
        # for entry in final_entries:
            res_str = ''
            # for loc in entry.labels:
            for loc in labels:
                # print("loc " + str(loc))
                # print("char "+self.map[str(loc)])
                res_str += self.map[str(loc)]
            # res_set.append((res_str, len(entry.labels), entry.total_score, entry.lm_score))
            res_set.append(res_str)
        return res_set

    def ctc_beam_search_decoder_practical(self, prob: list, index: list, T: int, K: int, A: int, beam_width: int, blank_index: int, default_blank_prob:float):
        """
        Detailed implement of prefix beam search decoder for ctc model
        :param: prob_matrix: Output result of CTC model,a numpy array of shape(t, len(word_map))
        :param: beam_width: Number of beams kept each iteration
        :return: A most suitable Beam at the end time step of sequence of prob matrix :list of characters
        """
        time_steps = T
        blank_idx = blank_index

        prob_matrix = list(dict())
        for i in range(time_steps):
            row = dict()
            for j in range(i*K, (i+1)*K):
                key = int(index[j])
                value = float(prob[j])
                row.update({key: value})
                # print("i {}, j {}, key {}, value {} ".format(i, j, key, row[key]))
            prob_matrix.append(row)

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
                no_blank_score = 0.0
                if labels and labels[-1] in prob_matrix[i].keys():
                    # print("last.entries[labels].no_blank_score{}, labels[-1]{}, prob_matrix[i][labels[-1]{}".format(last.entries[labels].no_blank_score, labels[-1], prob_matrix[i][labels[-1]]))
                    no_blank_score = last.entries[labels].no_blank_score*prob_matrix[i][labels[-1]]
                    # print("if (!labels.empty()) no_blank_score " + str(no_blank_score))
                if blank_idx in prob_matrix[i].keys():
                    blank_score = last.entries[labels].total_score*prob_matrix[i][blank_idx]
                    # print("in prob matrix, last.entries[labels].total_score{}, prob_matrix[i][blank_idx]{}".format(last.entries[labels].total_score, prob_matrix[i][blank_idx]))
                    # print("blank_score " + str(blank_score))
                else:
                    blank_score = last.entries[labels].total_score*default_blank_prob
                    # print("not in prob matrix, last.entries[labels].total_score{}, default_blank_prob{}".format(
                    #     last.entries[labels].total_score, default_blank_prob))
                    # print("blank_score " + str(blank_score))
                self.add_beam(curr, labels)
                # print("copy add beam")
                curr.entries[labels].labels = labels
                # print("BEFORE curr.entries[labels_str].no_blank_score{} ".format(curr.entries[labels].no_blank_score))
                curr.entries[labels].no_blank_score += no_blank_score
                # print("curr.entries[labels].no_blank_score " + str(curr.entries[labels].no_blank_score))
                curr.entries[labels].blank_score += blank_score
                # print("curr.entries[labels].blank_score " + str(curr.entries[labels].blank_score))
                curr.entries[labels].total_score += (no_blank_score + blank_score)
                # print("curr.entries[labels].total_score " + str(curr.entries[labels].total_score))
                curr.entries[labels].lm_score = last.entries[labels].lm_score
                # print("curr.entries[labels].lm_score " + str(curr.entries[labels].lm_score))
                curr.entries[labels].LM = True

                for j in prob_matrix[i].keys():
                    new_labels = labels + (j, )
                    if labels and labels[-1] ==j:
                        no_blank_score = prob_matrix[i][j]*last.entries[labels].blank_score
                        # print("if (!labels.empty() && *(labels.rbegin()) == j) no_blank_score " + str(no_blank_score))
                    else:
                        # print("j "+str(j))
                        no_blank_score = prob_matrix[i][j]*last.entries[labels].total_score
                        # print("prob_matrix[i][j] "+str(prob_matrix[i][j])+"last.entries[labels].total_score "+str(last.entries[labels].total_score))
                        # print("else no_blank_score " + str(no_blank_score))
                    self.add_beam(curr, new_labels)
                    curr.entries[new_labels].labels = new_labels
                    curr.entries[new_labels].no_blank_score += no_blank_score
                    # print("extend curr.entries[new_labels].no_blank_score " + str(curr.entries[new_labels].no_blank_score))
                    curr.entries[new_labels].total_score += no_blank_score
                    # print("exdtend curr.entries[new_labels].total_score " + str(curr.entries[new_labels].total_score))
                    self.apply_lm(curr.entries[labels], curr.entries[new_labels])
                    # print(new_labels)

            last = curr
        last.normalize()
        final_labels = last.sort()[0:beam_width]
        # final_entries = last.sort_last()[0:beam_width]
        res_set = []
        for labels in final_labels:
            res_str = ''
            for loc in labels:
                res_str += self.map[str(loc)]
            res_set.append(res_str)
        return res_set

def testBeamSearch():
    "test decoder"
    mat= np.load(r"/home/augustus/Documents/CTC-Decoder/data/nps/603.npy")
    print('Test beam search')
    decoder = CtcDecoder()
    actual = decoder.ctc_beam_search_decoder(mat,1)
    print('Actual: "' + actual + '"')


if __name__ == '__main__':
    testBeamSearch()


