import numpy as np
import os, sys
src_path = os.path.abspath(os.path.join('..'))
sys.path.append(src_path)
from src.PrefixBeamSearch import CtcDecoder
from src.Editdistance import edit_distance
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

if __name__ == "__main__":

    consult_dict = {}
    with open("../data/list.txt", 'r') as f:
        for line in f:
            results = line.split(r"$$$")
            name = results[0].split(".")[0]
            greedy_result = results[1]
            ground_truth = results[2]
            consult_dict[name] = (greedy_result, ground_truth)

    paths = [os.path.join("../data/nps", relative_path) for relative_path in os.listdir("../data/nps")]

    def decoder_wrapper(path):
        decoder_lm = CtcDecoder()
        name = path.split("/")[-1]
        print("Decoding {}".format(name))
        data = np.load(path)
        result_lm = (decoder_lm.ctc_beam_search_decoder(data, beam_width=3))[0]
        result_greedy = consult_dict[name.split(".")[0]][0]
        result_truth = consult_dict[name.split(".")[0]][1]
        print("{} lm result:{}, greedy result:{}, truth:{}".format(name, result_lm, result_greedy, result_truth))
        lm_edit_distance = edit_distance(result_lm, result_truth)
        greedy_edit_distance = edit_distance(result_greedy, result_truth)
        lm_edit_precision = float(lm_edit_distance) / len(result_truth)
        greedy_edit_precision = float(greedy_edit_distance) / len(result_truth)
        print("{} lm_precision {}, greedy precsiion {}".format(name, lm_edit_precision, greedy_edit_precision))
        if result_lm == result_truth and result_greedy == result_truth:
            return 1, 1, lm_edit_precision, greedy_edit_precision
        elif result_lm == result_truth and result_greedy != result_truth:
            return 1, 0, lm_edit_precision, greedy_edit_precision
        elif result_lm != result_truth and result_greedy == result_truth:
            return 0, 1, lm_edit_precision, greedy_edit_precision
        else:
            return 0, 0, lm_edit_precision, greedy_edit_precision

    pool = ThreadPool(4)
    results = pool.map(decoder_wrapper, paths)
    pool.close()
    pool.join()
    print("Use python console to calculate the statistic info")