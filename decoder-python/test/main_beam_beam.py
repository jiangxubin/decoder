import numpy as np
import sys,  os
# decoder_path = os.path.abspath(os.path.join('...'))
# sys.path.append(decoder_path)
src_path = os.path.abspath(os.path.join('..'))
sys.path.append(src_path)
from src.PrefixBeamSearch_2 import CtcDecoder
from src.Editdistance import edit_distance


if __name__ == "__main__":
    decoder_lm = CtcDecoder()

    consult_dict = {}
    with open("../data/list.txt", 'r') as f:
        for line in f:
            results = line.split(r"$$$")
            name = results[0].split(".")[0]
            greedy_result = results[1]
            ground_truth = results[2]
            consult_dict[name] = (greedy_result, ground_truth)

    total_number = 0
    greedy_correctness = 0
    lm_correctness = 0
    greedy_precision_e = 0
    lm_predcision_e = 0
    for root, dirnames, files in os.walk("../data/nps"):
        for name in files:
    # for relative_dir in os.listdir("../data/nps"):
            path = os.path.join(root, name)
            print(path)
            data = np.load(path)
            total_number += 1
            result_lm = (decoder_lm.ctc_beam_search_decoder(data, beam_width=3))[0]
            result_greedy = consult_dict[name.split(".")[0]][0]
            result_truth = consult_dict[name.split(".")[0]][1]
            if result_lm == result_truth:
                lm_correctness += 1
            if result_greedy == result_truth:
                greedy_correctness += 1
            print("CTC model encoder matrix shape({}, {})".format(data.shape[0], data.shape[1]))
            print("Greedy Decoder output: {}".format(result_greedy))
            # print("LM Decoder ouput: {}".format(result_lm))
            print("LM Decoder output: {}".format(result_truth))
            # lm_pre = 1-float(edit_distance(result_lm, result_truth))/len(result_truth)
            # lm_predcision_e += lm_pre
            # greedy_pre = 1-float(edit_distance(result_greedy, result_truth))/len(result_truth)
            # greedy_precision_e += greedy_pre
            # print("Examples already tested:"+str(total_number))
            # print("lm raw precision:" + str(float(lm_correctness)/total_number))
            # print("greeedy raw precision:" + str(float(greedy_correctness)/total_number))
            # print("lm edit precision:" + str(lm_predcision_e/total_number))
            # print("greedy edit precision:" + str(greedy_precision_e/total_number))

    # print("Greedy precision:"+str(greedy_correctness/total_number))
    # print("LM precision:"+str(lm_correctness/total_number))
    # print("================")
