import numpy as np
import os, sys
src_path = os.path.abspath(os.path.join('..'))
sys.path.append(src_path)
from src.PrefixBeamSearch import CtcDecoder
from src.Editdistance import edit_distance


def is_english(data:str)->bool:
    try:
        data.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


if __name__ == "__main__":
    consult_dict = {}
    with open("../data/list.txt", "r") as f:
        for line in f:
            ground_truth = line.split(r"$$$")[2]
            if is_english(ground_truth):
                name = line.split("$$$")[0]
                index = name.split(".")[0]
                consult_dict[index] = (ground_truth, line.split("$$$")[1])
                # print(index)
                # print(ground_truth)

    decoder_lm = CtcDecoder()
    good_f = open("../data/good_case.txt", "w")
    bad_f = open("../data/bad_case.txt", "w")
    for index in consult_dict.keys():
        name = index + ".npy"
        print("Begin decode {}".format(name))
        full_path = os.path.join("../data/nps", name)
        try:
            data = np.load(full_path)
            lm_result = decoder_lm.ctc_beam_search_decoder(data, beam_width=3)[0]
            greedy_result = consult_dict[index][1]
            truth_result = consult_dict[index][0]
            greedy_ed = edit_distance(greedy_result, truth_result)
            lm_ed = edit_distance(lm_result, truth_result)
            print("greedy ed:"+str(greedy_ed)+" lm ed:"+str(lm_ed))
            print("Truth:{} Greedy:{} LM:{}".format(truth_result, greedy_result, lm_result))
            if greedy_ed > lm_ed:
                good_f.write(str(index) + truth_result+" "+lm_result+" "+greedy_result+"\n")
            elif greedy_ed < lm_ed:
                bad_f.write(str(index) + truth_result+" "+lm_result+" "+greedy_result+"\n")
        except FileNotFoundError:
            pass
    good_f.close()
    bad_f.close()
    print("Stop")

