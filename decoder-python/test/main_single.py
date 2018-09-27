import numpy as np
import os, sys
src_path = os.path.abspath(os.path.join('..'))
sys.path.append(src_path)
from src.PrefixBeamSearch import CtcDecoder
from src.Editdistance import edit_distance


if __name__ == "__main__":
    decoder_lm = CtcDecoder()
    # path = "/home/augustus/Documents/decoder/decoder-python/data/nps/1174.npy"
    # data = np.load(path)
    # print("Test whether c++ and python have the same prob matrix "+str(data[10, 10]))//        row.clear();
    # result = decoder_lm.ctc_beam_search_decoder(data, beam_width=2)
    prob_array = list()
    index_array = list()
    with open("/home/augustus/Documents/decoder/decoder-cpp/data/nps_test_single/38.txt", "r") as f:
        for line in f:
            temp = line.split()
            for index in temp:
                index_array.append(index)
    with open("/home/augustus/Documents/decoder/decoder-cpp/data/nps_test_single/38.text", "r") as f:
        for line in f:
            temp = line.split()
            for val in temp:
                prob_array.append(val)

    result_practical = decoder_lm.ctc_beam_search_decoder_practical(prob_array, index_array, int(len(prob_array)/500), 500, 6866, 2, 6865, 1e-5)
    print(result_practical[0])

