import numpy as np
import sys
sys.path.append('../beamsearch')
from PrefixBeamSearch import CtcDecoder
import os
from Editdistance import edit_distance


if __name__ == "__main__":
    decoder_lm = CtcDecoder()

    path = "/home/augustus/Documents/decoder/decoder-python/data/nps/1174.npy"
    data = np.load(path)
    print("Test whether c++ and python have the same prob matrix "+str(data[10, 10]))
    result = decoder_lm.ctc_beam_search_decoder(data, beam_width=2)
    print(result[0])

