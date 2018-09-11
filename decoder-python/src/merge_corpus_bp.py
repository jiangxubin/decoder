# -*- coding: utf-8 -*-
"""
Merge char_train.txt and SinaWeibo_single.txt to get a bi_lingual mixed corpus for training of KenLM
"""


def merge(c_path, e_path):
    out_f = open("../data/mixed.txt", "w")
    with open(c_path, "r") as in_cf:
        for line in in_cf:
            out_f.write(line)
    print("Chinese corpus write done")
    with open(e_path, "r") as in_ef:
        for line in in_ef:
            out_f.write(line)
    print("English corpus write done")
    out_f.close()
    print("Output file stream close")


if __name__ == "__main__":
    merge(r"../data/SinaWeibo_single.txt", r"../data/char_train.txt")