"""
Merge multi copus files into a bi_lingual mixed corpus for training of KenLM
"""


def merge(*args):
    out_f = open("../data/mixed.txt", "w")
    for arg in args:
        with open(arg, "r") as i_f:
            for line in i_f:
                try:
                    out_f.write(line)
                except UnicodeDecodeError:
                    pass
            print("{} has been write to mixed txt".format(arg))
    out_f.close()
    print("Output file stream close")


if __name__ == "__main__":
    merge(r"../data/SinaWeibo_single.txt", r"../data/Sougou_single.txt",r"../data/char_train.txt")
