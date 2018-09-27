import numpy as np
import os


def nps2txt(path):
    name = path.split(r"/")[-1].split(".")[-2]
    name += ".txt"
    print(name)
    f = open(os.path.join("../../decoder-cpp/data/nps", name), "w")
    data = np.load(path)
    print(data.shape)
    for i in range(data.shape[0]):
        line = ''
        for j in range(len(data[0,:])):
            line += str(data[i, j])
            line += ' '
        line += "\n"
        f.write(line)
    f.close()


if __name__ == "__main__":
    for name in os.listdir("../..//decoder-python/data/nps"):
        full_path = os.path.join("../../decoder-python/data/nps", name)
        nps2txt(full_path)