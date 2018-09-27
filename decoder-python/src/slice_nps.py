import numpy as np
import os


def slice_nps(in_path: str, desti_path, K: int):
    name = in_path.split("/")[-1].split(".")[-2]
    name += ".txt"
    data = np.load(in_path)
    file_path = os.path.join(desti_path, name)
    f = open(file_path, "w")
    for i in range(data.shape[0]):
        line = str()
        temp = np.argpartition(data[i], K)
        result_args = temp[:K]
        for arg in result_args:
            line += str(arg)
            line += ' '
        line += "\n"
        f.write(line)


def slice_nps_single(in_path: str, desti_path, K: int):
    index_name = in_path.split("/")[-1].split(".")[-2]
    val_name = index_name+".text"
    index_name += ".txt"
    data = np.load(in_path)
    file_path = os.path.join(desti_path, index_name)
    val_file_path = os.path.join(desti_path, val_name)
    f = open(file_path, "w")
    val_f = open(val_file_path, "w")
    # https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array
    for i in range(data.shape[0]):
        line = str()
        val_line = str()
        temp = np.argpartition(-data[i,:], K)
        val_temp = np.partition(-data[i,:], K)
        result_args = temp[:K]
        result_val = -val_temp[:K]
        for arg in result_args:
            line += str(arg)
            line += ' '
        for val in result_val:
            val_line += str(val)
            val_line += ' '
        f.write(line)
        val_f.write(val_line)
    f.close()
    val_f.close()


if __name__ == "__main__":
    for file_name in os.listdir("../data/nps"):
        full_path = os.path.join("../data/nps", file_name)
        print("Slice {}".format(full_path))
        # slice_nps(full_path, "../../decoder-cpp/data/nps_test", 500)
        slice_nps_single(full_path, "../../decoder-cpp/data/nps_test_single", 500)