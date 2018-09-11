eng_path = "../data/train.txt"


def characterization():
    char_train = open("../data/char_train.txt", "w")
    with open(eng_path, "r") as f:
        for line in f:
            res = []
            for index in range(len(line) - 1):
                res.append(line[index])
                if line[index] != " " and line[index + 1] != " ":
                    res.append(" ")
            res.append(line[-1])
            res_str = "".join(res)
            # print(res_str)
            char_train.write(res_str)
    char_train.close()


if __name__ == "__main__":
    characterization()
    print("stop")
