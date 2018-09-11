import os


if __name__ == "__main__":
    with open("/home/augustus/Documents/decoder/decoder-cpp/data/data_path.txt", "w") as f:
        for name in os.listdir("/home/augustus/Documents/decoder/decoder-cpp/data/nps"):
            full_path = os.path.join("/home/augustus/Documents/decoder/decoder-cpp/data/nps", name)
            full_path = full_path + "\n"
            f.write(full_path)