import kenlm


class LM:
    def __init__(self):
        self.model = kenlm.Model(r"../data/train.bin")

    def score(self, characters):
        return self.model.score(characters, bos=False, eos=False)


if __name__ == "__main__":
    model = kenlm.Model(r"../data/train.bin")
    # sentences = ["small dog", "big dog", "i love you", "i hate you", "hello world"]
    #sentences = ["small dog", "small", "dog", "at", "it", "i", "g", "he", "hel", "llo", "lo", "ell"]
    sentences = ["wo"]
    sent_socre = {item:model.score(item, bos=False, eos=False) for item in sentences}
    for K, V in sent_socre.items():
        print(K + " " + str(V))
