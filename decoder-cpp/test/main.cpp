#include<iostream>
#include <set>
#include "include/BeamSearch.h"
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"


using namespace std;
using namespace lm::ngram;

int main() {

    std::cout<<"Hello world"<<std::endl;
    char *path = "/home/augustus/Documents/decoder-python/data/mixed2.bin";
    Config config;
    config.load_method = util::READ;

    Model model(path,config);
    State state, out_state;
    lm::FullScoreReturn ret;
    float score;
    const Vocabulary &vocab = model.GetVocabulary();

    string line;
    while (getline(cin, line)) {
        state = model.BeginSentenceState();
        score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(line, ' '); it; ++it) {
            cout << it << endl;
            lm::WordIndex vocab = model.GetVocabulary().Index(*it);
            ret = model.FullScore(state, vocab, out_state);
            score += ret.prob;
            state = out_state;
        }
        ret = model.FullScore(state, model.GetVocabulary().EndSentence(), out_state);
        score += ret.prob;

        cout<<line.c_str()<<":"<<score<<"\n";
    }


    return 0;
}
