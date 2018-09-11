#include<iostream>
#include <set>
#include "include/BeamSearch.h"
#include "include/Intermediate.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <map>


using namespace std;
using namespace lm::ngram;


int main() {
    BeamSearch decoder;

    Transfer tf;
    vector<vector<float>> prob_matrix = tf.transfer("/home/augustus/Documents/decoder/decoder-cpp/data/nps/1174.txt");
    cout <<"Test whether c++ and python have the same prob matrix "<< prob_matrix[10][10] << endl;
    vector<string> res = decoder.beam_search_decoder(prob_matrix, 2);
    cout << res[0] << endl;
    return 0;
}
