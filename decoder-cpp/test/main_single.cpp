#include<iostream>
#include <set>
#include "include/BeamSearch.h"
#include "include/Intermediate.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <map>
#include <unordered_map>


using namespace std;
using namespace lm::ngram;


int main() {
    BeamSearch decoder;

//    Transfer tf;
//    vector<vector<float>> prob_matrix = tf.transfer("/home/augustus/Documents/decoder/decoder-cpp/data/nps/1174.txt");
//    vector<string> res = decoder.beam_search_decoder(prob_matrix, 2);
//    cout << res[0] << endl;

    ifstream ifs("/home/augustus/Documents/decoder/decoder-cpp/data/nps_test_single/38.text");
    string line;
    vector<float> temp_float;
    while(getline(ifs, line)){
        stringstream ss(line);
        float val;
        while(ss>>val){
            temp_float.push_back(val);
        }
    }
//    cout << temp_float.size() << endl;
    float prob[13000];
    for(int i=0; i<temp_float.size(); i++){
        prob[i] = temp_float[i];
    }
    float *prob_ptr = prob;


    ifstream ifs_1("/home/augustus/Documents/decoder/decoder-cpp/data/nps_test_single/38.txt");
    string line_1;
    vector<int> temp_index;
    while(getline(ifs_1, line_1)){
        int idx;
        stringstream ss(line_1);
        while(ss>>idx){
        temp_index.push_back(idx);
        }
    }
//    cout << temp_index.size()<< endl;

    int index[13000];
    for(int j=0; j<temp_index.size(); j++){
        index[j] = temp_index[j];
    }
    int *index_ptr = index;

    vector<string> res_p = decoder.beam_search_decoder(prob_ptr, index_ptr, 13000/500, 500, 6866, 2, 6865, 1e-5);
    for(auto iter=res_p.begin(); iter!=res_p.end(); iter++){
        cout<< *iter<< endl;
    }
    return 0;
}
