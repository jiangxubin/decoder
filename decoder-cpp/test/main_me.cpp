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
    map<string, pair<string, string>> consult_dict;
    std::ifstream fs("/home/augustus/Documents/decoder/decoder-cpp/data/list.txt");
    string line;
    string name;
    string greedy_result;
    string ground_truth;
    string delim = "$$$";
    while(getline(fs, line)){
        size_t pos=0;
        vector<string> tokens;
        string token;
        while((pos=line.find(delim))!=string::npos){
            token = line.substr(0, pos);
            tokens.push_back(token);
            line.erase(0, pos+delim.length());
        }
        name = tokens[0];
        greedy_result = tokens[1];
        ground_truth = tokens[2];
        consult_dict[name] = make_pair(greedy_result, ground_truth);

    }

    ifstream ifs_1("/home/augustus/Documents/decoder/decoder-cpp/data/data_path.txt");
    vector<string> paths;
    string line_1;
    while(getline(ifs_1, line_1)){
        paths.push_back(line_1);
        cout << line_1 << endl;
    }

    Transfer tf;
    for(auto ele:paths){
        vector<vector<float>> data = tf.transfer(ele);
        cout << ele << endl;
        cout<< data.size() << " " << data[0].size() << endl;
//        vector<string> res = decoder.beam_search_decoder(data, 2);
//        cout << res[0] << endl;
    }
    return 0;
}
