//
// Created by augustus on 9/18/18.
//
#include<iostream>
#include <set>
#include "include/BeamSearch.h"
#include "include/Intermediate.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <map>
#include <unordered_map>
#include <thread>


using namespace std;
using namespace lm::ngram;


int main() {
    BeamSearch decoder;

    ifstream ifs("/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/value.txt");
    string line;
    vector<float> temp_float;
    while(getline(ifs, line)){
        stringstream ss(line);
        float val;
        while(ss>>val){
            temp_float.push_back(val);
        }
    }
    cout << temp_float.size() << endl;
    float prob[13000];
    for(int i=0; i<temp_float.size(); i++){
        prob[i] = temp_float[i];
    }
    float *prob_ptr = prob;


    ifstream ifs_1("/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/index.txt");
    string line_1;
    vector<int> temp_index;
    while(getline(ifs_1, line_1)){
        int idx;
        stringstream ss(line_1);
        while(ss>>idx){
            temp_index.push_back(idx);
        }
    }
    cout << temp_index.size()<< endl;

    int index[13000];
    for(int j=0; j<temp_index.size(); j++){
        index[j] = temp_index[j];
    }
    int *index_ptr = index;

    ifstream ifs_2{"/home/augustus/Documents/decoder/decoder-cpp/data/multi_thread_test_data/length.txt"};
    string line_2;
    vector<int> temp_length;
    while(getline(ifs_2, line_2)){
        int len;
        stringstream ss(line_2);
        while(ss>>len){
            temp_length.push_back(len);
        }
    }
    int length[4];
    for(int k=0; k<temp_length.size(); k++){
        length[k] = temp_length[k];
    }
    int *length_ptr = length;

    vector<string> res_p = decoder.multi_threading_decoder(prob_ptr, index_ptr, length_ptr, 4, 13000/500, 500, 6866, 2, 6865, (float)1e-5);
    for(auto iter=res_p.begin(); iter!=res_p.end(); iter++){
        cout<< *iter<< endl;
    }
    return 0;
}