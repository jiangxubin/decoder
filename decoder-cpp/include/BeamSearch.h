//
// Created by augustus on 8/30/18.
//

#ifndef BEAMSEARCH_H
#define BEAMSEARCH_H
#endif //CTC_DECODER_BEAMSEARCH_H
#include <map>
#include <unordered_map>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"
using namespace std;
using namespace lm::ngram;


class BeamEntry{
public:
    float total_score;
    float lm_score;
    float no_blank_score;
    float blank_score;
    bool LM=false;
    vector<int> labels;
    BeamEntry();

};

class BeamState{
public:
//    map<set<int>, BeamEntry> entries;
    map<string, BeamEntry> entries;
    void normalize();
    vector<vector<int>> sort();//Here we couldn't use this form vector<vector<int>>& sort();
    BeamState();
};


class BeamSearch{
public:
    Model model{"/home/augustus/Documents/decoder/decoder-python/data/mixed.bin"};
//    Model model;
    map<int, string> dict_map;
    string map_path = "/home/augustus/Documents/decoder/decoder-python/data/ocr_char_set.txt";
    BeamSearch();
//    BeamSearch(string model_path);
    void apply_lm(BeamEntry &parent_beam, BeamEntry &child_beam);
    void add_beam(BeamState &state, vector<int> labels);
    string labels2string(vector<int> labels);
    vector<string> beam_search_decoder(vector<vector<float>> prob_matrix, int beam_width);
    vector<string> beam_search_decoder(float *prob, int *index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob);
    void beam_search_decoder_m(vector<float> prob, vector<int> index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob, string &result);
    vector<string> multi_threading_decoder(float *prob, int *index, int *seq_len, int N, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob);
};