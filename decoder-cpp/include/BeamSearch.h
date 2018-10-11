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

/// Class of fundamental component in Beam search
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


/// Container of BeamEntry which provides some operation to its member
class BeamState{
public:
//    map<set<int>, BeamEntry> entries;

    /// A map which stores BeamEntry which uses a string converted from vector of labels as key
    map<string, BeamEntry> entries;

    /// An operation which normalize all scores for the convenience of later comparision
    void normalize();

    /// An operation which sorts all BeamEntry by score
    vector<vector<int>> sort();//Here we couldn't use this form vector<vector<int>>& sort() for local variable can't be used as return value;
    BeamState();
};

/// Main class of Beam search algorithm
class BeamSearch{
public:

    /// Load the language model into a model object
    Model model{"/home/augustus/Documents/decoder/decoder-python/data/mixed.bin"};

    /// A map which stores the word and corresponding frequency
    map<int, string> dict_map;
    string map_path = "/home/augustus/Documents/decoder/decoder-python/data/ocr_char_set.txt";


    BeamSearch();
    void apply_lm(BeamEntry &parent_beam, BeamEntry &child_beam);
    void add_beam(BeamState &state, vector<int> labels);

    /// An auxiliary function. In C++ it is forbidden to use a set as a map key while
    /// in Python it is free to use set  as dict key. So it is necessary to convert a vector of labels into a string which
    /// can be used as map key
    string labels2string(vector<int> labels);

    /// Single thread decoder which uses matrix as input
    /// \param prob_matrix: A matrix of shape(time-steps， length-of-all-possibilities)
    /// \param beam_width : : Number of results to keep among every filter of all candidates
    /// \return :A vector which stores possible decode results of an example
    vector<string> beam_search_decoder(vector<vector<float>> prob_matrix, int beam_width);

    /// Single thread decoder which use pointer as input
    /// \param prob: a float pointer which points to a tensor of output soft-max probability
    /// \param index: a int pointer which points to a tensor of index corresponding to the probability
    /// \param T : Max time-steps of all examples among  a batch
    /// \param K : Top k probabilities to keep out of all possibilities
    /// \param A : Number of all possibilities
    /// \param beam_width : Number of results to keep among every filter of all candidates
    /// \param blank_index: index of "blank" char which is 6865 by default
    /// \param default_blank_prob : Default compensate possibility for "blank" char
    /// \return :A vector which stores possible decode results of an example
    vector<string> beam_search_decoder(float *prob, int *index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob);
    void beam_search_decoder_m(vector<float> prob, vector<int> index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob);

    /// Multi threads decoder wrapper of normal single thread decoder
    /// \param prob: a float pointer which points to a tensor of output soft-max probability
    /// \param index: a int pointer which points to a tensor of index corresponding to the probability
    /// \param seq_len :a int pointer which points to a array of time-steps of each example
    /// \param N : Number of examples in a batch
    /// \param T : Max time-steps of all examples among  a batch
    /// \param K : Top k probabilities to keep out of all possibilities
    /// \param A : Number of all possibilities
    /// \param beam_width : Number of results to keep among every filter of all candidates
    /// \param blank_index : index of "blank" char which is 6865 by default
    /// \param default_blank_prob : Default compensate possibility for "blank" char
    /// \return : A vector of decode result for each example in the batch
    vector<string> multi_threading_decoder(float *prob, int *index, int *seq_len, int N, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob);

};