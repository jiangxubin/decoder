//
// Created by augustus on 8/30/18.
//
#include "include/BeamSearch.h"
#include <fstream>
#include <sstream>


BeamEntry::BeamEntry() {
    this->blank_score = 0;
    this->lm_score = 1;
    this->no_blank_score = 0;
    this->total_score = 0;
    this->LM = false;
}

void BeamState::normalize() {
    for(auto iter=this->entries.begin(); iter!=this->entries.end(); iter++){
        int label_number = iter->second.labels.size();
        if(label_number!=0){
            this->entries[iter->first].lm_score = std::pow(this->entries[iter->first].lm_score,1.0/label_number);
        }
        else{
            this->entries[iter->first].lm_score = std::pow(this->entries[iter->first].lm_score,1.0);
        }

    }
}

BeamState::BeamState() {

}

vector<vector<int>> BeamState::sort() {
    vector<BeamEntry> items;
    for(auto iter=this->entries.begin(); iter!=entries.end(); iter++){
        items.push_back(iter->second);
    }
    std::sort(items.begin(),items.end(), [](BeamEntry lft, BeamEntry rht){
        return lft.total_score*lft.lm_score > rht.total_score*rht.lm_score;
    });
    vector< vector<int>> sorted_beams;
    for(auto ele:items){
        sorted_beams.push_back(ele.labels);
    }
    return sorted_beams;
}


//BeamSearch::BeamSearch(string model_path):model(model_path){
BeamSearch::BeamSearch(){
    ifstream ifile(map_path);
    string line;
    int index;
    string character;
    while(getline(ifile, line)){
        stringstream ss(line);
        ss>>index>>character;
        cout << index << character << endl;
        this->dict_map[index] = character;
    }
    this->dict_map[dict_map.rbegin()->first]=string();
}

void BeamSearch::apply_lm(BeamEntry &parent_beam, BeamEntry &child_beam) {
    float lm_weights = 0.01;
    if(!child_beam.LM){
        string first, second;
        if(parent_beam.labels.size()!=0){
            first = this->dict_map[parent_beam.labels.back()];
        }
        else{
            first = this->dict_map[6865];
        }
        cout << "first " << first << endl;
        second = this->dict_map[child_beam.labels.back()];
        cout << "second " << second << endl;
        string uni_gram;
        uni_gram += first;
        string bi_gram = uni_gram;
        bi_gram+=second;

        const Vocabulary &vocab = model.GetVocabulary();
        State state, out_state;
        lm::FullScoreReturn ret;
        state = model.BeginSentenceState();
        float first_score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(uni_gram, ' '); it; ++it) {
//            lm::WordIndex vocab = model.GetVocabulary().Index(*it);
            ret = model.FullScore(state, vocab.Index(*it), out_state);
            first_score += ret.prob;
            state = out_state;
        }
        ret = model.FullScore(state, model.GetVocabulary().EndSentence(), out_state);
        first_score += ret.prob;
        cout<< "unigram score "<<uni_gram<<" "<<first_score<<endl;

        State state_1, out_state_1;
        lm::FullScoreReturn ret_1;
        state_1 = model.BeginSentenceState();
        float bi_score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(bi_gram, ' '); it; ++it) {
//            lm::WordIndex vocab = model.GetVocabulary().Index(*it);
            ret_1 = model.FullScore(state_1, vocab.Index(*it), out_state_1);
            bi_score += ret_1.prob;
            state_1 = out_state_1;
        }
        ret_1 = model.FullScore(state_1, model.GetVocabulary().EndSentence(), out_state_1);
        bi_score += ret_1.prob;
        cout<<"bigram score"<<bi_gram << " "<<bi_score<<endl;

        float conditional_prob = pow(pow(10, (bi_score-first_score)), lm_weights);
        child_beam.lm_score = parent_beam.lm_score*conditional_prob;
        child_beam.LM = true;
    }
}

void BeamSearch::add_beam(BeamState &state, vector<int> labels) {
    string label_str = labels2string(labels);
    auto iter = state.entries.find(label_str);
    if(!(iter!=state.entries.end())){
        state.entries[label_str]= BeamEntry();
    }
}


string BeamSearch::labels2string(vector<int> labels) {
    string res;
    for(auto ele:labels){
        res += to_string(ele);
    }
    return res;

}

vector<string> BeamSearch::beam_search_decoder(vector<vector<float>> prob_matrix, int beam_width) {
    int time_step = (int)prob_matrix.size();
    int dict_length = (int)prob_matrix[0].size();
    int blank_idx = dict_length - 1;

    BeamState last;
    vector<int> labels;
    string labels_str = labels2string(labels);


    last.entries[labels_str] = BeamEntry();
    last.entries[labels_str].total_score = 1;
    last.entries[labels_str].blank_score = 1;
    cout << last.entries.size() << endl;

    for(int i=0; i< time_step; i++){
        BeamState curr;
        cout << curr.entries.size()<<endl;
        vector<vector<int>> candidate_labels = last.sort();
        cout << candidate_labels.size() << endl;
        vector<vector<int>> best_labels;
        if(candidate_labels.size()<=beam_width){
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.end());
        }
        else{
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.begin()+beam_width);
        }
        for(auto labels:best_labels) {
            string labels_str = labels2string(labels);
//            cout << labels_str << endl;
            float no_blank_score = 0;
            if (!labels.empty()) {
                no_blank_score = last.entries[labels_str].no_blank_score * prob_matrix[i][*labels.rbegin()];
//                cout <<"if (!labels.empty()) " <<"no_blank_score "<< no_blank_score << endl;
            }
            float blank_score = last.entries[labels_str].total_score * prob_matrix[i][blank_idx];
//            cout << "blank_score " << blank_score << endl;
            this->add_beam(curr, labels);
//            cout << "copy add beam" <<endl;
            curr.entries[labels_str].labels = labels;
            curr.entries[labels_str].no_blank_score += no_blank_score;
//            cout << "copy curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str].no_blank_score<< endl;
            curr.entries[labels_str].blank_score += blank_score;
//            cout << "copy curr.entries[labels_str].blank_score "<< curr.entries[labels_str].blank_score<<endl;
            curr.entries[labels_str].total_score += no_blank_score + blank_score;
//            cout << "copy curr.entries[labels_str].total_score " << curr.entries[labels_str].total_score <<endl;
            curr.entries[labels_str].lm_score = last.entries[labels_str].lm_score;
//            cout << "copy curr.entries[labels_str].lm_score " << curr.entries[labels_str].lm_score<<endl;
            curr.entries[labels_str].LM = true;
            for (int j = 0; j < dict_length - 1; j++) {
                vector<int> new_labels = labels;
                new_labels.push_back(j);
                string new_labels_str = labels2string(new_labels);
//                cout << "string new_labels_str " <<new_labels_str<<endl;
                if (!labels.empty() && *(labels.rbegin()) == j) {
                    no_blank_score = prob_matrix[i][j] * last.entries[labels_str].blank_score;
//                    cout << "if (!labels.empty() && *(labels.rbegin()) == j) "<< "no_blank_score " << no_blank_score << endl;
                } else {
                    no_blank_score = prob_matrix[i][j] * last.entries[labels_str].total_score;
//                    cout<<"else " << "no_blank_score " << no_blank_score << endl;
                }
                this->add_beam(curr, new_labels);
//                cout << "extend add beam" << endl;

                curr.entries[new_labels_str].labels = new_labels;;
                curr.entries[new_labels_str].no_blank_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].no_blank_score " << curr.entries[new_labels_str].no_blank_score <<endl;
                curr.entries[new_labels_str].total_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].total_score " << curr.entries[new_labels_str].total_score<<endl;
                this->apply_lm(curr.entries[labels_str], curr.entries[new_labels_str]);
//                cout << "Extend apply lm "<< endl;
            }
        }
        last = curr;
//        cout << "last = curr" << endl;
    }
    last.normalize();
    vector<vector<int>> final_labels_candidate = last.sort();
    vector<vector<int>> final_labels;
    final_labels = vector<vector<int>>(final_labels_candidate.begin(), final_labels_candidate.begin()+beam_width);

    vector<string> res_set;
    for(auto &labels:final_labels){
        string res;
        for(auto loc:labels){
            string now = this->dict_map[loc];
            res += now;
        }
        res_set.push_back(res);
    }
    return res_set;
}


vector<string>BeamSearch::beam_search_decoder(float *prob, int *index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob){
    int time_step = T;
    int blank_idx = blank_index;

    vector<map<int, float>> prob_matrix;
    vector<vector<int>> insert_order;
    for(int i=0;i<time_step; i++){
        map<int, float> row;
        vector<int> row_order;
        for(int j = i*K; j<(i+1)*K;j++){
            int key = index[j];
            float value = prob[j];
            row[key] = value;
            row_order.push_back(key);

//            cout <<"i " << i << " j " << j << " key " << key << " value " << row[key] << endl;
        }
        prob_matrix.push_back(row);
        insert_order.push_back(row_order);
    }

    BeamState last;
    vector<int> labels;
    string labels_str = labels2string(labels);
    last.entries[labels_str] = BeamEntry();
    last.entries[labels_str].total_score = 1;
    last.entries[labels_str].blank_score = 1;
//    cout << last.entries.size() << endl;

    for(int i=0; i<time_step; i++){
        BeamState curr;
        vector<vector<int>> candidate_labels = last.sort();
        vector<vector<int>> best_labels;
        if(candidate_labels.size()<=beam_width){
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.end());
        }
        else{
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.begin()+beam_width);
        }
        for(auto labels:best_labels) {
            float no_blank_score = 0.0;
            string labels_str_1 = labels2string(labels);
            if (!labels.empty()&&prob_matrix[i].find(labels.back())!=prob_matrix[i].end()) {
//                cout << "last.entries[labels_str].no_blank_score " << last.entries[labels_str_1].no_blank_score << "*labels.rbegin() " << labels.back() << "prob_matrix[i][*labels.rbegin()] "<< prob_matrix[i][labels.back()]<< endl;
                no_blank_score = last.entries[labels_str_1].no_blank_score * prob_matrix[i][labels.back()];
//                cout <<"if (!labels.empty()) " <<"no_blank_score "<< no_blank_score << endl;
            }
            float blank_score;
            if(prob_matrix[i].find(blank_idx)!=prob_matrix[i].end()){
                blank_score = last.entries[labels_str_1].total_score * prob_matrix[i][blank_idx];
//                cout << "in prob matrix " <<"last.entries[labels_str].total_score "<<last.entries[labels_str_1].total_score<< "prob_matrix[i][blank_idx]"<<prob_matrix[i][blank_idx] << endl;
//                cout << "blank_score "<< blank_score<< endl;
            }
            else{
                blank_score = last.entries[labels_str_1].total_score * default_blank_prob;
//                cout << "not in prob matrix " <<"last.entries[labels_str].total_score "<<last.entries[labels_str_1].total_score<< "default_blank_prob"<<default_blank_prob;
//                cout << "blank_score " << blank_score << endl;
            }

            this->add_beam(curr, labels);
//            cout << "copy add beam" <<endl;
            curr.entries[labels_str_1].labels = labels;
//            cout << "BEFORE curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str_1].no_blank_score<<endl;
            curr.entries[labels_str_1].no_blank_score += no_blank_score;
//            cout << "copy curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str_1].no_blank_score<< endl;
            curr.entries[labels_str_1].blank_score += blank_score;
//            cout << "copy curr.entries[labels_str].blank_score "<< curr.entries[labels_str_1].blank_score<<endl;
            curr.entries[labels_str_1].total_score += (no_blank_score + blank_score);
//            cout << "copy curr.entries[labels_str].total_score " << curr.entries[labels_str_1].total_score <<endl;
            curr.entries[labels_str_1].lm_score = last.entries[labels_str_1].lm_score;
//            cout << "copy curr.entries[labels_str].lm_score " << curr.entries[labels_str_1].lm_score<<endl;
            curr.entries[labels_str_1].LM = true;

            for (auto iter = insert_order[i].begin(); iter!=insert_order[i].end(); iter++) {
                vector<int> new_labels = labels;
                new_labels.push_back(*iter);
                string new_labels_str = labels2string(new_labels);
                float no_blank_score=0;
                if (!labels.empty() && labels.back()== *iter) {
                    no_blank_score = prob_matrix[i][*iter] * last.entries[labels_str_1].blank_score;
//                    cout << "if (!labels.empty() && *(labels.rbegin()) == j) "<< "no_blank_score " << no_blank_score << endl;
                } else {
//                    cout <<"iter->first "<< *iter << endl;
                    no_blank_score = prob_matrix[i][*iter] * last.entries[labels_str_1].total_score;
//                    cout << "prob_matrix[i][iter->first] " << prob_matrix[i][*iter] << "last.entries[labels_str].total_score "<< last.entries[labels_str_1].total_score<<endl;
//                    cout<<"else " << "no_blank_score " << no_blank_score << endl;
                }
                this->add_beam(curr, new_labels);
//                cout << "extend add beam" << endl;

                curr.entries[new_labels_str].labels = new_labels;
                curr.entries[new_labels_str].no_blank_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].no_blank_score " << curr.entries[new_labels_str].no_blank_score <<endl;
                curr.entries[new_labels_str].total_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].total_score " << curr.entries[new_labels_str].total_score<<endl;
                this->apply_lm(curr.entries[labels_str_1], curr.entries[new_labels_str]);
//                cout << "curr.entries[new_labels_str] " << curr.entries[new_labels_str].lm_score << endl;
//                cout << "Extend apply lm "<< endl;
            }
        }
        last = curr;
//        cout << "last = curr" << endl;
    }
    last.normalize();
    vector<vector<int>> final_labels_candidate = last.sort();
    vector<vector<int>> final_labels;
    final_labels = vector<vector<int>>(final_labels_candidate.begin(), final_labels_candidate.begin()+beam_width);

    vector<string> res_set;
    for(auto labels:final_labels){
        string res;
        for(auto loc:labels){
            string now = this->dict_map[loc];
            res += now;
        }
        res_set.push_back(res);
    }
    return res_set;
}


void BeamSearch::beam_search_decoder_m(vector<float> prob, vector<int> index, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob, string &result){
    int time_step = T;
    int blank_idx = blank_index;

    vector<map<int, float>> prob_matrix;
    vector<vector<int>> insert_order;
    for(int i=0;i<time_step; i++){
        map<int, float> row;
        vector<int> row_order;
        for(int j = i*K; j<(i+1)*K;j++){
            int key = index[j];
            float value = prob[j];
            row[key] = value;
            row_order.push_back(key);

//            cout <<"i " << i << " j " << j << " key " << key << " value " << row[key] << endl;
        }
        prob_matrix.push_back(row);
        insert_order.push_back(row_order);
    }

    BeamState last;
    vector<int> labels;
    string labels_str = labels2string(labels);
    last.entries[labels_str] = BeamEntry();
    last.entries[labels_str].total_score = 1;
    last.entries[labels_str].blank_score = 1;
//    cout << last.entries.size() << endl;

    for(int i=0; i<time_step; i++){
        BeamState curr;
        vector<vector<int>> candidate_labels = last.sort();
        vector<vector<int>> best_labels;
        if(candidate_labels.size()<=beam_width){
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.end());
        }
        else{
            best_labels = vector<vector<int>>(candidate_labels.begin(), candidate_labels.begin()+beam_width);
        }
        for(auto labels:best_labels) {
            float no_blank_score = 0.0;
            string labels_str_1 = labels2string(labels);
            if (!labels.empty()&&prob_matrix[i].find(labels.back())!=prob_matrix[i].end()) {
//                cout << "last.entries[labels_str].no_blank_score " << last.entries[labels_str_1].no_blank_score << "*labels.rbegin() " << labels.back() << "prob_matrix[i][*labels.rbegin()] "<< prob_matrix[i][labels.back()]<< endl;
                no_blank_score = last.entries[labels_str_1].no_blank_score * prob_matrix[i][labels.back()];
//                cout <<"if (!labels.empty()) " <<"no_blank_score "<< no_blank_score << endl;
            }
            float blank_score;
            if(prob_matrix[i].find(blank_idx)!=prob_matrix[i].end()){
                blank_score = last.entries[labels_str_1].total_score * prob_matrix[i][blank_idx];
//                cout << "in prob matrix " <<"last.entries[labels_str].total_score "<<last.entries[labels_str_1].total_score<< "prob_matrix[i][blank_idx]"<<prob_matrix[i][blank_idx] << endl;
//                cout << "blank_score "<< blank_score<< endl;
            }
            else{
                blank_score = last.entries[labels_str_1].total_score * default_blank_prob;
//                cout << "not in prob matrix " <<"last.entries[labels_str].total_score "<<last.entries[labels_str_1].total_score<< "default_blank_prob"<<default_blank_prob;
//                cout << "blank_score " << blank_score << endl;
            }

            this->add_beam(curr, labels);
//            cout << "copy add beam" <<endl;
            curr.entries[labels_str_1].labels = labels;
//            cout << "BEFORE curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str_1].no_blank_score<<endl;
            curr.entries[labels_str_1].no_blank_score += no_blank_score;
//            cout << "copy curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str_1].no_blank_score<< endl;
            curr.entries[labels_str_1].blank_score += blank_score;
//            cout << "copy curr.entries[labels_str].blank_score "<< curr.entries[labels_str_1].blank_score<<endl;
            curr.entries[labels_str_1].total_score += (no_blank_score + blank_score);
//            cout << "copy curr.entries[labels_str].total_score " << curr.entries[labels_str_1].total_score <<endl;
            curr.entries[labels_str_1].lm_score = last.entries[labels_str_1].lm_score;
//            cout << "copy curr.entries[labels_str].lm_score " << curr.entries[labels_str_1].lm_score<<endl;
            curr.entries[labels_str_1].LM = true;

            for (auto iter = insert_order[i].begin(); iter!=insert_order[i].end(); iter++) {
                vector<int> new_labels = labels;
                new_labels.push_back(*iter);
                string new_labels_str = labels2string(new_labels);
                float no_blank_score=0;
                if (!labels.empty() && labels.back()== *iter) {
                    no_blank_score = prob_matrix[i][*iter] * last.entries[labels_str_1].blank_score;
//                    cout << "if (!labels.empty() && *(labels.rbegin()) == j) "<< "no_blank_score " << no_blank_score << endl;
                } else {
//                    cout <<"iter->first "<< *iter << endl;
                    no_blank_score = prob_matrix[i][*iter] * last.entries[labels_str_1].total_score;
//                    cout << "prob_matrix[i][iter->first] " << prob_matrix[i][*iter] << "last.entries[labels_str].total_score "<< last.entries[labels_str_1].total_score<<endl;
//                    cout<<"else " << "no_blank_score " << no_blank_score << endl;
                }
                this->add_beam(curr, new_labels);
//                cout << "extend add beam" << endl;

                curr.entries[new_labels_str].labels = new_labels;
                curr.entries[new_labels_str].no_blank_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].no_blank_score " << curr.entries[new_labels_str].no_blank_score <<endl;
                curr.entries[new_labels_str].total_score += no_blank_score;
//                cout << "extend curr.entries[new_labels_str].total_score " << curr.entries[new_labels_str].total_score<<endl;
                this->apply_lm(curr.entries[labels_str_1], curr.entries[new_labels_str]);
//                cout << "curr.entries[new_labels_str] " << curr.entries[new_labels_str].lm_score << endl;
//                cout << "Extend apply lm "<< endl;
            }
        }
        last = curr;
//        cout << "last = curr" << endl;
    }
    last.normalize();
    vector<vector<int>> final_labels_candidate = last.sort();
    vector<int> best_labels = final_labels_candidate[0];
    string best_result;
    for(auto loc:best_labels){
        best_result += this->dict_map[loc];
    }
    result = best_result;
}


vector<string>BeamSearch::multi_threading_decoder(float *prob, int *index, int *seq_len, int N, int T, int K, int A, int beam_width, int blank_index, float default_blank_prob) {
    vector<string> result(N);
    vector<thread> threads(N);
    for(int i=0; i<N; i++){
        vector<float> single_prob(prob+i*T*K, prob+(i+1)*T*K);
        vector<int> single_index(index+i*T*K, index+(i+1)*T*K);
        int time_steps = seq_len[i];
//        threads[i]=thread{&BeamSearch::beam_search_decoder_m, this, single_prob, single_index, time_steps, K, A, beam_width, blank_index, default_blank_prob, result[i]};
        threads[i]=thread{&BeamSearch::beam_search_decoder_m, this, single_prob, single_index, time_steps, K, A, beam_width, blank_index, default_blank_prob, result[i]};
        threads[i].join();
    }
    return result;
}