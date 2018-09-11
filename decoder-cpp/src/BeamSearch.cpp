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
        items.push_back((*iter).second);
    }
    std::sort(items.begin(),items.end(), [](BeamEntry lft, BeamEntry rht){
        return lft.total_score*lft.lm_score > rht.total_score*rht.lm_score;
    });
    vector< vector<int>> sorted_beams;
    for(auto &iter:items){
        sorted_beams.push_back(iter.labels);
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
        this->dict_map[index] = character;
//        cout << index << character << endl;
    }
}

void BeamSearch::apply_lm(BeamEntry &parent_beam, BeamEntry &child_beam) {
    float lm_weights = 0.01;
    if(!child_beam.LM){
        string first, second;
        if(parent_beam.labels.size()!=0){
            first = this->dict_map[*(parent_beam.labels.rbegin())];
        }
        else{
            first = this->dict_map.rbegin()->second;
        }
        second = this->dict_map[*(child_beam.labels.rbegin())];
        string uni_gram=first;
        string bi_gram = uni_gram + second;

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
//        cout<<first<<":"<<first_score<<"\n";

        State state_1, out_state_1;
        lm::FullScoreReturn ret_1;
        state_1 = model.BeginSentenceState();
        float bi_score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(bi_gram, ' '); it; ++it) {
//            lm::WordIndex vocab = model.GetVocabulary().Index(*it);
            ret_1 = model.FullScore(state, vocab.Index(*it), out_state_1);
            bi_score += ret_1.prob;
            state_1 = out_state_1;
        }
        ret_1 = model.FullScore(state, model.GetVocabulary().EndSentence(), out_state_1);
        bi_score += ret_1.prob;
//        cout<<bi_gram<<":"<<bi_score<<"\n";

        float conditional_prob = pow(pow(10, (bi_score-first_score)), lm_weights);
        child_beam.lm_score = parent_beam.lm_score*conditional_prob;
        child_beam.LM = true;
    }
}

void BeamSearch::add_beam(BeamState &state, vector<int> labels) {
    string label_str = labels2string(labels);
    auto iter = state.entries.find(label_str);
    if(iter==state.entries.end()){
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
                cout <<"if (!labels.empty()) " <<"no_blank_score "<< no_blank_score << endl;
            }
            float blank_score = last.entries[labels_str].total_score * prob_matrix[i][blank_idx];
            cout << "blank_score " << blank_score << endl;
            this->add_beam(curr, labels);
            cout << "copy add beam" <<endl;
            curr.entries[labels_str].labels = labels;
            curr.entries[labels_str].no_blank_score += no_blank_score;
            cout << "copy curr.entries[labels_str].no_blank_score "<<curr.entries[labels_str].no_blank_score<< endl;
            curr.entries[labels_str].blank_score += blank_score;
            cout << "copy curr.entries[labels_str].blank_score "<< curr.entries[labels_str].blank_score<<endl;
            curr.entries[labels_str].total_score += no_blank_score + blank_score;
            cout << "copy curr.entries[labels_str].total_score " << curr.entries[labels_str].total_score <<endl;
            curr.entries[labels_str].lm_score = last.entries[labels_str].lm_score;
            cout << "copy curr.entries[labels_str].lm_score " << curr.entries[labels_str].lm_score<<endl;
            curr.entries[labels_str].LM = true;
            for (int j = 0; j < dict_length - 1; j++) {
                vector<int> new_labels = labels;
                new_labels.push_back(j);
                string new_labels_str = labels2string(new_labels);
//                cout << "string new_labels_str " <<new_labels_str<<endl;
                if (!labels.empty() && *(labels.rbegin()) == j) {
                    no_blank_score = prob_matrix[i][j] * last.entries[labels_str].blank_score;
                    cout << "if (!labels.empty() && *(labels.rbegin()) == j) "<< "no_blank_score " << no_blank_score << endl;
                } else {
                    no_blank_score = prob_matrix[i][j] * last.entries[labels_str].total_score;
                    cout<<"else " << "no_blank_score " << no_blank_score << endl;
                }
                this->add_beam(curr, new_labels);
                cout << "extend add beam" << endl;

                curr.entries[new_labels_str].labels = new_labels;;
                curr.entries[new_labels_str].no_blank_score += no_blank_score;
                cout << "extend curr.entries[new_labels_str].no_blank_score " << curr.entries[new_labels_str].no_blank_score <<endl;
                curr.entries[new_labels_str].total_score += no_blank_score;
                cout << "extend curr.entries[new_labels_str].total_score " << curr.entries[new_labels_str].total_score<<endl;
                this->apply_lm(curr.entries[labels_str], curr.entries[new_labels_str]);
                cout << "Extend apply lm "<< endl;
            }
        }
        last = curr;
        cout << "last = curr" << endl;
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
