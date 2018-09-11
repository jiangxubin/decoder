//
// Created by augustus on 9/10/18.
//
#include "include/Intermediate.h"


vector<vector<float>> Transfer::transfer(string path){
    ifstream ifs(path);
    vector<vector<float>> data;
    string line;
    while(getline(ifs, line)){
        vector<float> temp;
        float ele;
        istringstream iss(line);
        while(iss>>ele){
            temp.push_back(ele);
        }
        data.push_back(temp);
    }

    return data;
}