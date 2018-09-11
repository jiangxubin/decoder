//
// Created by augustus on 8/30/18.
//

#include "../include/EditDistance.h"
#include <vector>


EditDistance::EditDistance() {

}

EditDistance::~EditDistance() {

}

int EditDistance::edit_distance(string &left, string &right) {
    if(left.empty()) return (int)right.size();
    if(right.empty()) return (int)left.size();
    int data[left.size()+1][right.size()+1];
    for(int i=0; i<right.size()+1; i++){
        data[0][i] = i;
    }
    for(int j=0; j<left.size()+1; j++){
        data[j][0] = j;
    }
    for(int m=0; m<left.size(); m++){
        for(int n=0; n < right.size(); n++){
            if(left[m]==right[n]){
                data[m+1][n+1] = data[m][n];
            }
            else{
                int a = data[m+1][n] + 1;
                int b = data[m][n+1] + 1;
                int c = data[m][n] + 1;
                data[m+1][n+1] = min(a, min(b, c));
            }
        }
    }
    return data[left.size()][right.size()];
}