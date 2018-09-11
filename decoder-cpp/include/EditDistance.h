//
// Created by augustus on 8/30/18.
//

#ifndef CTC_DECODER_EDITDISTANCE_H
#define CTC_DECODER_EDITDISTANCE_H
#include <string>

using namespace std;

class EditDistance {
public:
    EditDistance();
    ~EditDistance();
    int edit_distance(string &left, string &right);
};


#endif //CTC_DECODER_EDITDISTANCE_H
