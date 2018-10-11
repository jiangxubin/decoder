//
// Created by augustus on 8/30/18.
//

#ifndef CTC_DECODER_EDITDISTANCE_H
#define CTC_DECODER_EDITDISTANCE_H
#include <string>

using namespace std;


/// Class which provides a interface of calculating the edit distance between two strings.
class EditDistance {
public:
    EditDistance();
    ~EditDistance();
    /// Main function of calculating edit distance
    int edit_distance(string &left, string &right);
};


#endif //CTC_DECODER_EDITDISTANCE_H
