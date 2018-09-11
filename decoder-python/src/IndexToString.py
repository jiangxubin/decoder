"""
Convert char index from beam search to correspondding string
"""
map_path = r"../data/ocr_char_set.txt"


class IndexToString:
    def __init__(self):
        self.map = {}
        with open(map_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                result = line.split()
                self.map.update({result[0]: result[1]})

    def index_to_string(self, index_data: list)->str:
        """
        Query the ocr_char_set.txt and find the string based on the index
        :param index_data:a set of all indexes
        :return:result string
        """
        result = ""
        for index in index_data:
            # print(index)
            # print(str(index))
            # print(type(str(index)))
            # print("corresponding char "+self.map.get(str(index)))
            # print(type(result))
            result += self.map.get(str(index))
        return result

