"""
Library used to calculate edit-distance between two strings
"""
import numpy as np


def edit_distance(left: str, right: str)->int:
    """
    Quick function to calculate edit-distacne
    :param left: string one
    :param right: string two
    :return: edit distance
    """
    if left is None:
        return len(right)
    if right is None:
        return len(left)

    dis = np.zeros((len(left)+1, len(right)+1))
    for m in range(len(right)+1):
        dis[0][m] = m
    for n in range(len(left)+1):
        dis[n][0] = n
    for i in range(len(left)):
        for j in range(len(right)):
            if left[i] == right[j]:
                dis[i+1][j+1] = dis[i][j]
            else:
                a = dis[i][j+1] + 1
                b = dis[i+1][j] + 1
                c = dis[i][j] + 1
                dis[i+1][j+1] = min(a, min(b, c))

    return dis[len(left)][len(right)]


if __name__ == "__main__":
    test_l = "FITNESSSYSTEM"
    test_r = 'FITNESSSYSTEM'
    res = edit_distance(test_l, test_r)
    print("Stop")
