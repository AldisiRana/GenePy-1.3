# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:51:25 2019

@author: Enrico
"""


def cross_annotate_cadd(
    *,
    freq_df,
    cadd_df,
):
    freq = [(val[0], val[1]) for val in freq_df.values]
    cadd = {
        (val[0], val[1]): val[-2] for val in cadd_df.values
    }
    scores = {}
    for val in freq:
        if val not in cadd.keys():
            val = (val[0], val[1]-1)
        try:
            scores[val] = cadd[val]
        except:
            scores[val] = 'NaN'
            print('Fail to annotate ' + str(val))
    return scores


if __name__ == '__main__':
    cross_annotate_cadd()
