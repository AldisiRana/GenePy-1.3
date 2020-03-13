# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:51:25 2019

@author: Enrico
"""
import numpy as np
import pandas as pd


def cross_annotate_cadd(
    *,
    freqanno_file,
    caddout_file,
    output_file
):
    freq_df = pd.read_csv(freqanno_file, sep='\t', index_col=False)
    cadd_df = pd.read_csv(caddout_file, sep='\t', skiprows=1, index_col=False)
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
    np.savetxt(output_file, ["RawScore"]+list(scores.values()), fmt='%s')
    return scores
