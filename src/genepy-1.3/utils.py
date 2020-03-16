# -*- coding: utf-8 -*-

import click
import numpy as np
import pandas as pd


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


def calculate_genepy(gene_df):
    gene_df = gene_df.replace(to_replace='0/0+', value=0, regex=True)
    gene_df = gene_df.replace(to_replace='0/[123456789]+', value=1, regex=True)
    gene_df = gene_df.replace(to_replace='[123456789]/[123456789]', value=2, regex=True)
    gene_df = gene_df.replace(to_replace='\./\.[\S]*', value=0, regex=True)
    scores = np.array(gene_df['CADD13_RawScore'])
    scores[scores == '.'] = np.nan
    scores = scores.astype('float')
    known_fa_all = np.array(gene_df['gnomAD_exome_ALL'])
    known_fa_all[known_fa_all == '.'] = np.nan
    known_fa_all = known_fa_all.astype('float')
    # To be continued ...
    return