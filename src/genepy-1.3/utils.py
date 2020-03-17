# -*- coding: utf-8 -*-
import os

import click
import numpy as np
import pandas as pd
import re

from .make_scores_mat_6 import score_db


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
            val = (val[0], val[1] - 1)
        try:
            scores[val] = cadd[val]
        except:
            scores[val] = 'NaN'
            print('Fail to annotate ' + str(val))
    return scores


def calculate_genepy(gene_df, score_col):

    gene_df = gene_df.replace(to_replace='0/0+', value=0, regex=True)
    gene_df = gene_df.replace(to_replace='0/[123456789]+', value=1, regex=True)
    gene_df = gene_df.replace(to_replace='[123456789]/[123456789]', value=2, regex=True)
    gene_df = gene_df.replace(to_replace='\./\.[\S]*', value=0, regex=True)
    scores = np.array(gene_df[score_col])
    scores[scores == '.'] = np.nan
    scores = scores.astype('float')
    scores = (scores - (-7.535037))/(35.788538-(-7.535037))
    known_fa_all = np.array(gene_df.filter(regex='exome_ALL'))
    known_fa_all[known_fa_all == '.'] = np.nan
    known_fa_all = known_fa_all.astype('float')
    freqs = np.zeros((gene_df.shape[0], 2))
    freqs[:] = np.nan
    freqs[:, 1] = list(known_fa_all)
    freqs[:, 1][np.isnan(freqs[:, 1])] = 3.98e-6  # 1 allele out 125,748 indiv in gnomADexome (251496 alleles)
    freqs[:, 1][freqs[:, 1] == 0] = 3.98e-6
    freqs[:, 1][freqs[:, 1] == 1] = 1 - 3.98e-6
    freqs[:, 0] = 1 - freqs[:, 1]

    samples = [c for c in gene_df if re.match("\d", c[0])]
    samples_df = gene_df[samples]
    samples_df = samples_df.astype(float) / 2.00
    matrix = score_db(samples_df, scores, freqs)

    return matrix


def score_db(gene, samples, score, freq):
    # first make copies of the score and samples into S and db1, respectively
    samples_header = samples.columns
    S = np.copy(score)
    db1 = samples.to_numpy()

    out1 = []
    for i in range(db1.shape[0]):
        if ~np.isnan(S[i]):  # if deleteriousness score is available
            deleter = float(S[i])  # store the score value into the into the deleter variable
            db1[i][db1[i] == 0.5] = deleter * -np.log10(
                float(freq[i, 0]) * float(freq[i, 1]))  # compute GenePy score for heterozygous variants
            db1[i][db1[i] == 1] = deleter * -np.log10(float(freq[i, 1]) * float(freq[i, 1]))
            out1.append(db1[i])  # compute GenePy score for homozygous variants

    out1 = np.array(out1)  # then these values will be stored into the out1 array.

    out1 = np.sum(out1, axis=0)  # the out1 is then condensed by suming the columns for each sample.

    # formmating the data into columns of the Sample Heading, Value, Gene Name.
    U = np.vstack((samples_header, out1)).T

    return U


def chunks(genes, x):
    for i in range(0, len(genes), x):
        yield genes[i:i+x]


def run_parallel(meta_data, score_col, output_dir, genes):
    for gene in genes:
        gene_df = meta_data.loc[meta_data['Gene.refGene'] == gene]
        if gene_df.empty:
            click.echo("Error! Gene not found!")
            continue
        scores_matrix = calculate_genepy(gene_df, score_col)
        path = os.path.join(output_dir, gene+'_'+score_col+'_matrix')
        np.savetxt(path, scores_matrix, fmt='%s', delimiter='\t')