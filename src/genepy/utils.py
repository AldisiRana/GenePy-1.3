# -*- coding: utf-8 -*-
import os
import subprocess

import click
import numpy as np
import pandas as pd
import re
from pybiomart import Dataset
import scipy.stats as stats
from tqdm import tqdm


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
    known_fa_all = np.array(gene_df['AF'])
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


def score_db(samples, score, freq):
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


def run_parallel(header, meta_data, score_col, output_dir, genes):
    for gene in genes:
        if os.path.isfile(os.path.join(output_dir, gene+'_'+score_col+'_matrix')):
            click.echo('Scoring matrix exists!')
            continue
        if not os.path.isfile(gene+'.meta'):
            p = subprocess.call(['cp', header, gene+'.meta'])
            with open(gene+'.meta', 'a+') as file:
                p = subprocess.call('grep -E "\W'+gene+';?\s" '+meta_data, stdout=file, shell=True)
        gene_df = pd.read_csv(gene+'.meta', sep='\t', index_col=False)
        if gene_df.empty:
            click.echo("Error! Gene not found!")
            p = subprocess.call(['rm', gene + '.meta'])
            continue
        scores_matrix = calculate_genepy(gene_df, score_col)
        path = os.path.join(output_dir, gene+'_'+score_col+'_matrix')
        np.savetxt(path, scores_matrix, fmt='%s', delimiter='\t')
        p = subprocess.call(['rm', gene+'.meta'])


def merge_matrices(
    *,
    directory,
    output_path,
):
    """
    Merges multiple files in a directory, each file should contain the score of a gene across the samples.

    :param directory: the directory that contains files to merge.
    :param output_path: the path for the merged tsv file.
    :return: a dataframe combining all information from files.
    """
    full_data = pd.DataFrame(data=None, columns=['patient_id'])
    for filename in tqdm(os.listdir(directory), desc="merging matrices"):
        data = pd.read_csv(os.path.join(directory, filename), sep='\t',
                           names=['patient_id', filename.split('_')[0], ''])
        data = data.drop(columns=[''])
        full_data = pd.merge(data, full_data, on='patient_id', how='left')
    full_data.to_csv(output_path, sep='\t', index=False)
    return full_data


def normalize_gene_len(
    *,
    genes_lengths_file=None,
    matrix_file,
    output_path,
):
    """
    Normalize matrix by gene length.

    :param genes_lengths_file: a file containing genes, and their start and end bps.
    :param matrix_file: a tsv file containing a matrix of samples and their scores across genes.
    :param output_path: the path to save the normalized matrix.
    :return: a normalized dataframe.
    """
    if genes_lengths_file:
        genes_df = pd.read_csv(genes_lengths_file, sep='\t')
    else:
        gene_dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
        genes_df = gene_dataset.query(
            attributes=['external_gene_name', 'start_position', 'end_position'],
            only_unique=False,
        )
    genes_lengths = {
        row['Gene name']: round((row['Gene end (bp)'] - row['Gene start (bp)']) / 1000, 3)
        for _, row in genes_df.iterrows()
    }
    scores_df = pd.read_csv(matrix_file, sep='\t')
    unnormalized = []
    for (name, data) in tqdm(scores_df.iteritems(), desc="Normalizing genes scores"):
        if name == 'patient_id':
            continue
        if name not in genes_lengths.keys():
            unnormalized.append(name)
            continue
        # normalize genes by length
        scores_df[name] = round(scores_df[name] / genes_lengths[name], 5)
    # drop genes with unknown length
    scores_df = scores_df.drop(unnormalized, axis=1)
    if output_path:
        scores_df.to_csv(output_path, sep='\t', index=False)
    return scores_df


def find_pvalue(
    *,
    scores_file,
    genotype_file,
    output_file,
    genes=None,
    cases_column,
):
    """
    Calculate the significance of a gene in a population using Mann-Whitney-U test.
    :param scores_file: a tsv file containing the scores of genes across samples.
    :param genotype_file: a file containing the information of the sample.
    :param output_file: a path to save the output file.
    :param genes: a list of the genes to calculate the significance. if None will calculate for all genes.
    :param cases_column: the name of the column containing cases and controls information.
    :return: dataframe with genes and their p_values
    """
    scores_df = pd.read_csv(scores_file, sep='\t', index_col=False)
    genotype_df = pd.read_csv(genotype_file, sep=' ', index_col=False)
    merged_df = pd.merge(genotype_df, scores_df, on='patient_id', how='left')
    df_by_cases = merged_df.groupby(cases_column)
    cases = list(df_by_cases.groups.keys())
    p_values = []
    if genes is None:
        genes = scores_df.columns.tolist()[1:]
    for gene in tqdm(genes, desc='Calculating p_values for genes'):
        case_0 = df_by_cases.get_group(cases[0])[gene].tolist()
        case_1 = df_by_cases.get_group(cases[1])[gene].tolist()
        u_statistic, p_val = stats.mannwhitneyu(case_0, case_1)
        p_values.append([gene, u_statistic, p_val])
    p_values_df = pd.DataFrame(p_values, columns=['genes', 'u_statistic', 'p_value']).sort_values(by=['p_value'])
    p_values_df.to_csv(output_file, sep='\t', index=False)
    return p_values_df
