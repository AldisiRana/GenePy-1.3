# -*- coding: utf-8 -*-
import gzip
from contextlib import contextmanager

import click
import gc
import numpy as np
import modin.pandas as pd
from multiprocessing import Pool
import re
from scipy.stats import beta

from tqdm import tqdm


def cross_annotate_cadd(
    *,
    freq_df,
    cadd_df,
):
    freq = [(val[0], val[1], val[2], val[3]) for val in freq_df.values]
    cadd = {
        (val[0], val[1], val[2], val[3]): val[-2] for val in cadd_df.values
    }
    scores = {}
    for val in freq:
        if val not in cadd.keys():
            val = (val[0], val[1] - 1, val[2], val[3])
        try:
            scores[val] = cadd[val]
        except:
            scores[val] = 'NaN'
            print('Fail to annotate ' + str(val))
    return scores.values()


def preprocess_df(gene_df, score_col):
    scores = np.array(gene_df[score_col])
    scores = scores.astype('float')
    scores = (scores - (-7.535037)) / (35.788538 - (-7.535037))
    gene_df = gene_df.drop([score_col], axis=1)
    gene_df = gene_df.replace(to_replace='0/0+', value=0, regex=True)
    gene_df = gene_df.replace(to_replace='0/[123456789]+', value=1, regex=True)
    gene_df = gene_df.replace(to_replace='[123456789]/[123456789]', value=2, regex=True)
    gene_df = gene_df.replace(to_replace='\./\.[\S]*', value=0, regex=True)
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

    samples = [c for c in gene_df if re.match("[a-zA-Z0-9]+_+[0-9]", c)]
    samples_df = gene_df[samples]
    samples_df = samples_df.astype(float) / 2.00
    return samples_df, scores, freqs


def score_db(*, samples, score, freq, weight_function='log10', a=1, b=25):
    # first make copies of the score and samples into S and db1, respectively
    s = np.copy(score)
    db1 = samples.to_numpy()

    out1 = []
    if weight_function == 'log10':
        for i in range(db1.shape[0]):
            if ~np.isnan(s[i]):  # if deleteriousness score is available
                deleter = float(s[i])  # store the score value into the into the deleter variable
                db1[i][db1[i] == 0.5] = deleter * -np.log10(
                    float(freq[i, 0]) * float(freq[i, 1]))  # compute GenePy score for heterozygous variants
                db1[i][db1[i] == 1] = deleter * -np.log10(float(freq[i, 1]) * float(freq[i, 1]))
                out1.append(db1[i])  # compute GenePy score for homozygous variants
    elif weight_function == 'beta':
        for i in range(db1.shape[0]):
            if ~np.isnan(s[i]):  # if deleteriousness score is available
                deleter = float(s[i])  # store the score value into the into the deleter variable
                db1[i][db1[i] == 0.5] = deleter * beta.pdf((
                    float(freq[i, 0]) * float(freq[i, 1])), a, b)  # compute GenePy score for heterozygous variants
                db1[i][db1[i] == 1] = deleter * beta.pdf((float(freq[i, 1]) * float(freq[i, 1])), a, b)
                out1.append(db1[i])  # compute GenePy score for homozygous variants
    else:
        return Exception('The chosen weighting function is not available')

    out1 = np.array(out1)  # then these values will be stored into the out1 array.
    out1 = np.sum(out1, axis=0)  # the out1 is then condensed by suming the columns for each sample.

    samples_header = samples.columns
    final = np.vstack((samples_header, out1)).T
    return final


def chunks(genes, x):
    for i in range(0, len(genes), x):
        yield genes[i:i+x]


def combine_genotype_annotation(
    *,
    vcf_file,
    annovar_ready_file,
    annotated_file,
    scores_col,
):
    if vcf_file.endswith('.gz'):
        with gzip.open(vcf_file, 'rb') as f:
            for line in f.readlines():
                if line.startswith(b'#CHROM'):
                    line = line.decode('utf-8')
                    header = line.strip().split("\t")[9:]
                    break
    else:
        with open(vcf_file, 'rb') as f:
            for line in f:
                if line.startswith(b'#CHROM'):
                    line = line.decode('utf-8')
                    header = line.strip().split("\t")[9:]
                    break
    geneanno = pd.read_csv(annovar_ready_file, sep='\t', header=None).iloc[:, 17:]
    geneanno.columns = header
    freqanno = pd.read_csv(
        annotated_file, sep='\t',
        usecols=['Chr', 'Start', 'Ref', 'Alt', 'Func.refGene', 'Gene.refGene', 'AF']+scores_col)
    click.echo("Combine Genotypes and annotations")
    full_df = pd.concat([freqanno, geneanno], axis=1)
    return full_df


def score_genepy(
    *,
    genepy_meta,
    genes,
    score_col,
    excluded=None,
    weight_function='log10',
    a=1,
    b=25
):
    full_df = pd.DataFrame()
    for gene in tqdm(genes, desc='Getting scores for genes'):
        gene_df = genepy_meta.loc[genepy_meta['Gene.refGene'] == gene]
        gene_df.loc[(gene_df[score_col] == '.'), score_col] = np.nan
        if gene_df[score_col].isnull().all():
            if excluded:
                with open(excluded, "a") as f:
                    f.write(gene + "\n")
            continue
        samples_df, scores, freqs = preprocess_df(gene_df, score_col)
        scores_matrix = score_db(
            samples=samples_df, score=scores, freq=freqs, weight_function=weight_function, a=a, b=b
        )
        score_df = pd.DataFrame(scores_matrix, columns=['sample_id', gene])
        full_df = full_df.append(score_df)
        del gene_df, score_df, samples_df, scores, freqs
        gc.collect()
    return full_df


def create_genes_list(filepath):
    gene_list = pd.read_csv(filepath, sep='\t', usecols=['Gene.refGene'])
    genes = list(gene_list['Gene.refGene'].unique())
    if '.' in genes:
        genes.remove('.')
    return genes


def gzip_reader(file_name):
    for row in gzip.open(file_name, "rb"):
        yield row


def file_reader(file_name):
    for row in open(file_name, "rb"):
        yield row


def process_annotated_vcf(vcf):
    if vcf.endswith('.gz'):
        file_gen = gzip_reader(vcf)
    else:
        file_gen = file_reader(vcf)
    i = 0
    for row in file_gen:
        if row.startswith(b'##'):
            i += 1
        else:
            break
    df = pd.read_csv(vcf, skiprows=i, sep='\t')
    for ind, row in df.iterrows():
        for value in row['INFO'].split(';'):
            if len(value.split('=')) < 2:
                continue
            if value.split('=')[0] not in df.columns:
                df[value.split('=')[0]] = "NaN"
            df.at[ind, value.split('=')[0]] = value.split('=')[1]
    df = df.drop(columns=['INFO'])
    df = df.rename(columns={"gene": 'Gene.refGene'})
    return df


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()