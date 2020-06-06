# -*- coding: utf-8 -*-
import gzip

import click
import numpy as np
import pandas as pd
import re


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

    samples = [c for c in gene_df if re.match("\d", c[0])]
    samples_df = gene_df[samples]
    samples_df = samples_df.astype(float) / 2.00
    return samples_df, scores, freqs


def score_db(samples, score, freq):
    # first make copies of the score and samples into S and db1, respectively
    s = np.copy(score)
    db1 = samples.to_numpy()

    out1 = []
    for i in range(db1.shape[0]):
        if ~np.isnan(s[i]):  # if deleteriousness score is available
            deleter = float(s[i])  # store the score value into the into the deleter variable
            db1[i][db1[i] == 0.5] = deleter * -np.log10(
                float(freq[i, 0]) * float(freq[i, 1]))  # compute GenePy score for heterozygous variants
            db1[i][db1[i] == 1] = deleter * -np.log10(float(freq[i, 1]) * float(freq[i, 1]))
            out1.append(db1[i])  # compute GenePy score for homozygous variants

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
    else:
        with open(vcf_file, 'rb') as f:
            for line in f:
                if line.startswith(b'#CHROM'):
                    line = line.decode('utf-8')
                    header = line.strip().split("\t")[9:]
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
    excluded
):
    full_df = pd.DataFrame(columns=['sample_id'])
    for gene in genes:
        gene_df = genepy_meta.loc[genepy_meta['Gene.refGene'] == gene]
        gene_df[score_col] = gene_df[score_col].replace('.', np.nan)
        if gene_df[score_col].isnull().all():
            with open(excluded, "a") as f:
                f.write(gene + "\n")
            click.echo('Gene does not have deleteriousness score!')
            continue
        samples_df, scores, freqs = preprocess_df(gene_df, score_col)
        scores_matrix = score_db(samples_df, scores, freqs)
        score_df = pd.DataFrame(scores_matrix, columns=['sample_id', gene])
        full_df = pd.merge(full_df, score_df, how='right')
    return full_df


def create_genes_list(filepath):
    gene_list = pd.read_csv(filepath, sep='\t', usecols=['Gene.refGene'])
    genes = list(gene_list['Gene.refGene'].unique())
    if '.' in genes:
        genes.remove('.')
    return genes
