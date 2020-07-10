# -*- coding: utf-8 -*-

import os
import subprocess
from functools import partial

import click
import numpy as np
import pandas as pd
from pybiomart import Dataset
import scipy.stats as stats
import statsmodels.api as sm
from tqdm import tqdm
import gc

from .utils import preprocess_df, score_db, score_genepy, process_annotated_vcf, \
    poolcontext


def run_parallel_genes_meta(header, meta_data, score_col, output_dir, excluded, genes):
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
        gene_df[score_col].replace('.', np.nan)
        if gene_df[score_col].isnull().all():
            with open(excluded, "a") as f:
                f.write(gene + "\n")
            click.echo('Gene does not have deleteriousness score!')
            p = subprocess.call(['rm', gene + '.meta'])
            continue
        samples_df, scores, freqs = preprocess_df(gene_df, score_col)
        scores_matrix = score_db(samples_df, scores, freqs)
        path = os.path.join(output_dir, gene+'_'+score_col+'_matrix')
        np.savetxt(path, scores_matrix, fmt='%s', delimiter='\t')
        p = subprocess.call(['rm', gene+'.meta'])


def run_parallel_annovar(del_m, build, output_dir, vcf):
    process_annovar(vcf, del_m, build, output_dir)


def run_parallel_scoring(combined_df, genes, output_file, excluded, score_col):
    scores_df = score_genepy(
        genepy_meta=combined_df, genes=genes, score_col=score_col, excluded=excluded
    )
    scores_df.to_csv(score_col + output_file, sep='\t', index=False)


def parallel_annotated_vcf_prcoessing(gene_list, scores_col, output_file, excluded, processes, vcf):
    df = process_annotated_vcf(vcf)
    if gene_list:
        with open(gene_list) as file:
            genes = [line.rstrip('\n').encode() for line in file]
    else:
        click.echo('Creating genes list')
        genes = list(gene_list['Gene.refGene'].unique())
        if '.' in genes:
            genes.remove('.')
    if len(scores_col) == 1:
        scores_df = score_genepy(
            genepy_meta=df, genes=genes, score_col=scores_col[0], excluded=excluded
        )
        scores_df.to_csv(output_file, sep='\t', index=False)
        del scores_df, df
        gc.collect()
    else:
        func = partial(run_parallel_scoring, df, genes, output_file, excluded)
        with poolcontext(processes=processes) as pool:
            pool.map(func, scores_col)
    click.echo('Scoring is complete.')


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
    pc_file=None,
    test='mannwhitneyu',
):
    """
    Calculate the significance of a gene in a population using Mann-Whitney-U test.
    :param pc_file:
    :param test:
    :param scores_file: a tsv file containing the scores of genes across samples.
    :param genotype_file: a file containing the information of the sample.
    :param output_file: a path to save the output file.
    :param genes: a list of the genes to calculate the significance. if None will calculate for all genes.
    :param cases_column: the name of the column containing cases and controls information.
    :return: dataframe with genes and their p_values
    """
    scores_df = pd.read_csv(scores_file, sep='\t', index_col=False)
    genotype_df = pd.read_csv(genotype_file, sep=' ', index_col=False)
    merged_df = pd.merge(genotype_df, scores_df, on='patient_id', how='right')
    df_by_cases = merged_df.groupby(cases_column)
    cases = list(df_by_cases.groups.keys())
    p_values = []
    if genes is None:
        genes = scores_df.columns.tolist()[1:]
    if test == 'mannwhitneyu':
        for gene in tqdm(genes, desc='Calculating p_values for genes'):
            case_0 = df_by_cases.get_group(cases[0])[gene].tolist()
            case_1 = df_by_cases.get_group(cases[1])[gene].tolist()
            try:
                u_statistic, p_val = stats.mannwhitneyu(case_0, case_1)
            except:
                continue
            p_values.append([gene, u_statistic, p_val])
        p_values_df = pd.DataFrame(p_values, columns=['genes', 'statistic', 'p_value']).sort_values(by=['p_value'])
    elif test == 'ttest_ind':
        for gene in tqdm(genes, desc='Calculating p_values for genes'):
            case_0 = df_by_cases.get_group(cases[0])[gene].tolist()
            case_1 = df_by_cases.get_group(cases[1])[gene].tolist()
            try:
                statistic, p_val = stats.ttest_ind(case_0, case_1)
            except:
                continue
            p_values.append([gene, statistic, p_val])
        p_values_df = pd.DataFrame(p_values, columns=['genes', 'statistic', 'p_value']).sort_values(by=['p_value'])
    elif test == 'logit':
        if not pc_file:
            raise Exception("Need principle components file.")
        pc_df = pd.read_csv(pc_file, sep='\t', index_col=False)
        merged_df = pd.merge(merged_df, pc_df, on='patient_id')
        for gene in tqdm(genes, desc='Calculating p_values for genes'):
            X = merged_df[[gene, 'PC1', 'PC2', 'PC3']]
            X = sm.add_constant(X)
            Y = merged_df[[cases_column]]
            try:
                logit_model = sm.Logit(Y, X)
                result = logit_model.fit()
            except:
                continue
            pval = list(result.pvalues)
            #add beta coeff
            p_values.append([gene]+pval)
        p_values_df = pd.DataFrame(
            p_values, columns=['genes', 'const_pval', 'p_value', 'PC1_pval', 'PC2_pvcal', 'PC3_pval']
        ).sort_values(by=['p_value'])
    else:
        raise Exception("The test you selected is not valid.")
    p_values_df.to_csv(output_file, sep='\t', index=False)
    return p_values_df


def process_annovar(vcf, del_m=None, build='hg38', output_dir=''):
    if del_m is None:
        del_m = ['cadd']
    sample = os.path.join(output_dir, vcf.split('/')[-1].split('.')[0])
    p = subprocess.call("./annovar/convert2annovar.pl -format vcf4 " + vcf +
                        " -outfile "+sample+".input  -allsample  -withfreq  -include 2>annovar.log", shell=True)
    f = ''
    m = ''
    for x in del_m:
        f = f+',f'
        m = m + ',' + x
    p = subprocess.call(
        "./annovar/table_annovar.pl " + sample + '.input' +
        " ./annovar/humandb/ -buildver " + build + " -out " + sample +
        " -remove -protocol refGene,gnomad211_exome" + m +" -operation g,f" + f +
        " --thread 40 -nastring . >>annovar.log",
        shell=True)


def cadd_scoring(vcf, output_dir=''):
    caddin = os.path.join(output_dir, vcf.split('/')[-1].split('.')[0] + '_caddin.vcf')
    p = subprocess.call('zgrep -v "^#" ' + vcf + ' >' + caddin, shell=True)
    p = subprocess.call("sed -i 's|^chr||g' " + caddin, shell=True)
    caddout = os.path.join(output_dir, vcf.split('/')[-1].split('.')[0] + '_caddout.tsv.gz ')
    p = subprocess.call(
        './CADD-scripts/CADD.sh -g GRCh38 -v v1.5 -o ' + caddout + caddin,
        shell=True)
    p = subprocess.call('rm '+caddin)

