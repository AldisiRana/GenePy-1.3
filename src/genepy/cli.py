# -*- coding: utf-8 -*-

import os
import subprocess
from contextlib import contextmanager

import click
import pandas as pd
from multiprocessing import Pool
from functools import partial

from .utils import cross_annotate_cadd, chunks, run_parallel, find_pvalue, normalize_gene_len, \
    merge_matrices


@click.group()
def main():
    """Handle genepy functions."""


@main.command()
@click.option('--vcf-file', required=True, help='the filtered vcf file.')
@click.option('--annovar-ready-file', required=True, help='the annover prepared file.')
@click.option('--annotated-file', required=True, help='the file with annotations.')
@click.option('--caddout-file', required=True, help='the output file from CADD.')
@click.option('--output-path', required=True, help='the path for the output file.')
def cross_annotate(
    *,
    vcf_file,
    annovar_ready_file,
    annotated_file,
    caddout_file,
    output_path
):
    click.echo("Reading annotated file ...")
    freqanno = pd.read_csv(annotated_file, sep='\t', index_col=False, usecols=[0, 1, 3, 4, 5, 6, 10])
    click.echo("Reading caddout file ...")
    cadd_df = pd.read_csv(caddout_file, sep='\t', skiprows=1, index_col=False)
    click.echo("Combine Genotypes and annotations")
    raw_scores = cross_annotate_cadd(freq_df=freqanno, cadd_df=cadd_df)
    caddanno = pd.DataFrame(raw_scores, columns=['CADD15_RAW'])
    click.echo("Reading vcf file ...")
    b1 = pd.read_csv(vcf_file, sep='\t', index_col=False, skiprows=34)
    b1 = b1.drop(b1.columns[:9], axis=1)
    click.echo("Reading annovar input file ...")
    geneanno = pd.read_csv(annovar_ready_file, sep='\t', index_col=False, header=None,
                     usecols=range(17, len(b1.columns) + 17))
    geneanno.columns = list(b1.columns)
    click.echo("Merging all files ...")
    final_df = pd.concat([freqanno, caddanno, geneanno], axis=1)
    final_df.to_csv(output_path, index=False, sep='\t')
    click.echo("Process is done.")
    return final_df


@main.command()
@click.option('--genepy-meta', required=True, help='The meta file with all variants and samples')
@click.option('--output-dir', required=True, help='the directory path for all scores')
@click.option('--gene-list', required=True, help='a list of all the genes to score')
@click.option('--score-col', required=True, help='the name of the score column in genepy meta.')
@click.option('--header', default=None, help='The file containing the header')
@click.option('--processes', default=5, help='Number of processes to run in parallel.')
@click.option('--chunk-size', default=400, help='the size of the chunk to split the genes.')
def get_genepy(
    *,
    genepy_meta,
    output_dir,
    gene_list,
    score_col,
    header,
    processes,
    chunk_size,
):
    """
    Caluclate genepy scores of genes across the samples.

    :param genepy_meta: The meta file with all variants and samples
    :param output_dir: the directory path for all scores
    :param gene_list: a list of all the genes to score
    :param score_col: the name of the score column in genepy meta.
    :param header: the file containing the header
    :param processes: Number of processes to run in parallel.
    :param chunk_size: the size of the chunk to split the genes.
    :return:
    """
    if header is None:
        click.echo('Creating header file ... ')
        p = subprocess.call('grep "^Chr" '+genepy_meta+'> header', shell=True)
	header = 'header'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # click.echo('Reading input dataframe ... ')
    # meta_data = pd.read_csv(genepy_meta, sep='\t', index_col=False, chunksize=1000000)
    # df = pd.concat(meta_data)
    click.echo('Processing gene list ... ')
    with open(gene_list) as file:
        genes = [line.rstrip('\n') for line in file]
    gene_chunks = list(chunks(genes, chunk_size))
    click.echo('Calculating genepy scores ... ')
    func = partial(run_parallel, header, genepy_meta, score_col, output_dir)
    with poolcontext(processes=processes) as pool:
        pool.map(func, gene_chunks)
    return "Scores are ready in " + output_dir


@main.command()
@click.option('-d', '--directory', required=True, help="The directory that contains the matrices to merge.")
@click.option('-o', '--output-path', required=True, help="The path to output the merged matrix.")
def merge(
    *,
    directory,
    output_path
):
    """This command merges all matrices in a directory into one big matrix"""
    click.echo("Starting the merging process")
    merge_matrices(
        directory=directory,
        output_path=output_path,
    )
    click.echo("Merging is done.")


@main.command()
@click.option('-m', '--matrix-file', required=True, help="The scoring matrix to normalize.")
@click.option('-g', '--genes-lengths-file',
              help="The file containing the lengths of genes. If not provided it will be produced.")
@click.option('-o', '--output-path', required=True, help="The path to output the normalized matrix.")
def normalize(
    *,
    matrix_file,
    genes_lengths_file=None,
    output_path=None
):
    """This command normalizes the scoring matrix by gene length."""
    click.echo("Normalization in process.")
    normalize_gene_len(
        matrix_file=matrix_file,
        genes_lengths_file=genes_lengths_file,
        output_path=output_path,
    )


@main.command()
@click.option('-s', '--scores-file', required=True, help="The scoring file of genes across a population.")
@click.option('-i', '--genotype-file', required=True, help="File containing information about the cohort.")
@click.option('-o', '--output-file', required=True, help="The path to output the pvalues of genes.")
@click.option('-g', '--genes',
              help="a list containing the genes to calculate. if not provided all genes will be used.")
@click.option('-t', '--test', required=True, type=click.Choice(['ttest_ind', 'mannwhitneyu']),
              help='statistical test for calculating P value.')
@click.option('-c', '--cases-column', required=True, help="the name of the column that contains the case/control type.")
def calculate_pval(
    *,
    scores_file,
    genotype_file,
    output_file,
    genes,
    cases_column,
    test,
):
    click.echo("The process for calculating the p_values will start now.")
    df = find_pvalue(
        scores_file=scores_file,
        output_file=output_file,
        genotype_file=genotype_file,
        genes=genes,
        cases_column=cases_column,
        test=test,
    )
    click.echo('Process is complete.')
    click.echo(df.info())


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()


if __name__ == "__main__":
    main()

