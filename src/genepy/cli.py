# -*- coding: utf-8 -*-

import os

import click
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from .utils import cross_annotate_cadd, calculate_genepy, chunks, run_parallel


@click.group()
def main():
    """Handle genepy functions."""


@main.command()
@click.option('--vcf-file', required=True, help='')
@click.option('--annovar-ready-file', required=True, help='')
@click.option('--annotated-file', required=True, help='')
@click.option('--caddout-file', required=True, help='')
@click.option('--output-path', required=True, help='')
def cross_annotate(
    *,
    vcf_file,
    annovar_ready_file,
    annotated_file,
    caddout_file,
    output_path
):
    a1 = pd.read_csv(annovar_ready_file, sep='\t', index_col=False, header=None)
    geneanno = a1.drop(a1.columns[:17], axis=1)
    b1 = pd.read_csv(vcf_file, sep='\t', index_col=False, skiprows=6)
    b1 = b1.drop(b1.columns[:9], axis=1)
    geneanno.columns = list(b1.columns)
    freqanno = pd.read_csv(annotated_file, sep='\t', index_col=False, usecols=[0, 1, 3, 4, 5, 6, 10])
    cadd_df = pd.read_csv(caddout_file, sep='\t', skiprows=1, index_col=False)
    raw_scores = cross_annotate_cadd(freq_df=freqanno, cadd_df=cadd_df)
    caddanno = pd.DataFrame(raw_scores, columns=['CADD15_RAW'])
    final_df = pd.concat([freqanno, caddanno, geneanno], axis=1)
    final_df.to_csv(output_path, index=False, sep='\t')
    return final_df


@main.command()
@click.option('--genepy-meta', required=True, help='')
@click.option('--output-dir', required=True, help='')
@click.option('--gene-list', required=True, help='')
@click.option('--score-col', required=True, help='')
def get_genepy(
    *,
    genepy_meta,
    output_dir,
    gene_list,
    score_col,
):
    os.mkdir(output_dir)
    meta_data = pd.read_csv(genepy_meta, sep='\t', index_col=False)
    with open(gene_list) as file:
        genes = list(file)
    gene_chunks = list(chunks(genes, 400))
    pool = Pool(processes=5)
    func = partial(run_parallel, meta_data, score_col, output_dir)
    pool.map(func, gene_chunks)
    return "Scores are ready in " + output_dir
