# -*- coding: utf-8 -*-

import os
import subprocess
from contextlib import contextmanager

import click
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
    click.echo("Reading annotated file ...")
    freqanno = pd.read_csv(annotated_file, sep='\t', index_col=False, usecols=[0, 1, 3, 4, 5, 6, 10])
    click.echo("Reading caddout file ...")
    cadd_df = pd.read_csv(caddout_file, sep='\t', skiprows=1, index_col=False)
    click.echo("Combine Genotypes and annotations")
    raw_scores = cross_annotate_cadd(freq_df=freqanno, cadd_df=cadd_df)
    caddanno = pd.DataFrame(raw_scores, columns=['CADD15_RAW'])
    click.echo("Reading vcf file ...")
    b1 = pd.read_csv(vcf_file, sep='\t', index_col=False, skiprows=6)
    b1 = b1.drop(b1.columns[:9], axis=1)
    click.echo("Reading annovar input file ...")
    a1 = pd.read_csv(annovar_ready_file, sep='\t', index_col=False, header=None,
                     usecols=range(17, len(b1.columns) + 17))
    geneanno = a1.drop(a1.columns[:17], axis=1)
    geneanno.columns = list(b1.columns)
    click.echo("Merging all files ...")
    final_df = pd.concat([freqanno, caddanno, geneanno], axis=1)
    final_df.to_csv(output_path, index=False, sep='\t')
    click.echo("Process is done.")
    return final_df


@main.command()
@click.option('--genepy-meta', required=True, help='')
@click.option('--output-dir', required=True, help='')
@click.option('--gene-list', required=True, help='')
@click.option('--score-col', required=True, help='')
@click.option('--header', default=None)
@click.option('--processes', default=5)
@click.option('--chunk-size', default=400)
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
    if header is None:
        click.echo('Creating header file ... ')
        p = subprocess.call('grep "^Chr" '+genepy_meta+'> header', shell=True)
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
        results = pool.map(func, gene_chunks)
    return "Scores are ready in " + output_dir


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

