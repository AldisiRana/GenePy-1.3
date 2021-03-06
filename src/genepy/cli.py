# -*- coding: utf-8 -*-

import os
import subprocess

import click
import pandas as pd
from functools import partial

from tqdm import tqdm

from .constants import SCORES_TO_COL_NAMES
from .pipeline import run_parallel_genes_meta, process_annovar, \
    cadd_scoring, run_parallel_annovar, run_parallel_scoring, annotated_vcf_prcoessing
from .utils import cross_annotate_cadd, chunks, score_genepy, combine_genotype_annotation, create_genes_list, \
    poolcontext

OUTPUT_PATH = click.option('-o', '--output-path', required=True, help='the path for the output file.')
WEIGHT_FUNCTION = click.option('--weight-function', default='log10', type=click.Choice(['log10', 'beta']),
              help='the weighting function for allele frequency')
A = click.option('-a', default=1.0, type=float, help='the alpha parameter for beta distribution')
B = click.option('-b', default=25.0, type=float, help='the b parameter for beta distribution')
PROCESSES = click.option('--processes', default=5, type=int, help='Number of processes to run in parallel.')
SCORES_COL = click.option('--scores-col', required=True, help="the name of scores column in matrices")
FILE_SEP = click.option('--file-sep', default='\t', help="the seperator for scores files")

@click.group()
def main():
    """Handle genepy functions."""


@main.command()
@click.option('--vcf-file', required=True, help='the filtered vcf file.')
@click.option('--annovar-ready-file', required=True, help='the annover prepared file.')
@click.option('--annotated-file', required=True, help='the file with annotations.')
@click.option('--caddout-file', default=None, help='the output file from CADD.')
@OUTPUT_PATH
def cross_annotate(
    *,
    vcf_file,
    annovar_ready_file,
    annotated_file,
    caddout_file,
    output_path
):
    click.echo('Extracting allele frequencies ... ')
    p = subprocess.call('cut -f 1,2,4,5,6,7,11 '+annotated_file+' >freqanno', shell=True)
    freqanno = pd.read_csv('freqanno', sep='\t', index_col=False)
    click.echo("Reading caddout file ...")
    cadd_df = pd.read_csv(caddout_file, sep='\t', skiprows=1, index_col=False)
    click.echo("Combine Genotypes and annotations")
    click.echo("Extracting genotypes ...")
    p = subprocess.call('cut -f 18- '+annovar_ready_file+' > a1', shell=True)
    p = subprocess.call("grep '^#CHR' "+vcf_file+" | cut -f 10- > b1", shell=True)
    p = subprocess.call('cat b1 a1 > geneanno', shell=True)
    click.echo("Merging all files ...")
    if caddout_file:
        raw_scores = cross_annotate_cadd(freq_df=freqanno, cadd_df=cadd_df)
        caddanno = pd.DataFrame(raw_scores, columns=['CADD15_RAW'])
        caddanno.to_csv('caddanno', sep='\t', index=False)
        p = subprocess.call('paste freqanno caddanno geneanno > ' + output_path, shell=True)
        p = subprocess.call('rm a1 b1 caddanno freqanno geneanno', shell=True)
    else:
        p = subprocess.call('paste freqanno geneanno > ' + output_path, shell=True)
        p = subprocess.call('rm a1 b1 freqanno geneanno', shell=True)
    click.echo("Process is done.")


@main.command()
@click.option('--genepy-meta', required=True, help='The meta file with all variants and samples')
@click.option('--output-dir', required=True, help='the directory path for all scores')
@click.option('--gene-list', required=True, help='a list of all the genes to score')
@SCORES_COL
@click.option('--af-col', default='AF', help='The name of the column with allele frequencies')
@click.option('--header', default=None, help='The file containing the header')
@PROCESSES
@click.option('--chunk-size', default=400, type=int, help='the size of the chunk to split the genes.')
@WEIGHT_FUNCTION
@A
@B
def get_genepy(
    *,
    genepy_meta,
    output_dir,
    gene_list,
    scores_col,
    af_col,
    header,
    processes,
    chunk_size,
    weight_function,
    a,
    b
):
    """
    Caluclate genepy scores of genes across the samples.

    :param genepy_meta: The meta file with all variants and samples
    :param output_dir: the directory path for all scores
    :param gene_list: a list of all the genes to score
    :param scores_col: the name of the score column in genepy meta.
    :param header: the file containing the header
    :param processes: Number of processes to run in parallel.
    :param chunk_size: the size of the chunk to split the genes.
    :return:
    """
    if header is None:
        click.echo('Creating header file ... ')
        p = subprocess.call('grep "^Chr" '+genepy_meta+'> header', shell=True)
        header = "header"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    click.echo('Processing gene list ... ')
    with open(gene_list) as file:
        genes = [line.rstrip('\n') for line in file]
    gene_chunks = list(chunks(genes, chunk_size))
    excluded = output_dir+'.excluded'
    open(excluded, 'a').close()
    click.echo('Calculating genepy scores ... ')
    func = partial(
        run_parallel_genes_meta, header, genepy_meta, scores_col, af_col, output_dir, excluded, weight_function, a, b
    )
    with poolcontext(processes=processes) as pool:
        pool.map(func, gene_chunks)
    return "Scores are ready in " + output_dir


@main.command()
@PROCESSES
@OUTPUT_PATH
@click.option('--vcf', required=True, help='The annotated vcf file')
@click.option('--scores-col', default='RawScore', help='scores column names must be provided')
@WEIGHT_FUNCTION
@A
@B
def score_annotated_vcf(
    *,
    vcf,
    scores_col,
    processes,
    output_path,
    weight_function,
    a,
    b,
):
    click.echo('processing annotated vcf')
    df = annotated_vcf_prcoessing(
        vcf=vcf,
        processes=processes,
        scores_col=scores_col,
        output_file=output_path,
        weight_function=weight_function,
        a=a,
        b=b
    )
    click.echo('genepy scoring is done.')


@main.command()
@click.option('--vcf-dir', required=True, help='the directory path with all meta files')
@click.option('--annotated-files-dir', default=None, help='the directory path for annotated files')
@click.option('--gene-list', default=None, help='a list of all the genes to score. if not provided it will be created')
@click.option('--del-matrix', default=['cadd'],
              type=click.Choice(['cadd', 'cadd13', 'dann', 'gwava', 'revel', 'eigen', 'ljb26_all']),
              help='one or multiple del matrices', multiple=True)
@click.option('--build', default='hg38', type=click.Choice(['hg18', 'hg19', 'hg38']),
              help='build version for annotations')
@OUTPUT_PATH
@PROCESSES
@WEIGHT_FUNCTION
@A
@B
def get_genepy_folder(
    *,
    vcf_dir,
    annotated_files_dir,
    gene_list,
    del_matrix,
    build,
    output_path,
    processes,
    weight_function,
    a,
    b
):
    vcf_files = []
    for file in os.listdir(vcf_dir):
        if file.endswith(('gvcf.gz', '.vcf', 'vcf.gz', 'gvcf', 'gz')):
            vcf_files.append(os.path.join(vcf_dir, file))
    excluded = output_path + '.excluded'
    open(excluded, 'w').close()
    del_anno_folder = click.confirm("Delete annotations folder before program termination?", abort=False)
    del_temp = True
    if not del_anno_folder:
        del_temp = click.confirm("Delete temporary files before program termination?", abort=False)
    if not os.path.isdir(annotated_files_dir):
        os.mkdir(annotated_files_dir)
    func = partial(run_parallel_annovar, del_matrix, build, annotated_files_dir)
    with poolcontext(processes=processes) as pool:
        pool.map(func, vcf_files)
    annotated_files = []
    input_files = []
    for file in os.listdir(annotated_files_dir):
        if file.endswith('.input'):
            input_files.append(os.path.join(annotated_files_dir, file))
        elif file.endswith('_multianno.txt'):
            annotated_files.append(os.path.join(annotated_files_dir, file))
    if len(annotated_files) != len(input_files):
        return Exception("Error in annovar processing! Files do not match!")
    if gene_list:
        with open(gene_list) as file:
            genes = [line.rstrip('\n') for line in file]
    else:
        genes = create_genes_list(annotated_files[0])
    for i in tqdm(range(len(annotated_files)), desc='Processing annotated files'):
        for matrix in del_matrix:
            combined_df = combine_genotype_annotation(
                vcf_file=vcf_files[i],
                annovar_ready_file=input_files[i],
                annotated_file=annotated_files[i],
                scores_col=SCORES_TO_COL_NAMES[matrix]
            )
            if len(SCORES_TO_COL_NAMES[matrix]) == 1:
                scores_df = score_genepy(
                    genepy_meta=combined_df,
                    genes=genes,
                    score_col=SCORES_TO_COL_NAMES[matrix][0],
                    excluded=excluded,
                    weight_function=weight_function,
                    a=a,
                    b=b,
                )
                scores_df.to_csv(output_path, sep='\t', index=False)
            else:
                func = partial(
                    run_parallel_scoring, combined_df, genes, output_path, excluded, weight_function, a, b
                )
                with poolcontext(processes=processes) as pool:
                    pool.map(func, SCORES_TO_COL_NAMES[matrix])
    click.echo('Scoring is complete.')
    if del_anno_folder:
        click.echo('Annotations folder will be deleted now!')
        os.removedirs(annotated_files_dir)
    elif del_temp:
        click.echo('Temporary files will be deleted now!')
        for file in os.listdir(annotated_files_dir):
            if file.endswith('.input') or file.endswith('_multianno.txt'):
                continue
            os.remove(file)


@main.command()
@click.option('--vcfs', required=True)
@click.option('--output-dir', default='')
def annovar_processing(
    *,
    vcfs,
    output_dir
):
    if os.path.isfile(vcfs):
        process_annovar(vcfs)
    elif os.path.isdir(vcfs):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for file in os.listdir(vcfs):
            if file.endswith(('gvcf.gz', '.vcf', 'vcf.gz', 'gvcf')):
                process_annovar(os.path.join(vcfs, file), output_dir)
    else:
        return Exception('Vcf file/folder path does not exist!')
    return 'Process is complete.'


@main.command()
@click.option('--vcfs', required=True)
@click.option('--output-dir', default='')
def get_cadd_scores(
    *,
    vcfs,
    output_dir
):
    # needs cadd environment
    if os.path.isfile(vcfs):
        cadd_scoring(vcfs)
    elif os.path.isdir(vcfs):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for file in os.listdir(vcfs):
            if file.endswith(('gvcf.gz', '.vcf', 'vcf.gz', 'gvcf')):
                cadd_scoring(os.path.join(vcfs, file), output_dir)
    else:
        return Exception('Vcf file/folder path does not exist!')
    return 'CADD scoring is complete.'


if __name__ == "__main__":
    main()

