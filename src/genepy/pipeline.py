# -*- coding: utf-8 -*-
import gzip
import os
import subprocess
from functools import partial

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import preprocess_df, score_db, score_genepy, poolcontext, gzip_reader, parallel_line_scoring


def run_parallel_genes_meta(header, meta_data, score_col, af_col, output_dir, excluded, weight_function, a, b, genes):
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
            click.echo("Error!" + gene + " not found!")
            p = subprocess.call(['rm', gene + '.meta'])
            with open(excluded, "a") as f:
                f.write(gene + "\n")
            continue
        gene_df[score_col].replace('.', np.nan)
        if gene_df[score_col].isnull().all():
            with open(excluded, "a") as f:
                f.write(gene + "\n")
            click.echo('Gene does not have deleteriousness score!')
            p = subprocess.call(['rm', gene + '.meta'])
            continue
        samples_df, scores, freqs = preprocess_df(gene_df, score_col, af_col)
        scores_matrix = score_db(
            samples=samples_df, score=scores, freq=freqs, weight_function=weight_function, a=a, b=b)
        path = os.path.join(output_dir, gene+'_'+score_col+'_matrix')
        np.savetxt(path, scores_matrix, fmt='%s', delimiter='\t')
        p = subprocess.call(['rm', gene+'.meta'])


def run_parallel_annovar(del_m, build, output_dir, vcf):
    process_annovar(vcf, del_m, build, output_dir)


def run_parallel_scoring(combined_df, genes, output_file, excluded, weight_function, a, b, score_col):
    scores_df = score_genepy(
        genepy_meta=combined_df,
        genes=genes,
        score_col=score_col,
        excluded=excluded,
        weight_function=weight_function,
        a=a,
        b=b,
    )
    scores_df.to_csv(score_col + output_file, sep='\t', index=False)


def annotated_vcf_prcoessing(*, scores_col, output_file, processes, vcf, weight_function, a, b):
    file_gen = gzip_reader(vcf)
    for row in file_gen:
        if row.startswith(b'##'):
            continue
        elif row.startswith(b'#'):
            header = row.decode("utf-8").strip('\n').split('\t')
            samples = header[header.index('FORMAT') + 1:]
            break
    df = pd.DataFrame(samples, columns=['sample_id'])
    with gzip.open(vcf, 'rb') as f:
        while True:
            lines = f.readlines(100000000)
            if not lines:
                break
            func = partial(parallel_line_scoring, scores_col, header, weight_function, a, b)
            with poolcontext(processes=processes) as pool:
                print('processing file chunk ...')
                p = pool.map(func, lines)
                for tup in tqdm(p, desc='Combining scores to df'):
                    if not tup:
                        continue
                    gene = tup[1]
                    scores_df = tup[0]
                    if gene in df.columns:
                        df[gene] = df[gene] + scores_df[gene]
                    else:
                        df = pd.merge(df, scores_df, on='sample_id')
    df.to_csv(output_file, sep='\t', index=False)
    return df


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

