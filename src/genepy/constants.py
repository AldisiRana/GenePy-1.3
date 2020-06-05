# -*- coding: utf-8 -*-

"""Constants for genepy"""


SCORES_TO_COL_NAMES = {
    'cadd': ['CADD_Raw'],
    'ljb26_all': ['SIFT_score', 'Polyphen2_HDIV_score', 'Polyphen2_HVAR_score', 'LRT_score', 'MutationTaster_score',
                  'MutationAssessor_score', 'FATHMM_score', 'RadialSVM_score', 'LR_score', 'VEST3_score', 'CADD_raw'],
    'gwava': ['GWAVA_region_score', 'GWAVA_tss_score'],
    'revel': ['REVEL'],
    'eigen': ['Eigen'],
    'dann': ['dann'],
    'cadd13': ['CADD13_RawScore'],
}
