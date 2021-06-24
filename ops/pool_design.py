from collections import defaultdict, Counter
import scipy.sparse
import numpy as np
import pandas as pd
import os
from Levenshtein import distance

import ops.utils
from ops.constants import *


# LOAD TABLES

def validate_design(df_design):
    for group, df in df_design.groupby('group'):
        x = df.drop_duplicates(['prefix_length', 'edit_distance'])
        if len(x) > 1:
            cols = ['group', DESIGN, 'prefix_length', 'edit_distance']
            error = 'multiple prefix specifications for group {0}:\n{1}'
            raise ValueError(error.format(group, df[cols]))
            
    return df_design


def load_gene_list(filename):
    return (pd.read_csv(filename, header=None)
     .assign(design=os.path.splitext(filename)[0])
     .rename(columns={0: 'gene_id'})
    )


def validate_genes(df_genes, df_sgRNAs):
    missing = set(df_genes['gene_id']) - set(df_sgRNAs['gene_id'])
    if missing:
        error = '{0} gene ids missing from sgRNA table: {1}'
        missing_ids = ', '.join(map(str, missing))
        raise ValueError(error.format(len(missing), missing_ids))

    duplicates = df_genes[[SUBPOOL, 'gene_id']].duplicated(keep=False)
    if duplicates.any():
        error = 'duplicate genes for the same subpool: {0}'
        xs = df_genes.loc[duplicates, [SUBPOOL, 'gene_id']].values
        raise ValueError(error.format(xs))

    return df_genes


def select_prefix_group(df_genes, df_sgRNAs):
    # doesn't shortcut if some genes need less guides
    prefix_length, edit_distance = (
        df_genes[['prefix_length', 'edit_distance']].values[0])

    return (df_sgRNAs
        .join(df_genes.set_index('gene_id'), on='gene_id', how='inner')
        .pipe(select_guides, prefix_length, edit_distance)
        .sort_values([SUBPOOL, 'gene_id', 'rank'])
    #     # .set_index('gene_id')
    #     # .pipe(df_genes.join, on='gene_id', how='outer', rsuffix='_remove')
    #     # .pipe(lambda x: x[[c for c in x.columns if not c.endswith('_remove')]])
        .assign(selected_rank = lambda x: 
            ops.utils.rank_by_order(x, [SUBPOOL, 'gene_id']))
        .query('selected_rank <= sgRNAs_per_gene')
        .sort_values([SUBPOOL, 'gene_id', 'selected_rank'])
        .drop(['selected_rank'], axis=1)
    )


def select_guides(df_input, prefix_length, edit_distance):
    """`df_input` has 'gene_id', sgRNAs_per_gene
    """
    # TODO: NEEDS EDIT DISTANCE
    if edit_distance != 1:
        raise NotImplementedError('edit distance needs doing')

    selected_guides = (df_input
     .assign(prefix=lambda x: x['sgRNA'].str[:prefix_length])
     .pipe(lambda x: x.join(x['gene_id'].value_counts().rename('sgRNAs_per_id'), 
         on='gene_id'))
     .sort_values(['rank', 'sgRNAs_per_id'])
     .drop_duplicates('prefix')
     ['sgRNA'].pipe(list)
     )
    return df_input.query('sgRNA == @selected_guides')


# FILTER SGRNAS

def filter_sgRNAs(df_sgRNAs, homopolymer=5):
    cut = [has_homopolymer(x, homopolymer) or has_BsmBI_site(x) 
            for x in df_sgRNAs[SGRNA]]
    return df_sgRNAs[~np.array(cut)]

def has_homopolymer(x, n):
    a = 'A'*n in x
    t = 'T'*n in x
    g = 'G'*n in x
    c = 'C'*n in x
    return a | t | g | c
   
def has_BsmBI_site(x):
    x = 'CACCG' + x.upper() + 'GTTT'
    return 'CGTCTC' in x or 'GAGACG' in x


# OLIGOS

def build_sgRNA_oligos(df, dialout_primers, 
                        left='CGTCTCgCACCg', right='GTTTcGAGACG'):

    template = '{fwd}{left}{sgRNA}{right}{rev}'
    arr = []
    for s, d in df[[SGRNA, 'dialout']].values:
        fwd, rev = dialout_primers[d]
        rev = reverse_complement(rev)
        oligo = template.format(fwd=fwd, rev=rev, sgRNA=s, 
                                left=left, right=right)
        arr += [oligo]
    return arr


def build_test(df_oligos, dialout_primers):
    """Pattern-match sgRNA cloning and dialout primers.
    """
    sites = 'CGTCTC', reverse_complement('CGTCTC')
    pat = ('(?P<dialout_fwd>.*){fwd}.CACCG'
           '(?P<sgRNA_cloned>.*)'
           'GTTT.{rev}(?P<dialout_rev>.*)')
    pat = pat.format(fwd=sites[0], rev=sites[1])

    kosuri = {}
    for i, (fwd, rev) in enumerate(dialout_primers):
        kosuri[fwd] = 'fwd_{0}'.format(i)
        kosuri[rev] = 'rev_{0}'.format(i)

    def validate_design(df):
        if not (df[VECTOR] == 'CROPseq').all():
            raise ValueError('can only validate CROPseq design')
        return df

    return (df_oligos
     .pipe(validate_design)
     .assign(sgRNA=lambda x: x['sgRNA'].str.upper())
     .assign(oligo=lambda x: x['oligo'].str.upper())
     .pipe(lambda x: pd.concat([x, x['oligo'].str.extract(pat)], axis=1))
     .assign(dialout_rev=lambda x: x['dialout_rev'].apply(reverse_complement))
     .assign(dialout_fwd_ix=lambda x: x['dialout_fwd'].apply(kosuri.get))      
     .assign(dialout_rev_ix=lambda x: x['dialout_rev'].apply(kosuri.get))            
     .assign(dialout_ix=lambda x: 
             x['dialout_fwd_ix'].str.split('_').str[1].astype(int))
    )


def validate_test(df_test):
    """Check sgRNA cloning and identiy of dialout primers.
    """
    assert df_test.eval('sgRNA_cloned == sgRNA').all()

    assert (df_test['dialout_fwd_ix'].str[-1] == 
            df_test['dialout_rev_ix'].str[-1]).all()

    assert df_test.eval('dialout_ix== dialout').all()

    print('Looking good!')

    return df_test


def reverse_complement(seq):
    watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'U': 'A',
                'N': 'N'}

    watson_crick.update({k.lower(): v.lower() 
        for k, v in watson_crick.items()})

    return ''.join(watson_crick[x] for x in seq)[::-1]


# EXTERNAL

def import_brunello(filename):
    """Import "Brunello Library Target Genes", which can be found at:
    https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/
    """
    columns = {'Target Gene ID': 'gene_id'
              ,'Target Gene Symbol': GENE_SYMBOL
              ,'sgRNA Target Sequence': SGRNA
              , 'Rule Set 2 score': 'sgRNA_score'
              , 'Combined rank': 'sgRNA_score'
              }


    def reassign_nontargeting(df):
        """Given non-targeting sgRNAs a gene ID of -1.
        """
        new_ids = []
        new_symbols = []
        for i, s in df[['gene_id', GENE_SYMBOL]].values:
            if s == 'Non-Targeting Control':
                new_ids.append(-1)
                new_symbols.append('nontargeting')
            else:
                new_ids.append(i)
                new_symbols.append(s)

        return df.assign(**{'gene_id': new_ids, GENE_SYMBOL: new_symbols})


    return (pd.read_csv(filename, sep='\t')
        .rename(columns=columns)
        .pipe(reassign_nontargeting)
        .pipe(ops.utils.cast_cols, int_cols=['gene_id'])
        .assign(**{'sgRNA_score': lambda x: x['sgRNA_score'].fillna(0)})
        .assign(**{'rank': lambda x: 
            x.groupby('gene_id')['sgRNA_score']
             .rank(ascending=False, method='first').astype(int)})
        [['gene_id', GENE_SYMBOL, 'rank', SGRNA]]
        .sort_values(['gene_id', 'rank'])
        )


def import_tkov3(filename, df_ncbi):  
    print('tko')  
    columns = {'GENE': GENE_SYMBOL, 'SEQUENCE': SGRNA}
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)['gene_id']
    return (pd.read_table(filename)
     .rename(columns=columns)
     [[GENE_SYMBOL, SGRNA]]
     .join(symbols_to_ids, on=GENE_SYMBOL, how='inner')
     .assign(**{'rank': lambda x: ops.utils.rank_by_order(x, 'gene_id')})
    )

def import_brunello_v2(filename):
    """Import "Brunello Library Target Genes", which can be found at:
    https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/
    """
    columns = {'Target Gene ID': 'gene_id'
              ,'Target Gene Symbol': 'gene_symbol'
              ,'sgRNA Sequence': 'sgRNA'
              , 'Combined Rank': 'rank'
              }


    def reassign_nontargeting(df):
        """Given non-targeting sgRNAs a gene ID of -1.
        """
        new_ids = []
        new_symbols = []
        for i, s in df[['gene_id', 'gene_symbol']].values:
            if s == 'Non-Targeting Control':
                new_ids.append(-1)
                new_symbols.append('nontargeting')
            else:
                new_ids.append(i)
                new_symbols.append(s)

        return df.assign(**{'gene_id': new_ids, 'gene_symbol': new_symbols})

    
    return (pd.read_csv(filename, sep='\t')
        .rename(columns=columns)
        .pipe(reassign_nontargeting)
        .pipe(ops.utils.cast_cols, int_cols=['gene_id'])
        [['gene_id', 'gene_symbol', 'rank', 'sgRNA']]
        .sort_values(['gene_id', 'rank'])
        )


def import_tkov3(filename, df_ncbi):    
    columns = {'GENE': GENE_SYMBOL, 'SEQUENCE': SGRNA}
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)['gene_id']
    return (pd.read_table(filename)
     .rename(columns=columns)
     [[GENE_SYMBOL, SGRNA]]
     .join(symbols_to_ids, on=GENE_SYMBOL, how='inner')
     .assign(**{'rank': lambda x: ops.utils.rank_by_order(x, 'gene_id')})
    )


def import_hugo_ncbi(filename):
    columns = {'Approved symbol': GENE_SYMBOL,
               'NCBI Gene ID(supplied by NCBI)': 'gene_id'}
    return (pd.read_csv(filename, comment='#', sep='\t')
         .rename(columns=columns)
         .dropna()
         .pipe(ops.utils.cast_cols, int_cols=['gene_id']))


def import_dialout_primers(filename):
    """Returns an array of (forward, reverse) primer pairs.
    """
    return pd.read_csv('kosuri_dialout_primers.csv').values