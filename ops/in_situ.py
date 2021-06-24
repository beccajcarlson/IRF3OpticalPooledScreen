import numpy as np
import pandas as pd
from ops.constants import *
import ops.utils

#imaging_order = 'GTAC'


def extract_base_intensity(maxed, peaks, cells, threshold_peaks):

    # reads outside of cells get label 0
    read_mask = (peaks > threshold_peaks)
    print(sum(sum(read_mask)))
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T
    print(values.shape)

    return values, labels, positions

def extract_base_intensity_DO(log, cells, threshold_bases, channel_DO=1):

    # reads outside of cells get label 0
    # read_mask = (log[0, 1] > threshold_bases)
    # print(log.shape, read_mask.shape, log[0, 1].shape, sum(sum(read_mask)))
    # print(log[0,1][read_mask].shape)
    # values = log[0, 1][read_mask]#.transpose([2, 0, 1]) # start at 1 to skip DAPI

    read_mask = (log[0, channel_DO] > threshold_bases)
    print(log.shape, read_mask.shape)
    values = log[:, channel_DO:channel_DO+1, read_mask].transpose([2, 0, 1])
    print(values.shape)
    #
    labels = cells[read_mask]
    print(labels.shape, read_mask.shape, values.shape)
    positions = np.array(np.where(read_mask)).T
    print(positions.shape)
    return values, labels, positions


def format_bases(values, labels, positions, cycles, bases):    
    index = (CYCLE, cycles), (CHANNEL, bases)

    try:
        df = ops.utils.ndarray_to_dataframe(values, index)
    except ValueError:
        print('failed to reshape peaks to sequencing bases, writing empty table')
        return pd.DataFrame()

    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
    df = (df.stack([CYCLE, CHANNEL])
       .reset_index()
       .rename(columns={0: INTENSITY, 'level_0': READ})
       .join(pd.Series(labels, name=CELL), on=READ)
       .join(df_positions, on=READ)
       .sort_values([CELL, READ, CYCLE])
       )

    return df

def format_bases_DO(values, labels, positions, bases):    
    index = (CYCLE, ['c1']), (CHANNEL, bases)
    print('vals')
    print(values, values.shape)
    try:
        df = ops.utils.ndarray_to_dataframe(values, index)
    except ValueError:
        print('failed to reshape peaks to sequencing bases, writing empty table')
        return pd.DataFrame()
    print(df.head())
    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
    df = (df.stack([CHANNEL])
       .reset_index()
       .rename(columns={0: INTENSITY, 'level_0': READ})
       .join(pd.Series(labels, name=CELL), on=READ)
       .join(df_positions, on=READ)
       .sort_values([CELL, READ, 'c1'])
       )

    return df

def do_percentile_call(df_bases, cycles=12, channels=4, correction_only_in_cells=False, imaging_order='GTAC'):
    """Call reads from raw base signal using median correction. Use the 
    `correction_within_cells` flag to specify if correction is based on reads within 
    cells, or all reads.
    """
    print(imaging_order)
    print('nchannels ', channels)
    if correction_only_in_cells:
        # first obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query('cell > 0'), channels=channels)
        _, W = transform_percentiles(X_.reshape(-1, channels))

        # then apply to all data
        X = dataframe_to_values(df_bases, channels=channels)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases, channels=channels)
        Y, W = transform_percentiles(X.reshape(-1, channels))
        print(Y,Y.shape,X.shape,X)
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels, imaging_order=imaging_order)

    return df_reads

def do_median_call(df_bases, cycles=12, channels=4, correction_only_in_cells=False, imaging_order='GTAC'):
    """Call reads from raw base signal using median correction. Use the 
    `correction_within_cells` flag to specify if correction is based on reads within 
    cells, or all reads.
    """
    print('nchannels ', channels)
    if correction_only_in_cells:
        # first obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query('cell > 0'), channels=channels)
        _, W = transform_medians(X_.reshape(-1, channels))

        # then apply to all data
        X = dataframe_to_values(df_bases, channels=channels)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
        print('Y',Y.shape)
    else:
        X = dataframe_to_values(df_bases, channels=channels)
        Y, W = transform_medians(X.reshape(-1, channels))
        print(Y,Y.shape,X.shape,X)
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads

def do_median_call_bycycle(df_bases, cycles=12, channels=4, correction_only_in_cells=True, imaging_order='GTAC'):
    """Call reads from raw base signal using median correction. Use the 
    `correction_within_cells` flag to specify if correction is based on reads within 
    cells, or all reads.
    """
    print('nchannels ', channels)


    print(set(df_bases.cycle))

    Y_cycle = []
    if correction_only_in_cells:
        # first obtain transformation matrix W
        for cycle in range(1,cycles+1):
            print('cycle: ', cycle)
            df_bases_cycle = df_bases[df_bases.cycle == cycle]
            X_ = dataframe_to_values(df_bases_cycle.query('cell > 0'), channels=channels)
            #print('x_')
            #print(X_.shape)
            _, W = transform_medians(X_.reshape(-1, channels))
            #print('w')
            #print(W)
            #print(W.shape)

            # then apply to all data
            X = dataframe_to_values(df_bases_cycle, channels=channels)
            Y_cycle.append(W.dot(X.reshape(-1, channels).T).T.astype(int))
            print((W.dot(X.reshape(-1, channels).T).T.astype(int))[0:2])

    # else:
    #     X = dataframe_to_values(df_bases, channels=channels)
    #     Y, W = transform_medians(X.reshape(-1, channels))
    #     print(Y,Y.shape,X.shape,X)

    shape = (len(Y_cycle)*Y_cycle[0].shape[0], Y_cycle[0].shape[1])
    Y = np.hstack(Y_cycle).reshape(shape)
    print(Y[0:2])
    print(Y_cycle[0].shape, Y.shape, df_bases.shape, df_bases.head(), df_bases.tail())
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)
    print('df reads')
    print(df_reads.shape)
    return df_reads

def do_median_call_nomedian(df_bases, cycles=12, channels=4, correction_only_in_cells=False):
    """Call reads from raw base signal using median correction. Use the 
    `correction_within_cells` flag to specify if correction is based on reads within 
    cells, or all reads.
    """
    X = dataframe_to_values(df_bases, channels=channels)
    df_reads = call_barcodes(df_bases, X, cycles=cycles, channels=channels)
    print(df_reads, X)
    return df_reads


def clean_up_bases(df_bases):
    """Sort. Pre-processing for `dataframe_to_values`.
    """
    return df_bases.sort_values(['well', 'site', 'cell', 'read', 'cycle', 'channel'])


def call_cells(df_reads):
    """Determine count of top barcodes 
    """
    cols = ['well', 'site', 'cell']
    s = (df_reads
       .drop_duplicates(['well', 'site', 'read'])
       .groupby(cols)['barcode']
       .value_counts()
       .rename('count')
       .sort_values(ascending=False)
       .reset_index()
       .groupby(cols)
        )

    return (df_reads
      .join(s.nth(0)['barcode'].rename(BARCODE_0),                 on=cols)
      .join(s.nth(0)['count'].rename(BARCODE_COUNT_0).fillna(0), on=cols)
      .join(s.nth(1)['barcode'].rename(BARCODE_1),                 on=cols)
      .join(s.nth(1)['count'].rename(BARCODE_COUNT_1).fillna(0), on=cols)
      .join(s['count'].sum() .rename(BARCODE_COUNT),             on=cols)
      .drop_duplicates(cols)
      .drop(['barcode'], axis=1) # drop the read barcode
    )


def call_cells_highmoi(df_reads):
    """Determine count of top barcodes 
    """
    cols = ['well', 'site', 'cell']
    s = (df_reads
       .drop_duplicates(['well', 'site', 'read'])
       .groupby(cols)['barcode']
       .value_counts()
       .rename('count')
       .sort_values(ascending=False)
       .reset_index()
       .groupby(cols)
        )

    return (df_reads
      .join(s.nth(0)['barcode'].rename(BARCODE_0),                 on=cols)
      .join(s.nth(0)['count'].rename(BARCODE_COUNT_0).fillna(0), on=cols)
      .join(s.nth(1)['barcode'].rename(BARCODE_1),                 on=cols)
      .join(s.nth(1)['count'].rename(BARCODE_COUNT_1).fillna(0), on=cols)
      .join(s.nth(2)['barcode'].rename(BARCODE_2),                 on=cols)
      .join(s.nth(2)['count'].rename(BARCODE_COUNT_2).fillna(0), on=cols)
      .join(s.nth(3)['barcode'].rename(BARCODE_3),                 on=cols)
      .join(s.nth(3)['count'].rename(BARCODE_COUNT_3).fillna(0), on=cols)
      .join(s.nth(4)['barcode'].rename(BARCODE_4),                 on=cols)
      .join(s.nth(4)['count'].rename(BARCODE_COUNT_4).fillna(0), on=cols)
      .join(s.nth(5)['barcode'].rename(BARCODE_5),                 on=cols)
      .join(s.nth(5)['count'].rename(BARCODE_COUNT_5).fillna(0), on=cols)
      .join(s.nth(6)['barcode'].rename(BARCODE_6),                 on=cols)
      .join(s.nth(6)['count'].rename(BARCODE_COUNT_6).fillna(0), on=cols)
      .join(s.nth(7)['barcode'].rename(BARCODE_7),                 on=cols)
      .join(s.nth(7)['count'].rename(BARCODE_COUNT_7).fillna(0), on=cols)
      .join(s.nth(8)['barcode'].rename(BARCODE_8),                 on=cols)
      .join(s.nth(8)['count'].rename(BARCODE_COUNT_8).fillna(0), on=cols)
      .join(s.nth(9)['barcode'].rename(BARCODE_9),                 on=cols)
      .join(s.nth(9)['count'].rename(BARCODE_COUNT_9).fillna(0), on=cols)
      .join(s['count'].sum() .rename(BARCODE_COUNT),             on=cols)
      .query('cell > 0') # remove reads not in a cell
      .drop_duplicates(cols)
      .drop(['barcode'], axis=1) # drop the read barcode
    )    


def dataframe_to_values(df, value='intensity', channels=4):
    """Dataframe must be sorted on [cycles, channels]. 
    Returns N x cycles x channels.
    """
    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df[value]).reshape(-1, n_cycles, channels)
    return x

def dataframe_to_values_channelonly(df, cycle, value='intensity', channels=4):
    """Dataframe must be sorted on [cycles, channels]. 
    Returns N x cycles x channels.
    """
    cycles = df['cycle'].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    x = np.array(df[value]).reshape(-1, n_cycles, channels)
    return x

def transform_percentiles(X):
    """For each dimension, find points where that dimension is max. Use median of those points to define new axes. 
    Describe with linear transformation W so that W * X = Y.
    """

    def get_percentiles(X):
        arr = []
        for i in range(X.shape[1]):
            print('percentiles')
            # print(X[:,i:i+1].shape)
            # print(X.shape)
            rowsums=np.sum(X,axis=1)[:,np.newaxis]
            X_rel = (X/rowsums)
            perc = np.nanpercentile(X_rel[:,i],98)
            print(perc)
            high = X[X_rel[:,i] >= perc]
            print()
            #print(a.shape)
            #print((X/a)[0:2,:])
            #print((X[:,i:i+1]/(X+1e-6))[0:2,:])
            arr += [np.median(high, axis = 0)]
            #print(arr)
        M = np.array(arr)
        print(M)
        return M

    M = get_percentiles(X).T
    print(X)
    print(M)
    print(M.shape)
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W


def transform_medians(X):
    """For each dimension, find points where that dimension is max. Use median of those points to define new axes. 
    Describe with linear transformation W so that W * X = Y.
    """

    def get_medians(X):
        arr = []
        for i in range(X.shape[1]):
            arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
            print(arr)
        M = np.array(arr)
        return M

    M = get_medians(X).T
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W


def call_barcodes(df_bases, Y, cycles=12, channels=4, imaging_order='GTAC'):
    bases = sorted(imaging_order[:channels])
    df_reads = df_bases.drop_duplicates(['well', 'site', 'read']).copy()
    print(df_reads.head())
    print(Y)
    df_reads[BARCODE] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)
    Q = quality(Y.reshape(-1, cycles, channels))
    # needed for performance later
    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:,i]
 
    return (df_reads
        .assign(Q_min=lambda x: x.filter(regex='Q_\d+').min(axis=1))
        .drop([CYCLE, CHANNEL, INTENSITY], axis=1)
        )


def call_bases_fast(values, bases):
    """4-color: bases='ACGT'
    """
    assert values.ndim == 3
    assert values.shape[2] == len(bases)
    calls = values.argmax(axis=2)
    calls = np.array(list(bases))[calls]
    return [''.join(x) for x in calls]


def quality(X):
    X = np.abs(np.sort(X, axis=-1).astype(float))
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])
    Q = (Q * 2).clip(0, 1)
    return Q


def reads_to_fastq(df, microscope='MN', dataset='DS', flowcell='FC'):

    wrap = lambda x: '{' + x + '}'
    join_fields = lambda xs: ':'.join(map(wrap, xs))

    a = '@{m}:{d}:{f}'.format(m=microscope, d=dataset, f=flowcell)
    b = join_fields([WELL, CELL, 'well_tile', READ, POSITION_I, POSITION_J])
    c = '\n{b}\n+\n{{phred}}'.format(b=wrap(BARCODE))
    fmt = a + b + c 
    
    well_tiles = sorted(set(df[WELL] + '_' + df[TILE]))
    fields = [WELL, TILE, CELL, READ, POSITION_I, POSITION_J, BARCODE]
    
    Q = df.filter(like='Q_').values
    
    reads = []
    for i, row in enumerate(df[fields].values):
        d = dict(zip(fields, row))
        d['phred'] = ''.join(phred(q) for q in Q[i])
        d['well_tile'] = well_tiles.index(d[WELL] + '_' + d[TILE])
        reads.append(fmt.format(**d))
    
    return reads
    

def dataframe_to_fastq(df, file, dataset):
    s = '\n'.join(reads_to_fastq(df, dataset))
    with open(file, 'w') as fh:
        fh.write(s)
        fh.write('\n')


def phred(q):
    """Convert 0...1 to 0...30
    No ":".
    No "@".
    No "+".
    """
    n = int(q * 30 + 33)
    if n == 43:
        n += 1
    if n == 58:
        n += 1
    return chr(n)


def add_clusters(df_cells, neighbor_dist=50):
    """Assigns -1 to clusters with only one cell.
    """
    from scipy.spatial.kdtree import KDTree
    import networkx as nx

    x = df_cells[GLOBAL_X] + df_cells[POSITION_J]
    y = df_cells[GLOBAL_Y] + df_cells[POSITION_I]
    barcodes = df_cells[BARCODE_0]
    barcodes = np.array(barcodes)

    kdt = KDTree(np.array([x, y]).T)
    num_cells = len(df_cells)
    print('searching for clusters among %d cells' % num_cells)
    pairs = kdt.query_pairs(neighbor_dist)
    pairs = np.array(list(pairs))

    x = barcodes[pairs]
    y = x[:, 0] == x[:, 1]

    G = nx.Graph()
    G.add_edges_from(pairs[y])

    clusters = list(nx.connected_components(G))

    cluster_index = np.zeros(num_cells, dtype=int) - 1
    for i, c in enumerate(clusters):
        cluster_index[list(c)] = i

    df_cells[CLUSTER] = cluster_index
    return df_cells


def index_singleton_clusters(clusters):
    clusters = clusters.copy()
    filt = clusters == -1
    n = clusters.max()
    clusters[filt] = range(n, n + len(filt))
    return clusters