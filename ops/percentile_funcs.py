

@staticmethod
    def _call_reads_percentiles(df_bases, peaks=None, correction_only_in_cells=True, imaging_order='GTAC'):
        print(imaging_order)
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return


        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_percentile_call, cycles=cycles, channels=channels, 
                    imaging_order=imaging_order, correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads


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

    def transform_percentiles(X):
    """For each dimension, find points where that dimension is >=98th percentile intensity. Use median of those points to define new axes. 
    Describe with linear transformation W so that W * X = Y.
    """

    def get_percentiles(X):
        arr = []
        for i in range(X.shape[1]):
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
