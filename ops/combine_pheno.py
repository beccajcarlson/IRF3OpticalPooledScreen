# imports
from joblib import Parallel, delayed
#from dask import compute, delayed
from dask.distributed import Client

import sys
import math

import mahotas
from ops.imports_ipython import *

import skimage

from nd2reader import ND2Reader
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from sklearn import impute
from sklearn.preprocessing import StandardScaler


def fix_df_channel(df):

    # drop cols with 0 SD - should be very few
    firstcol_loc = df.columns.get_loc("channel_corrch1_nuclear_corr")
    to_drop_sd = list(df.columns[firstcol_loc:][(df.iloc[:,firstcol_loc:].std(axis = 0) == 0)])

    print('sd done, dropping %s columns'%len(to_drop_sd))
    # drop cols with less than 10% cv
    thresh = 10
    to_drop_cv = list(df.columns[firstcol_loc:][np.abs(df.iloc[:,firstcol_loc:].std(axis = 0)
                                            /np.abs(df.iloc[:,firstcol_loc:].mean(axis = 0)))*100<thresh])

    print('cv done, dropping %s columns'%len(to_drop_cv))
    # drop cols with more than .1% na
    to_drop_na = list(df.iloc[:,firstcol_loc:].columns[(df.iloc[:,firstcol_loc:].isna().sum(axis = 0)/
                                                        df.iloc[:,firstcol_loc:].shape[0] > .001)])
    
    print('na done, dropping %s columns'%len(to_drop_na))
    to_drop = to_drop_sd+to_drop_cv+to_drop_na
    print('drop percent: ', len(to_drop)/len(df.columns)*100)

    df.drop(to_drop, axis=1, inplace=True)
    


    imp=impute.SimpleImputer(missing_values=np.nan, strategy='median')
    df.iloc[:,firstcol_loc:]=imp.fit_transform(df.iloc[:,firstcol_loc:])
    df.iloc[:,firstcol_loc:]=StandardScaler().fit_transform(df.iloc[:,firstcol_loc:])
    
    return df

key = pd.read_hdf('/home/rcarlson/datadisk/df_pheno_tokeep.hdf')

from pandas.io.common import EmptyDataError

# load all sbs cell coordinates
def read_csv_par(f):
    try:
        df = pd.read_csv(f)       
        ints = df.columns[df.dtypes == 'int64']
        df[ints] = df[ints].astype('int32')
        floats = df.columns[df.dtypes == 'float64']
        df[floats] = df[floats].astype('float32')
      
    except EmptyDataError:
        df = pd.DataFrame()
    return df


def expand_zernike_pftas(df):
        
        try: 
            df.channel_zernike_nuclear = df.channel_zernike_nuclear.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])
            df.channel_zernike_cell = df.channel_zernike_cell.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])
            df.channel_zernike_cyto = df.channel_zernike_cyto.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])

            df.channel_pftas_nuclear = df.channel_pftas_nuclear.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])
            df.channel_pftas_cell = df.channel_pftas_cell.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])
            df.channel_pftas_cyto = df.channel_pftas_cyto.apply(lambda x: [s.strip() for s in list(filter(None,(x[1:-1].split(" "))))])


            df[['channel_zernike_nuc' + str(n) for n in range(1,31)]] = pd.DataFrame(df.channel_zernike_nuclear.values.tolist(), index = df.index) 
            #df.drop(['channel_zernike_nuclear'], axis = 1, inplace = True)

            df[['channel_zernike_cell' + str(n) for n in range(1,31)]] = pd.DataFrame(df.channel_zernike_cell.values.tolist(), index = df.index)
            #df.drop(['channel_zernike_cell'], axis = 1, inplace = True)

            df[['channel_zernike_cyto' + str(n) for n in range(1,31)]] = pd.DataFrame(df.channel_zernike_cyto.values.tolist(), index = df.index)
            #df.drop(['channel_zernike_cyto'], axis = 1, inplace = True)


            df[['channel_pftas_nuc' + str(n) for n in range(1,55)]] = pd.DataFrame(df.channel_pftas_nuclear.values.tolist(), index = df.index)
            #df.drop(['channel_pftas_nuclear'], axis = 1, inplace = True)

            df[['channel_pftas_cell' + str(n) for n in range(1,55)]] = pd.DataFrame(df.channel_pftas_cell.values.tolist(), 
                                                                   index = df.index)
            #df.drop(['channel_pftas_cell'], axis = 1, inplace = True)

            df[['channel_pftas_cyto' + str(n) for n in range(1,55)]] = pd.DataFrame(df.channel_pftas_cyto.values.tolist(), 
                                                                   index = df.index)
            #df.drop(['channel_pftas_cyto'], axis = 1, inplace = True)
            df.drop(['channel_zernike_nuclear', 'channel_zernike_cell', 'channel_zernike_cyto', 
                             'channel_pftas_nuclear', 'channel_pftas_cell', 'channel_pftas_cyto'], axis = 1, inplace = True)

            # bring non-feature columns to front
            nonfeat_cols = ['plate','well','site','i','j','cell']
            cols = [col for col in df if col not in nonfeat_cols]
            for s in range(len(nonfeat_cols)):
                cols.insert(s,nonfeat_cols[s])
            df = df[cols]
            to_drop = ['label']
            df.drop(to_drop,axis=1,inplace = True)

            firstcol_loc = df.columns.get_loc("channel_corrch1_nuclear_corr")
            objs = df.columns[firstcol_loc:][df.dtypes[firstcol_loc:] == 'object']
            df[objs] = df[objs].astype('float32')
        
        except EmptyDataError:
            df = pd.DataFrame()
        return df

# load all sbs cell coordinates

def read_csv_extended_morph(f):
    try:
        df_morph = pd.read_csv(f)
        
        df_morph[['hu_moments_nuclear' + str(n) for n in range(1,8)]] = pd.DataFrame(df_morph.hu_moments_nuclear.values.tolist(),
                                                                                             index = df_morph.index)
        df_morph[['hu_moments_cell' + str(n) for n in range(1,8)]] = pd.DataFrame(df_morph.hu_moments_cell.values.tolist(),
                                                                                     index = df_morph.index)

        df_morph.drop(['hu_moments_cell', 'hu_moments_nuclear'], axis = 1, inplace = True)

    except EmptyDataError:
        df_morph = pd.DataFrame()
    return df_morph

def parallel_process(func, args_list, n_jobs, tqdn=True):
    from joblib import Parallel, delayed
    work = args_list
    if tqdn:
        from tqdm import tqdm_notebook 
        work = tqdm_notebook(work, 'work')
    return Parallel(n_jobs=n_jobs)(delayed(func)(w) for w in work)

files = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.dapi.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/dapi.hdf','x',mode='w') ## ~20G

iles = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.pex.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/pex.hdf','x',mode='w') ## ~20G

iles = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.mito.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/mito.hdf','x',mode='w') ## ~20G

iles = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.mda5.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/mda5.hdf','x',mode='w') ## ~20G

iles = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.rig.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/rig.hdf','x',mode='w') ## ~20G

iles = glob('/home/rcarlson/datadisk/process/tiled/M4*/pheno/*.sev.csv')
files = parallel_process(func = read_csv_par, args_list = files, n_jobs=700) 

files = parallel_process(expand_zernike_pftas, files, n_jobs=600) # took ~50min

df = pd.concat(files)
print('concat done, shape: ',df.shape)
to_keep = (pd.Series(list(zip(df.plate, df.well, df.site, df.cell)))
        .isin(list(zip(key.plate, key.well, key.tile, key.cell))))

df = df.reset_index()[to_keep]
df.drop('index',axis=1,inplace = True)
print('drop done, new shape: ',df.shape)
df = fix_df_channel(df)
print('fix done')
df.to_hdf('/home/rcarlson/datadisk/mergehdfs/sev.hdf','x',mode='w') ## ~20G