import inspect
import functools
import os
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')

import numpy as np
import pandas as pd
import skimage
import ops.features
import ops.process
import ops.io
import ops.in_situ
from ops.process import Align

from ops.stitching import *
from scipy.ndimage import morphology, distance_transform_edt
import mahotas 
from astropy.stats import median_absolute_deviation
from scipy import ndimage

class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """
    @staticmethod
    def _aligned_masks_totable(data, wildcards, cellchannel=1, gridchannel=2):
 

        def masked(region):
            return region.intensity_image_full[region.filled_image]     
        import scipy
        features_cell = {
            'sbs_cell' : lambda r: int((scipy.stats.mode(masked(r), axis=None))[0]),
            'cell'               : lambda r: r.label
        }

        features_site = {
            'sbs_site' : lambda r: int((scipy.stats.mode(masked(r), axis=None))[0]),
            'cell'               : lambda r: r.label
        }

        print(data.shape)
        print(data[0,0].shape, data[channel,0].shape)

        dfc =  Snake._extract_features(data=data[cellchannel,0], labels=data[0,0], wildcards=wildcards, features=features_cell)
        dfs =  Snake._extract_features(data=data[gridchannel,0], labels=data[0,0], wildcards=wildcards, features=features_site)

        df = (pd.concat([dfc.set_index('cell'), dfs.set_index('cell')], axis=1, join='inner')
                .reset_index())
        df = df.loc[:, ~df.columns.duplicated()]

        return df



    @staticmethod
    def _extract_phenotype_extended_morphology(data_phenotype, nuclei, cells, cytoplasm, wildcards):
 

        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

     

        features_nuclear = {
            'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
            'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
            'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
            'orientation_nuclear' : lambda r: r.orientation,
            'hu_moments_nuclear': lambda r: r.moments_hu,
            'solidity_nuclear': lambda r: r.solidity,
            'extent_nuclear': lambda r: r.extent,
            'cell'               : lambda r: r.label
        }

        features_cell = {
            'euler_cell' : lambda r: r.euler_number,
            'eccentricity_cell' : lambda r: r.eccentricity, #cell
            'major_axis_cell' : lambda r: r.major_axis_length, #cell
            'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
            'orientation_cell' : lambda r: r.orientation,
            'hu_moments_cell': lambda r: r.moments_hu,
            'solidity_cell': lambda r: r.solidity,
            'extent_cell': lambda r: r.extent,
            'cell'               : lambda r: r.label
        }

        features_cytoplasm = {
            'euler_cyto' : lambda r: r.euler_number,
            'area_cyto'       : lambda r: r.area,
            'perimeter_cyto' : lambda r: r.perimeter,
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

        ## calculate radius distances, mass displacement
        dist = []
        for i in np.unique(cells):
            if i == 0:
                continue
            mask = cells == i
            tmp = data_phenotype[1].copy()
            tmp[mask == False] = 0
            dists = distance_transform_edt(tmp, sampling=None, return_distances=True)
                   
            max_rad = np.max(dists)
            mean_rad = np.mean(dists)
            median_rad = np.median(dists)
            rad_std = np.std(dists)

            comgray = ndimage.measurements.center_of_mass(tmp)
            binary = tmp.copy()
            binary[binary!=0] = 1
            combin = ndimage.measurements.center_of_mass(binary) #different from skimage centroid (i,j) in resulting .pkl
            euclid_dist_cell = ((comgray[0] - combin[0]) ** 2 +
                        (comgray[1] - combin[1]) ** 2) ** 0.5

            tmp_msd = []
            nonzero = np.nonzero(tmp) #inds of nonzero objects
            for x in range(len(nonzero[0])):
                tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
            msd_cell = np.mean(tmp_msd)

            mask = nuclei == i
            tmp = data_phenotype[1].copy()
            tmp[mask == False] = 0
            comgray = ndimage.measurements.center_of_mass(tmp)
            binary = tmp.copy()
            binary[binary!=0] = 1
            combin = ndimage.measurements.center_of_mass(binary) #technically no need to recalc, this is just skimage centroid
            euclid_dist_nuc = ((comgray[0] - combin[0]) ** 2 +
                        (comgray[1] - combin[1]) ** 2) ** 0.5

            tmp_msd = []
            nonzero = np.nonzero(tmp) #inds of nonzero objects
            for x in range(len(nonzero[0])):
                tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
            msd_nuc = np.mean(tmp_msd)

            dist.append((i, max_rad, mean_rad, rad_std, 
                euclid_dist_cell, euclid_dist_nuc, msd_cell, msd_nuc))

        df_radii = pd.DataFrame(dist, columns = ['cell', 'max_cell_radius', 'mean_cell_radius', 'radius_cell_SD',
                                                'cell_mass_displacement', 'nuc_mass_displacement',
                                            'mean_sq_dist_cell', 'mean_sq_dist_nuc'])
    
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell'), df_radii.set_index('cell')], axis=1, join='inner')
                .reset_index())

       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    @staticmethod
    def _extract_phenotype_extended_channel(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,corrchannel1,
         corrchannel2, corrchannel3, corrchannel4, corrchannel5, corrchannel6):
 

        def correlate_channel_corrchannel(region, channel, corrchannel):
            if corrchannel == None:
                return 0
            corrch = region.intensity_image_full[corrchannel]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    
    
        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

        def mahotas_zernike(region, channel):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            return mfeat

        def mahotas_pftas(region, channel):
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat



        features_nuclear = {
            'channel_corrch1_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel1),
            'channel_corrch2_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel2),
            'channel_corrch3_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel3),
            'channel_corrch4_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel4),   
            'channel_corrch5_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel5),
            'channel_corrch6_nuclear_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel6),           
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
            'channel_zernike_nuclear': lambda r: mahotas_zernike(r, channel),
            'channel_pftas_nuclear': lambda r: mahotas_pftas(r, channel),
            'cell'               : lambda r: r.label

        }

        features_cell = {
            'channel_corrch1_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel1),
            'channel_corrch2_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel2),
            'channel_corrch3_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel3),
            'channel_corrch4_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel4),   
            'channel_corrch5_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel5),
            'channel_corrch6_cell_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel6),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_min': lambda r: np.min(masked(r,channel)),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'channel_cell_sd': lambda r: np.std(masked(r,channel)),
            'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
            'channel_zernike_cell': lambda r: mahotas_zernike(r, channel),
            'channel_pftas_cell': lambda r: mahotas_pftas(r, channel),
            'cell'               : lambda r: r.label
        }

        features_cytoplasm = {
            'channel_corrch1_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel1),
            'channel_corrch2_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel2),
            'channel_corrch3_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel3),
            'channel_corrch4_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel4),   
            'channel_corrch5_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel5),
            'channel_corrch6_cyto_corr' : lambda r: correlate_channel_corrchannel(r, channel, corrchannel6),
            'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
            'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
            'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
            'channel_cyto_min'    : lambda r: masked(r, channel).min(),
            'channel_cyto_max'    : lambda r: masked(r, channel).max(),
            'channel_cyto_sd': lambda r: np.std(masked(r,channel)),
            'channel_cyto_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cyto_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cyto_75': lambda r: np.percentile(masked(r, channel),75),
            'channel_zernike_cyto': lambda r: mahotas_zernike(r, channel),
            'channel_pftas_cyto': lambda r: mahotas_pftas(r, channel),
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 
        
       
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
                .reset_index())

       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    @staticmethod
    def _segment_cells(data, nuclei, threshold, channel = None):
            """Segment cells from aligned data. Matches cell labels to nuclei labels.
            Note that labels can be skipped, for example if cells are touching the 
            image boundary.
            """
            if data.ndim == 4:
                # no DAPI, min over cycles, mean over channels
                if channel != None:
                    mask = data[:, channel].mean(axis = 0)
                else:
                    mask = data[:, 1:].min(axis=0).mean(axis=0)
            elif data.ndim == 3:
                if channel != None:
                    mask = data[channel]
                else:
                    mask = np.median(data[1:], axis=0)
            elif data.ndim == 2:
                mask = data
            else:
                raise ValueError

            mask = mask > threshold
            try:
                # skimage precision warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cells = ops.process.find_cells(nuclei, mask)
            except ValueError:
                print('segment_cells error -- no cells')
                cells = nuclei

            return cells

    @staticmethod
    def _align_by_channel_6ch(data_1, data_2, data_3, data_4, data_5, data_6, channel_index1=0, channel_index2=0, 
            channel_index3=0, channel_index4=0, channel_index5=0, channel_index6=0, upsample_factor=1):
            """Align a series of images to the first, using the channel at position 
            `channel_index`. The first channel is usually DAPI.
            """

            # add new axis to single-channel images
            if data_1.ndim == 2:
                data_1 = data_1[np.newaxis,:]
            if data_2.ndim == 2:
                data_2 = data_2[np.newaxis,:]
            if data_3.ndim == 2:
                data_3 = data_3[np.newaxis,:]
            if data_4.ndim == 2:
                data_4 = data_4[np.newaxis,:]
            if data_5.ndim == 2:
                data_5 = data_5[np.newaxis,:]
            if data_6.ndim == 2:
                data_6 = data_6[np.newaxis,:]

            print('aligning')
            images = data_1[channel_index1], data_2[channel_index2]
            _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
            offsets = [offset] * len(data_2)
            aligned2 = ops.process.Align.apply_offsets(data_2, offsets)

            images = data_1[channel_index1], data_3[channel_index3]
            _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
            offsets = [offset] * len(data_3)
            aligned3 = ops.process.Align.apply_offsets(data_3, offsets)

            images = data_1[channel_index1], data_4[channel_index4]
            _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
            offsets = [offset] * len(data_4)
            aligned4 = ops.process.Align.apply_offsets(data_4, offsets)

            images = data_1[channel_index1], data_5[channel_index5]
            _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
            offsets = [offset] * len(data_5)
            aligned5 = ops.process.Align.apply_offsets(data_5, offsets)

            images = data_1[channel_index1], data_6[channel_index6]
            _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
            offsets = [offset] * len(data_6)
            aligned6 = ops.process.Align.apply_offsets(data_6, offsets)
                    #print(data_1.shape)
            #print(aligned.shape)
            aligned = np.vstack((data_1, aligned2, aligned3, aligned4, aligned5, aligned6))
            return aligned

    def fix_channel_offsets(data, channel_offsets):
            d = data.transpose([1, 0, 2, 3])
            x = [lasagna.utils.offset(a, b) for a,b in zip(d, channel_offsets)]
            x = np.array(x).transpose([1, 0, 2, 3])
            return x

    @staticmethod
    def _find_cytoplasm(nuclei, cells):

            mask = (nuclei == 0) & (cells > 0)
            
            cytoplasm = cells.copy()
            cytoplasm[mask == False] = 0

            return cytoplasm



    @staticmethod
    def _crop_files(data, nsites = 1):
        import math
        data = np.array(data)
        if data.ndim == 4:
            data = data[0,...]
        print(data.shape)
        stitched_size = data.shape[-1]
        cropped = []
        for i in range(int(math.sqrt(nsites))):
            for j in range(int(math.sqrt(nsites))):
                cropped.append(data[:,int(i*stitched_size/math.sqrt(nsites)):int(stitched_size/math.sqrt(nsites)*(i+1)),
                                 int(j*stitched_size/math.sqrt(nsites)):int(stitched_size/math.sqrt(nsites)*(j+1))])

        
        print(len(cropped))
        print(type(cropped))
        print(type(cropped[0]))
        print(cropped[0].shape)
        cropped = np.array(cropped)#.reshape(nsites,-1,int(stitched_size/math.sqrt(nsites)))
        cropped = np.repeat(cropped[np.newaxis,...], nsites, axis = 0) 
        print(cropped.shape)
        return cropped

    @staticmethod
    def _nd2_to_tif(data):
     ## If z present, max projects along z, keeps multichannel together
        from nd2reader import ND2Reader
        imagelist = []
        with ND2Reader(data) as images:
            images.bundle_axes = ''
          
            if 'z' in images.sizes.keys():
                print('z-stack')
                images.bundle_axes += 'z'
            if 'c' in images.sizes.keys():
                print('multichannel')
                images.bundle_axes += 'c'

            images.bundle_axes += 'xy'
            print(images.bundle_axes)
            images.iter_axes = 'v'
            for fov in images:
                if 'z' in images.sizes.keys():
                    fov = fov.max(axis = 0) # max project z
                imagelist.append(np.array(fov))
            print('imlist len ', len(imagelist))
            print(imagelist[0].shape)
        return imagelist


    @staticmethod
    def _stitch_well2(data, seq_lengths=None, predefined_seq_lengths=None, order_flags='Meander', from_nd2 = True, overlap_frac=0.15,
        channel_index = None, multicycle=False):

        ## generates image of well from tiles using multi-band blending of adjacent tiles based on predefined grid 
        ## seq_lengths = list of #s of fov per row of well
        ## predefined_seq_lengths = a string corresponding to a commonly used grid
        ## order_flags = list of flags as long as seq_lengths. 1 indicates that image row order is reversed
        ## overlap_frac = fraction of adjacent images overlapping
        ## from nd2 - if false, expects list of tifs, if true, fovs in nd2 format 

        import time
        from itertools import accumulate


        if (seq_lengths == None) & (predefined_seq_lengths == None):
            error = 'Must choose a value for seq_lengths or predefined_seq_lengths'
            raise ValueError(error)
        
        if predefined_seq_lengths == 'seq_lengths_10x_6w':
            seq_lengths = [5,9,13,15,17,17,19,19,21,21,21,21,21,19,19,17,17,15,13,9,5]

        if predefined_seq_lengths == 'seq_lengths_20x_6w':
            seq_lengths = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]

        if order_flags == 'Meander':
            order_flags = [1 if (i % 2 != 0) else 0 for i in range(len(seq_lengths))]

        start = time.time()

        
        if from_nd2 == True:
            from nd2reader import ND2Reader
            imagelist = []
            with ND2Reader(data) as images:
                for fov in images:
                    imagelist.append(fov)
            print('images read in from nd2')

        else:
            imagelist = data
            print('images read in from tif')

        if multicycle == True:
            imagelist = [i[0,:] for i in imagelist]

        if channel_index != None:
            imagelist = [i[channel_index,:] for i in imagelist]


        img_width = imagelist[0].shape[1]
        # break list of images into list of sublists of images from the same row
        imagelist = [imagelist[end - length:end] if flag == 0 else imagelist[end - length:end][::-1] for length, end, flag in zip(seq_lengths, accumulate(seq_lengths), order_flags)]

        overlap_w = int(overlap_frac * img_width)
        blendedlist = []
        for i in range(len(imagelist)):
            for j in range(len(imagelist[i])-1):
                if j == 0:
                    blended = multi_band_blending_subset(imagelist[i][j], imagelist[i][j+1], overlap_w=overlap_w)
                else:
                    blended = multi_band_blending_subset(blended, imagelist[i][j+1], overlap_w=overlap_w)

            blendedlist.append(blended)

        ### blend columns

        dim = max([i.shape[1] for i in blendedlist])
        rows = np.zeros((len(blendedlist),img_width,dim))

        for i in range(len(blendedlist)):
            dimdiff = dim - blendedlist[i].shape[1]
            if dimdiff == 0:
                rows[i] = blendedlist[i]
            else:
                rows[i,:,int(dimdiff/2):-int(dimdiff/2)] = blendedlist[i]

        rows = np.swapaxes(rows,1,2)

        for i in range(len(blendedlist)-1):
            if i == 0:
                blended = multi_band_blending_subset(rows[i], rows[i+1], overlap_w=overlap_w)
            else:
                blended = multi_band_blending_subset(blended, rows[i+1], overlap_w=overlap_w)

        blended = np.swapaxes(blended,0,1).astype('uint16')
        end = time.time()
        print(end - start)

        return blended

    @staticmethod
    def _stitch_sitegrid(data_shape=1480, seq_lengths=None, predefined_seq_lengths=None, order_flags='Meander', overlap_frac=0.15,
        nsites=333):

        import time
        from itertools import accumulate


        if (seq_lengths == None) & (predefined_seq_lengths == None):
            error = 'Must choose a value for seq_lengths or predefined_seq_lengths'
            raise ValueError(error)
        
        if predefined_seq_lengths == 'seq_lengths_10x_6w':
            seq_lengths = [5,9,13,15,17,17,19,19,21,21,21,21,21,19,19,17,17,15,13,9,5]

        if predefined_seq_lengths == 'seq_lengths_20x_6w':
            seq_lengths = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]

        if order_flags == 'Meander':
            order_flags = [1 if (i % 2 != 0) else 0 for i in range(len(seq_lengths))]
        else:
            order_flags = [0 for i in range(len(seq_lengths))]

        start = time.time()

        img_width = data_shape

        imagelist = [np.zeros((data_shape,data_shape)).fill(i) for i in range(nsites)]
        # break list of images into list of sublists of images from the same row
        imagelist = [imagelist[end - length:end] if flag == 0 else imagelist[end - length:end][::-1] for length, end, flag in zip(seq_lengths, accumulate(seq_lengths), order_flags)]

        overlap_w = int(overlap_frac * img_width)
        blendedlist = []
        for i in range(len(imagelist)):
            for j in range(len(imagelist[i])-1):
                if j == 0:
                    blended = multi_band_blending_subset(imagelist[i][j], imagelist[i][j+1], overlap_w=overlap_w)
                else:
                    blended = multi_band_blending_subset(blended, imagelist[i][j+1], overlap_w=overlap_w)

            blendedlist.append(blended)

        ### blend columns

        dim = max([i.shape[1] for i in blendedlist])
        rows = np.zeros((len(blendedlist),img_width,dim))

        for i in range(len(blendedlist)):
            dimdiff = dim - blendedlist[i].shape[1]
            if dimdiff == 0:
                rows[i] = blendedlist[i]
            else:
                rows[i,:,int(dimdiff/2):-int(dimdiff/2)] = blendedlist[i]

        rows = np.swapaxes(rows,1,2)

        for i in range(len(blendedlist)-1):
            if i == 0:
                blended = multi_band_blending_subset(rows[i], rows[i+1], overlap_w=overlap_w)
            else:
                blended = multi_band_blending_subset(blended, rows[i+1], overlap_w=overlap_w)

        blended = np.swapaxes(blended,0,1).astype('uint16')
        end = time.time()
        print(end - start)

        return blended

    @staticmethod
    def _align_cell_mask(data_1, data_2, channel_index1, channel_index2, data_1_cells, data_2_cells, site, pheno_seq_lengths, overlap, pheno_sbs_mag_fold_change = 2,
        upsample_factor=1):
        """Align a series of images to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        import math
        
        small_shape = data_1.shape[0]

        if pheno_seq_lengths == 'seq_lengths_20x_6w':

            grid_20x = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]
            arr20x = np.zeros((max(grid_20x),max(grid_20x)))
            for i in range(arr20x.shape[0]):
                middle = int(len(arr20x[i])/2) ## only works if odd
                num_tiles = grid_20x[i]
                width = math.trunc(num_tiles/2)
                arr20x[i][middle-width:middle+width+1] = 1
            fillrows = np.nonzero(arr20x)[0]
            fillcols = np.nonzero(arr20x)[1]
            arr20x[arr20x == 0] = -1
            ct = 0
            for i in (np.unique(fillrows)):
                nonzeroinds = fillcols[np.argwhere(fillrows==i)]
                if i % 2 == 0: #row is even, grid goes right
                    for j in range(len(nonzeroinds)): #col indices
                        arr20x[i,nonzeroinds[j]] = ct
                        ct += 1
                else:
                    for j in reversed(range(len(nonzeroinds))): #col indices
                        arr20x[i,nonzeroinds[j]] = ct
                        ct += 1
            center_point = (arr20x.shape[0]/2.,arr20x.shape[1]/2.)
            print(np.argwhere(arr20x == site), center_point)
            tl_coords = (np.argwhere(arr20x == site) - center_point)*small_shape*(1-overlap)/pheno_sbs_mag_fold_change
        else:
            error = 'No predefined phenotyping grid shape with this name: {0}'
            raise ValueError(error.format(pheno_seq_lengths))

        print(tl_coords, site)
        from skimage.util import img_as_uint

        data_1 = np.rot90(np.fliplr(img_as_uint(skimage.transform.resize(data_1, tuple([int(s/pheno_sbs_mag_fold_change) for s in data_1.shape]),
                        anti_aliasing=True, mode = 'constant'))))
        # # add new axis to single-channel images
        # if data_1.ndim == 2:
        #     data_1 = data_1[np.newaxis,:]
        # if data_2.ndim == 2:
        #     data_2 = data_2[np.newaxis,:]

        # subsample data_2 to region around the cell mask, +/- half a site to account for offsets
        center_point = (data_2.shape[0]/2.,data_2.shape[1]/2.)
        ystart = int(center_point[0]+tl_coords[0][0]-int(small_shape/2))
        yend = int(center_point[0]+tl_coords[0][0]+int(small_shape*3/2))
        xstart = int(center_point[1]+tl_coords[0][1]-int(small_shape/2))
        xend = int(center_point[1]+tl_coords[0][1]+int(small_shape*3/2))
        print(ystart, yend, xstart, xend, center_point, tl_coords)
        data_2 = data_2[max(ystart,0):yend,
                          max(xstart,0):xend]
    
        print(data_2.shape)

        
        data_1m = np.zeros(data_2.shape, dtype = 'uint16')
        data_1m[:data_1.shape[0],:data_1.shape[1]] = data_1 

        # data_2m = data_2.copy()
        # mask = data_1m != 0
        # data_1m[mask == True] = 1 # binary 0/1
        # #data_1m[mask == False] = 1e-3 # binary 0/1, but float to avoid division by zero in register_translation
        # #data_1m = img_as_uint(data_1m)
        # mask = data_2m != 0
        # data_2m[mask == True] = 1 # binary 0/1
        #data_2m[mask == False] = 1e-3 # binary 0/1
        #data_1m = ndimage.distance_transform_edt(-data_1m)
        #data_2m = ndimage.distance_transform_edt(-data_2m)
        #save_output('data1m.tif',data_1m)
        #save_output('data2m.tif',data_2m)

        #print(data_2m.shape, data_1m.shape, len(data_1m))

        print('aligning')
        images = data_1m[0], data_2[0]
        #_, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offset, _, _, = skimage.feature.register_translation(data_1m[0], data_2[0])
        print(offset)
        print(len(data_1m),data_2.dtype,data_1m.dtype)
        offsets = [offset] * len(data_2)
        aligned2 = ops.process.Align.apply_offsets(data_2, offsets)
        print(aligned2.shape, data_2m.shape)
        #aligned = np.vstack((data_1m, aligned2))
        aligned = np.stack((data_1m[np.newaxis,:], aligned2[np.newaxis,:]))
        print(aligned.shape)

        return aligned

    @staticmethod
    def _align_cell_masktrash(data_1, data_2, site, pheno_seq_lengths, overlap, pheno_sbs_mag_fold_change = 2,
        upsample_factor=1):
        """Align a series of images to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        import math
        
        small_shape = data_1.shape[0]

        if pheno_seq_lengths == 'seq_lengths_20x_6w':

            grid_20x = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]
            arr20x = np.zeros((max(grid_20x),max(grid_20x)))
            for i in range(arr20x.shape[0]):
                middle = int(len(arr20x[i])/2) ## only works if odd
                num_tiles = grid_20x[i]
                width = math.trunc(num_tiles/2)
                arr20x[i][middle-width:middle+width+1] = 1
            fillrows = np.nonzero(arr20x)[0]
            fillcols = np.nonzero(arr20x)[1]
            arr20x[arr20x == 0] = -1
            ct = 0
            for i in (np.unique(fillrows)):
                nonzeroinds = fillcols[np.argwhere(fillrows==i)]
                if i % 2 == 0: #row is even, grid goes right
                    for j in range(len(nonzeroinds)): #col indices
                        arr20x[i,nonzeroinds[j]] = ct
                        ct += 1
                else:
                    for j in reversed(range(len(nonzeroinds))): #col indices
                        arr20x[i,nonzeroinds[j]] = ct
                        ct += 1
            center_point = (arr20x.shape[0]/2.,arr20x.shape[1]/2.)
            print(np.argwhere(arr20x == site), center_point)
            tl_coords = (np.argwhere(arr20x == site) - center_point)*small_shape*(1-overlap)/pheno_sbs_mag_fold_change
        else:
            error = 'No predefined phenotyping grid shape with this name: {0}'
            raise ValueError(error.format(pheno_seq_lengths))
        # if sbs_seq_lengths == 'seq_lengths_10x_6w':
        #     seq_lengths = [5,9,13,15,17,17,19,19,21,21,21,21,21,19,19,17,17,15,13,9,5]

        # else:
        #     error = 'No predefined SBS grid shape with this name: {0}'
        #     raise ValueError(error.format(sbs_seq_lengths))


        print(tl_coords, site)
        from skimage.util import img_as_uint

        data_1 = np.rot90(np.fliplr(img_as_uint(skimage.transform.resize(data_1, tuple([int(s/pheno_sbs_mag_fold_change) for s in data_1.shape]),
                        anti_aliasing=True, mode = 'constant'))))
        # # add new axis to single-channel images
        # if data_1.ndim == 2:
        #     data_1 = data_1[np.newaxis,:]
        # if data_2.ndim == 2:
        #     data_2 = data_2[np.newaxis,:]

        # subsample data_2 to region around the cell mask, +/- half a site to account for offsets
        center_point = (data_2.shape[0]/2.,data_2.shape[1]/2.)
        ystart = int(center_point[0]+tl_coords[0][0]-int(small_shape/2))
        yend = int(center_point[0]+tl_coords[0][0]+int(small_shape*3/2))
        xstart = int(center_point[1]+tl_coords[0][1]-int(small_shape/2))
        xend = int(center_point[1]+tl_coords[0][1]+int(small_shape*3/2))
        print(ystart, yend, xstart, xend, center_point, tl_coords)
        data_2 = data_2[max(ystart,0):yend,
                          max(xstart,0):xend]
    
        save_output('data1.tif',data_1)
        print(data_2.shape)

        
        data_1m = np.zeros(data_2.shape, dtype = 'uint16')
        data_1m[:data_1.shape[0],:data_1.shape[1]] = data_1 

        data_2m = data_2.copy()
        mask = data_1m != 0
        data_1m[mask == True] = 1 # binary 0/1
        #data_1m[mask == False] = 1e-3 # binary 0/1, but float to avoid division by zero in register_translation
        #data_1m = img_as_uint(data_1m)
        mask = data_2m != 0
        data_2m[mask == True] = 1 # binary 0/1
        #data_2m[mask == False] = 1e-3 # binary 0/1
        #data_1m = ndimage.distance_transform_edt(-data_1m)
        #data_2m = ndimage.distance_transform_edt(-data_2m)
        save_output('data1m.tif',data_1m)
        save_output('data2m.tif',data_2m)

        print(data_2m.shape, data_1m.shape, len(data_1m))

        print('aligning')
        # images = data_1m[0], data_2m[0]
        # #_, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        # offset, _, _, = skimage.feature.register_translation(data_1m[0], data_2m[0])
        # print(offset)
        # print(len(data_1m),data_2m.dtype,data_1m.dtype)
        # offsets = [offset] * len(data_2m)
        # aligned2 = ops.process.Align.apply_offsets(data_2m, offsets)
        # save_output('aligned2.tif',aligned2[0])
        # print(aligned2.shape, data_2m.shape)
        # #aligned = np.vstack((data_1m, aligned2))
        # aligned = np.stack((data_1m[np.newaxis,:], aligned2[np.newaxis,:]))
        # print(aligned.shape)

        from skimage.feature import ORB, match_descriptors
        from skimage.transform import EuclideanTransform, warp
        from skimage.measure import ransac

        orb = ORB(n_keypoints=100, fast_threshold=0.05) #https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
    
        orb.detect_and_extract(data_1m)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors
                
        orb.detect_and_extract(data_2m)   
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors
        print('match descriptors')
        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
        src = keypoints2[matches12[:, 1]][:, ::-1]
        dst = keypoints1[matches12[:, 0]][:, ::-1]

        print('ransac')        
        model_robust, inliers = ransac((src, dst), EuclideanTransform,
                           min_samples=2, residual_threshold=15, max_trials=100)
        print(inliers)
        print('warping')
        aligned2 = warp(data_2m, model_robust.inverse, preserve_range=True).astype(data_2m.dtype)

        save_output('aligned2.tif',aligned2)
        aligned = np.stack((data_1m[np.newaxis,:], aligned2[np.newaxis,:]))
        return aligned

    @staticmethod
    def _align_by_channel_4ch(data_1, data_2, data_3, data_4,channel_index1=0, channel_index2=0, 
        channel_index3=0, channel_index4=0, upsample_factor=1):
        """Align a series of images to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        print(upsample_factor)
        # add new axis to single-channel images
        if data_1.ndim == 2:
            data_1 = data_1[np.newaxis,:]
        if data_2.ndim == 2:
            data_2 = data_2[np.newaxis,:]
        if data_3.ndim == 2:
            data_3 = data_3[np.newaxis,:]
        if data_4.ndim == 2:
            data_4 = data_4[np.newaxis,:]
 
        print('aligning')
        images = data_1[channel_index1], data_2[channel_index2]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned2 = ops.process.Align.apply_offsets(data_2, offsets)

        images = data_1[channel_index1], data_3[channel_index3]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_3)
        aligned3 = ops.process.Align.apply_offsets(data_3, offsets)

        images = data_1[channel_index1], data_4[channel_index4]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_4)
        aligned4 = ops.process.Align.apply_offsets(data_4, offsets)

        aligned = np.vstack((data_1, aligned2, aligned3, aligned4))
        return aligned

    @staticmethod
    def _align_by_channel_5ch(data_1, data_2, data_3, data_4, data_5, channel_index1=0, channel_index2=0, 
        channel_index3=0, channel_index4=0, channel_index5=0, upsample_factor=1):
        """Align a series of images to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """

        # add new axis to single-channel images
        if data_1.ndim == 2:
            data_1 = data_1[np.newaxis,:]
        if data_2.ndim == 2:
            data_2 = data_2[np.newaxis,:]
        if data_3.ndim == 2:
            data_3 = data_3[np.newaxis,:]
        if data_4.ndim == 2:
            data_4 = data_4[np.newaxis,:]
        if data_5.ndim == 2:
            data_5 = data_5[np.newaxis,:]

        print('aligning')
        images = data_1[channel_index1], data_2[channel_index2]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned2 = ops.process.Align.apply_offsets(data_2, offsets)

        images = data_1[channel_index1], data_3[channel_index3]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_3)
        aligned3 = ops.process.Align.apply_offsets(data_3, offsets)

        images = data_1[channel_index1], data_4[channel_index4]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_4)
        aligned4 = ops.process.Align.apply_offsets(data_4, offsets)

        images = data_1[channel_index1], data_5[channel_index5]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_5)
        aligned5 = ops.process.Align.apply_offsets(data_5, offsets)
        #print(data_1.shape)
        #print(aligned.shape)

        aligned = np.vstack((aligned5, data_1, aligned2, aligned3, aligned4))
        #aligned = np.vstack((data_1, aligned2, aligned3, aligned4, aligned5))
        return aligned

    def fix_channel_offsets(data, channel_offsets):
        d = data.transpose([1, 0, 2, 3])
        x = [lasagna.utils.offset(a, b) for a,b in zip(d, channel_offsets)]
        x = np.array(x).transpose([1, 0, 2, 3])
        return x

    @staticmethod
    def _align_SBS_rigidbody(data, window=1, remove_c0_DAPI=False):
        print('align rigid body')
        """Rigid alignment of sequencing cycles and channels with rotation and translation. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).

        """
        print(len(data))
        print(data[0].shape)
        if remove_c0_DAPI == True:
            dapi = data[0][0][np.newaxis,:]#, len(data), axis=0)
            data[0] = data[0][1:]

        print(len(data))


        data = np.array(data)#
        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'
       
        from skimage.feature import ORB, match_descriptors
        from skimage.transform import EuclideanTransform, warp
        from skimage.measure import ransac


        orb = ORB(n_keypoints=300, fast_threshold=0.001, downscale = .05) #https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
        
        data_window = Align.apply_window(data, window = window)
        print('data_window shape: ', data_window.shape)
        rotatedall = []
        for cycle in range(data.shape[0]):
            rotated = []
            for channel in range(data.shape[1]):
                print(cycle, channel)
                if (cycle == 0) & (channel == 0):
                    print('image1')
                    image1 = data_window[cycle, channel]
                    orb.detect_and_extract(image1)
                    keypoints1 = orb.keypoints
                    descriptors1 = orb.descriptors
                    rotated.append(data[cycle, channel])
                    #continue
                else:
                    print(cycle, channel)
                    image2 = data_window[cycle,channel]
                    orb.detect_and_extract(image2)   
                    keypoints2 = orb.keypoints
                    descriptors2 = orb.descriptors
                    print('match descriptors')
                    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
                    src = keypoints2[matches12[:, 1]][:, ::-1]
                    dst = keypoints1[matches12[:, 0]][:, ::-1]

                    print('ransac')        
                    model_robust, inliers = ransac((src, dst), EuclideanTransform,
                                       min_samples=20, residual_threshold=0.5, max_trials=500)

                    print('warping')
                    image2 = warp(data[cycle, channel], model_robust.inverse, preserve_range=True).astype(data.dtype)
                    print(np.mean(image2))
                    print('mean done')
                    rotated.append(image2)
                    
            rotatedall.append(rotated)

        rotatedall = np.array(rotatedall)
        

        print(dapi.shape, rotatedall.shape)
        if remove_c0_DAPI == True:
            rotatedall = np.hstack((dapi,rotatedall))

        print(rotatedall.shape)
        return rotatedall

    @staticmethod
    def _align_stack_rigidbody(data, remove_c0_DAPI=False, window = 1.5, upsample_factor = 2):
        print('align rigid body')
        """Rigid alignment of sequencing cycles and channels with rotation and translation. 

        Expects `data` to be an array with dimensions (CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one.

        """
        data = np.array(data).astype('float64')
        if remove_c0_DAPI == True:
            dapi = data[0,0:1]
            data = data[0,1:]

        assert data.ndim == 3, 'Input data must have dimensions CHANNEL, I, J'

        data_mask = Align.apply_window(data.copy(), window=window)
       

        import imreg_dft as ird
        from imreg_dft.tiles import resample

        rotated = []
        for channel in range(data.shape[0]):
            if (channel in (range(4))):
                if channel == 3:
                    image1 = ird.tiles.resample(data_mask[channel], upsample_factor)
                rotated.append(data[channel])
            elif channel == 4:
                image2 = ird.tiles.resample(data_mask[channel], upsample_factor)
       
                transform = ird.similarity(image1, image2, numiter=20, constraints = dict(scale=[1, 0], angle=[0, 0.2], tx=[0, 3], ty=[0, 3]))
                # constraints defined as center plus deviation from center
                print('tform', transform)
                result = ird.transform_img_dict(data[channel], transform, bgval=None, order=1)
                rotated.append(result)
            else: 
                result = ird.transform_img_dict(data[channel], transform, bgval=None, order=1)
                rotated.append(result)

                

        rotated = np.array(rotated)
           
        if remove_c0_DAPI == True:
            rotated = np.vstack((dapi,rotated))
        rotated = rotated.astype(data.dtype)

        return rotated



    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1,
        align_within_cycle=True, keep_trailing=False, n=1):
        """Rigid alignment of sequencing cycles and channels. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).
        """
        print(type(data))
        data = np.array(data)
        print(data.shape)
        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])

        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align between SBS channels for each cycle
        aligned = data.copy()
        if align_within_cycle:
            align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
            # if data.shape[1] == 4:
            #     n = 0
            #     align_it = lambda x: Align.align_within_cycle(x, window=window, 
            #         upsample_factor=upsample_factor, cutoff=cutoff)
            # else:
            #     n = 1
            
            aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])
            

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0, 
                                window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, 1:], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        images = data_1[channel_index], data_2[channel_index]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        return aligned
        
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J).
        """

        
        print(data.shape)
        print(data.ndim)
        if isinstance(data, list):
            dapi = data[0]
        elif data.ndim == 3:
            dapi = data[0]
        elif data.ndim == 4:
            dapi = data[0][0]
        else:
            dapi = data

        print(dapi.shape)
        print(dapi.ndim)
        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_stack(dapi, threshold, area_min, area_max):
        """Find nuclei from a nuclear stain (e.g., DAPI). Expects data to have shape (I, J) 
        (segments one image) or (N, I, J) (segments a series of DAPI images).
        """
        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        find_nuclei = ops.utils.applyIJ(ops.process.find_nuclei)
        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

   # @staticmethod
   # def _segment_cells(data, nuclei, threshold):
    #    """Segment cells from aligned data. Matches cell labels to nuclei labels.
   #     Note that labels can be skipped, for example if cells are touching the 
   #     image boundary.
   #     """
   #     if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
   #         mask = data[:, 1:].min(axis=0).mean(axis=0)
   #     elif data.ndim == 3:
   #         mask = np.median(data[1:], axis=0)
   #     elif data.ndim == 2:
   #         mask = data
   #     else:
    #        raise ValueError

     #   mask = mask > threshold
      #  try:
            # skimage precision warning
       #     with warnings.catch_warnings():
        #        warnings.simplefilter("ignore")
         #       cells = ops.process.find_cells(nuclei, mask)
       # except ValueError:
        #    print('segment_cells error -- no cells')
         #   cells = nuclei

        # return cells

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        loged = ops.process.log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _transform_log_bychannelcycle(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)

        cycles = data.shape[1]
        channels = data.shape[2]
        
        loged = []
        for cycle in range(cycles):
            tmp = []
            for channel in range(channels):
                print(cycle, channel)
                print(data.shape)
                print(data[0,cycle,channel].shape)
                tmp.append(ops.process.log_ndi(data[0,cycle, channel], sigma = sigma))
                
            loged.append(tmp)
        
        loged = np.array(loged)[np.newaxis, :]
        print(loged.shape)

        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation to estimate sequencing read locations.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        leading_dims = tuple(range(0, data.ndim - 2))
        consensus = np.std(data, axis=leading_dims)

        return consensus
    
    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """Find local maxima and label by difference to next-highest neighboring
        pixel.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        if data.ndim == 2:
            data = [data]

        peaks = [ops.process.find_peaks(x, n=width) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """Apply a maximum filter in a window of `width`.
        """
        import scipy.ndimage.filters

        if data.ndim == 2:
            data = data[None, None]
        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (e.g., well and tile) and 
        label at that position in integer mask `cells`.
        """

        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_peaks))

        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        for k,v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases, peaks=None, correction_only_in_cells=True):
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
            .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_reads_mbc(df_bases, peaks=None, position_column = 'site'):
        """Median correction performed independently for each tile *and cycle*.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        median call is by cycle

        Currently only written for correction_only_in_cells=True
        """
        if df_bases is None:
            return
        if len(df_bases.query('cell > 0')) == 0:
            return


        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        if position_column != 'site':
            df_bases['site'] = df_bases[position_column]

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call_bycycle, cycles=cycles, channels=channels)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_cells(df_reads, q_min=0):
        """Median correction performed independently for each tile.
        """
        if df_reads is None:
            return
        
        return (df_reads
            .query('Q_min >= @q_min')
            .pipe(ops.in_situ.call_cells))

    @staticmethod
    def _extract_features(data, labels, wildcards, features=None):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        from ops.features import features_basic
        print(features_basic)
        features = features.copy() if features else dict()
        features.update(features_basic)
        df = feature_table(data, labels, features)

        for k,v in sorted(wildcards.items()):
            df[k] = v
        
        return df

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)
             .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift_myc)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        if (nuclei.max() == 0) or (cells.max() == 0):
            return

        import ops.features

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n = (Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
            .rename(columns={'area': 'area_nuclear'}))

        df_c =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .drop(['i', 'j'], axis=1).rename(columns={'area': 'area_cell'}))


        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('label'), df_c.set_index('label')], axis=1, join='inner')
                .reset_index())

        return (df
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation_live(data, nuclei, wildcards):
        def _extract_phenotype_translocation_simple(data, nuclei, wildcards):
            import ops.features
            features = ops.features.features_translocation_nuclear_simple
            
            return (Snake._extract_features(data, nuclei, wildcards, features)
                .rename(columns={'label': 'cell'}))

        extract = _extract_phenotype_translocation_simple
        arr = []
        for i, (frame, nuclei_frame) in enumerate(zip(data, nuclei)):
            arr += [extract(frame, nuclei_frame, wildcards).assign(frame=i)]

        return pd.concat(arr)

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return (Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_geom(labels, wildcards):
        from ops.features import features_geom
        return Snake._extract_features(labels, labels, wildcards, features_geom)

    @staticmethod
    def _analyze_single(data, alignment_ref, cells, peaks, 
                        threshold_peaks, wildcards, channel_ix=1):
        if alignment_ref.ndim == 3:
            alignment_ref = alignment_ref[0]
        data = np.array([[alignment_ref, alignment_ref], 
                          data[[0, channel_ix]]])
        aligned = ops.process.Align.align_between_cycles(data, 0, window=2)
        loged = Snake._transform_log(aligned[1, 1])
        maxed = Snake._max_filter(loged, width=3)
        return (Snake._extract_bases(maxed, peaks, cells, bases=['-'],
                    threshold_peaks=threshold_peaks, wildcards=wildcards))

    @staticmethod
    def _track_live_nuclei(nuclei, tolerance_per_frame=5):
        
        # if there are no nuclei, we will have problems
        count = nuclei.max(axis=(-2, -1))
        if (count == 0).any():
            error = 'no nuclei detected in frames: {}'
            print(error.format(np.where(count == 0)))
            return np.zeros_like(nuclei)

        import ops.timelapse

        # nuclei coordinates
        arr = []
        for i, nuclei_frame in enumerate(nuclei):
            extract = Snake._extract_phenotype_minimal
            arr += [extract(nuclei_frame, nuclei_frame, {'frame': i})]
        df_nuclei = pd.concat(arr)

        # track nuclei
        motion_threshold = len(nuclei) * tolerance_per_frame
        G = (df_nuclei
          .rename(columns={'cell': 'label'})
          .pipe(ops.timelapse.initialize_graph)
        )

        cost, path = ops.timelapse.analyze_graph(G)
        relabel = ops.timelapse.filter_paths(cost, path, 
                                    threshold=motion_threshold)
        nuclei_tracked = ops.timelapse.relabel_nuclei(nuclei, relabel)

        return nuclei_tracked

    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], Snake.call_from_snakemake(f))

    @staticmethod
    def call_from_snakemake(f):
        """Turn a function that acts on a mix of image data, table data and other 
        arguments and may return image or table data into a function that acts on 
        filenames for image and table data, plus other arguments.

        If output filename is provided, saves return value of function.

        Supported input and output filetypes are .pkl, .csv, and .tif.
        """
        def g(**kwargs):

            # split keyword arguments into input (needed for function)
            # and output (needed to save result)
            input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)
            # load arguments provided as filenames

            input_kwargs = {k: load_arg(v) for k,v in input_kwargs.items()}
            results = f(**input_kwargs)

            if 'output' in output_kwargs:
                outputs = output_kwargs['output']
                
                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = '{0} output filenames provided for {1} results'
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

        return functools.update_wrapper(g, f)


Snake.load_methods()


def remove_channels(data, remove_index):
    """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
    """
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    data = data[..., channels_mask, :, :]
    return data


# IO


def load_arg(x):
    """Try loading data from `x` if it is a filename or list of filenames.
    Otherwise just return `x`.
    """
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]
    
    for f in one_file, many_files:
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # wasn't a file, probably a string arg
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # failed to load file
                return None
            pass
    else:
        return x


def save_output(filename, data, **kwargs):
    """Saves `data` to `filename`. Guesses the save function based on the
    file extension. Saving as .tif passes on kwargs (luts, ...) from input.
    """
    filename = str(filename)
    if data is None:
        # need to save dummy output to satisfy Snakemake
        with open(filename, 'w') as fh:
            pass
        return
    if filename.endswith('.tif'):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith('.pkl'):
        return save_pkl(filename, data)
    elif filename.endswith('.csv'):
        return save_csv(filename, data)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def load_csv(filename):
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None
    return df


def load_pkl(filename):
    df = pd.read_pickle(filename)
    if len(df) == 0:
        return None


def load_tif(filename):
    return ops.io.read_stack(filename)


def save_csv(filename, df):
    df.to_csv(filename, index=None)


def save_pkl(filename, df):
    df.to_pickle(filename)


def save_tif(filename, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # `data` can be an argument name for both the Snake method and `save_stack`
    # overwrite with `data_` 
    kwargs['data'] = data_
    ops.io.save_stack(filename, **kwargs)


def restrict_kwargs(kwargs, f):
    """Partition `kwargs` into two dictionaries based on overlap with default 
    arguments of function `f`.
    """

    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard


def load_file(filename):
    """Attempt to load file, raising an error if the file is not found or 
    the file extension is not recognized.
    """
    if not isinstance(filename, str):
        raise TypeError
    if not os.path.isfile(filename):
        raise IOError(2, 'Not a file: {0}'.format(filename))
    if filename.endswith('.tif'):
        return load_tif(filename)
    elif filename.endswith('.pkl'):
        return load_pkl(filename)
    elif filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise IOError(filename)


def get_arg_names(f):
    """List of regular and keyword argument names from function definition.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    """Get the kwarg defaults as a dictionary.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def load_well_tile_list(filename):
    if filename.endswith('pkl'):
        wells, tiles = pd.read_pickle(filename)[['well', 'site']].values.T
    elif filename.endswith('csv'):
        wells, tiles = pd.read_csv(filename)[['well', 'site']].values.T
    return wells, tiles
