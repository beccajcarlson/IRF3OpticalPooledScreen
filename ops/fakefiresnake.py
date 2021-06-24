import inspect
import functools
import os
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd
import skimage
import ops.features
import ops.process
import ops.io
import ops.rolling_ball
import ops.in_situ
from ops.process import Align

# ####IMPORTS FOR BECCA FUNCS######
from ops.stitching import *
from scipy.ndimage import morphology, distance_transform_edt
#import mahotas #for zernike features
from astropy.stats import median_absolute_deviation
from scipy import ndimage

######


class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    @staticmethod
    def _slice_channels(data, to_keep):
     ## If z present, max projects along z, keeps multichannel together
        data = np.array(data)
        print(data.shape)
        data = data[:,:to_keep,:]
        print(data.shape)
        return data


    @staticmethod
    def _nd2_to_tif(data, custom_range = None):
     ## If z present, max projects along z, keeps multichannel together
        from nd2reader import ND2Reader
        imagelist = []
        with ND2Reader(data) as images:
            print(images.sizes)

            images.bundle_axes = ''
            #print(images.sizes.keys())
            # if 'z' in images.sizes.keys():
            #     print('z-stack')
            #     images.bundle_axes += 'z'
            # if 't' in images.sizes.keys():
            #     print('timecourse')
            #     images.bundle_axes += 't'
            if 'c' in images.sizes.keys():
                print('multichannel')
                c_len = images.sizes['c']
            if 't' in images.sizes.keys():
                print('timecourse')
                t_len = images.sizes['t']
            if 'v' in images.sizes.keys():
                print('multipoint')
                v_len = images.sizes['v']
            #print(c_len, t_len, v_len)
            #images.bundle_axes += 'tczyx' 
            #print(images.bundle_axes)

            # if 't' in images.sizes.keys():
            #     print('timecourse')
            #     images.iter_axes = 't'
            # images.iter_axes = 'v'

            # print(images.iter_axes)
            # print(len(images))

            #img = ND2Reader.get_frame_2D(images, c=0, t=0, z=2430, v=0, x=0, y=0) # v and t can't be non-zero, don't know why
            #print(img.shape)
                # gives diff keyerrors
            if custom_range != None:
                v_len = custom_range
            for v in range(v_len):
                tmpt = []
                for t in range(t_len):
                    tmpc = []
                    for c in range(c_len):
                        #print(c,t,v)
                       # z = v_len + t#int(v_len*t + t)
                        z = (v_len)*t + v
                        #print(z)
                        tmpc.append(ND2Reader.get_frame_2D(images, c=c, z=z, x=0, y=0))
                    tmpt.append(np.array(tmpc))

                # print('tmpc tmpt shape')
                # print(len(tmpc),len(tmpt),tmpc[0].shape,tmpt[0].shape)
                imagelist.append(np.array(tmpt))

            print(len(imagelist))
            # for fov in images:

            #     print(fov)
            #     # if len(imagelist) == 10:
            #     #     return imagelist
            #     print(fov.shape)
            #     if 'z' in images.sizes.keys():
            #         fov = fov.max(axis = 0) # max project z
            #     imagelist.append(fov)

        return imagelist

    @staticmethod
    def _nd2_to_tif_z(data,start_site,end_site):
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
            #print(images.shape)
            print(images.bundle_axes)
            images.iter_axes = 'v'
            i = 0
            for fov in images[start_site:end_site]:
                if 'z' in images.sizes.keys():
                    fov = np.array(fov.max(axis = 0)) # max project z
                imagelist.append(np.array(fov).astype('float32'))
                i += 1
                print(i)
            print('imlist len ', len(imagelist))
            print(imagelist[0].shape)
        return imagelist

    @staticmethod
    def _nd2_to_tif_noz(data,start_site=None,end_site=None):
     ## If z present, max projects along z, keeps multichannel together
        from nd2reader import ND2Reader
        imagelist = []
        with ND2Reader(data) as images:
            images.bundle_axes = ''
          
            if 'c' in images.sizes.keys():
                print('multichannel')
                images.bundle_axes += 'c'

            images.bundle_axes += 'xy'
            #print(images.shape)
            #print(images.sizes, start_site, end_site, len(images[start_site:end_site]), len(images))
            images.iter_axes = 'z'
            i = 0

            if (start_site == None) & (end_site == None):
                for fov in images:
                    # print(fov)
                    # print(start_site,end_site)
                    # print(fov.shape)
                    imagelist.append(fov.astype('float32'))
                    #print(type(imagelist[0][0][0,0]))
                    i += 1
                    #print(i)
            else:
                for fov in images[start_site:end_site]:
                    # print(fov)
                    # print(start_site,end_site)
                    # print(fov.shape)
                    imagelist.append(fov.astype('float32'))
                    #print(type(imagelist[0][0][0,0]))
                    i += 1
                    #print(i)
            print('imlist len ', len(imagelist))
            print(imagelist[0].shape)
        return imagelist



    @staticmethod
    def _stitch_well(data, seq_lengths=None, predefined_seq_lengths=None, order_flags='Meander', overlap_frac=0.15):

        ## generates image of well from tiles using multi-band blending of adjacent tiles based on predefined grid 
        ## seq_lengths = list of #s of fov per row of well
        ## predefined_seq_lengths = a string corresponding to a commonly used grid
        ## order_flags = list of flags as long as seq_lengths. 1 indicates that image row order is reversed
        ## overlap_frac = fraction of adjacent images overlapping

        import time
        from itertools import accumulate
        from nd2reader import ND2Reader

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

        imagelist = []
        with ND2Reader(data) as images:
            for fov in images:
                imagelist.append(fov)
            
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
    def _rolling_ball_bsub(data, radius, shrink_factor=None, ball=None, 
    mem_cap=1e9):

        print('RB',data)
        balls = []
        for j in range(data.shape[1]): #channel
            balls.append(ops.rolling_ball.get_ball(data[0,j],radius=radius)) # derive ball for each channel at t=0

        bsub = []
        for i in range(data.shape[0]): #time
            tmp = []
            for j in range(data.shape[1]): #channel
                tmp.append(ops.rolling_ball.subtract_background(data[i,j],radius=radius,ball=balls[j][0],
                    shrink_factor=balls[j][1]))
            bsub.append(np.array(tmp))
        bsub = np.array(bsub)

        print(bsub.shape)



        return bsub

    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
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
    def _track_live_nuclei(nuclei, tolerance_per_frame=8, tstart=None, tend=None):
        print(nuclei.shape, tstart, tend)
        if (tstart != None) & (tend != None):
            nuclei = nuclei[tstart:tend,:,:]
        print(nuclei.shape)
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


    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=5, window=2, cutoff=1,
        align_within_cycle=True, keep_trailing=False, n=1):
        """Rigid alignment of sequencing cycles and channels. 
        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).
        """
        data = np.array(data)
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
    def _align_SBS_special_c0_dim(data, method='DAPI', upsample_factor=5, window=2, cutoff=1,
        align_within_cycle=True, keep_trailing=False, n=1):
        """Rigid alignment of sequencing cycles and channels. 
        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).
        """
        data = np.array(data)
        print(data.shape,data[0].shape)

        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])
       
        data[0] = data[0][np.r_[0,2:6]] #remove c0 channel 1
        data = np.stack(data,axis=0)
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
    def _align_SBS_rigidbody(data, upsample_factor=3., window=1.1, remove_c0_DAPI=False):
        print('align rigid body')
        """Rigid alignment of sequencing cycles and channels with rotation and translation. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).

        """

        if remove_c0_DAPI == True:
            dapi = np.repeat(data[0][0][np.newaxis,np.newaxis, :,:], len(data), axis=0)
            data[0] = data[0][1:]

        print(len(data))


        data = np.array(data)#
        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'
       
        from skimage.feature import ORB, match_descriptors
        from skimage.transform import EuclideanTransform, warp
        from skimage.measure import ransac


        orb = ORB(n_keypoints=200, fast_threshold=0.05) #https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
        
        data_window = Align.apply_window(data, window=window)

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
                                       min_samples=2, residual_threshold=1, max_trials=200)

                    print('warping')
                    image2 = warp(data[cycle, channel], model_robust.inverse, preserve_range=True).astype(data.dtype)
                    print(np.mean(image2))
                    print('mean done')
                    rotated.append(image2)
                    
            rotatedall.append(rotated)

        rotatedall = np.array(rotatedall)
        

        
        if remove_c0_DAPI == True:
            aligned = np.hstack((dapi,rotatedall))

        print(aligned.shape)
        return aligned


    @staticmethod
    def _align_stack_rigidbody(data, window=1.1, remove_c0_DAPI=False):
        print('align rigid body')
        """Rigid alignment of sequencing cycles and channels with rotation and translation. 

        Expects `data` to be an array with dimensions (CYCLE, CHANNEL, I, J).
        A centered subset of data is used if `window` is greater 
        than one. Subpixel alignment is done if `upsample_factor` is greater than
        one (can be slow).

        """
        data = np.array(data)
        if remove_c0_DAPI == True:
            dapi = data[0,0]
            data = data[0,1:]

        #print(len(data))


        #data = np.array(data)#
        print(data.shape)
        assert data.ndim == 3, 'Input data must have dimensions CHANNEL, I, J'
       
        from skimage.feature import ORB, match_descriptors
        from skimage.transform import EuclideanTransform, warp
        from skimage.measure import ransac


        orb = ORB(n_keypoints=200, fast_threshold=0.05) #https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
        
        data_window = Align.apply_window(data, window=window)
        data_window = data_window.astype('double')
        rotated = []
        for channel in range(data.shape[0]):
            print(channel)
            if (channel == 0):
                print(data_window)
                image1 = data_window[channel]
                print(image1.shape)
                print(image1.dtype)
                orb.detect_and_extract(image1)
                keypoints1 = orb.keypoints
                descriptors1 = orb.descriptors
                rotated.append(data[channel])
            else:
                image2 = data_window[channel]
                orb.detect_and_extract(image2)   
                keypoints2 = orb.keypoints
                descriptors2 = orb.descriptors
                print('match descriptors')
                matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
                src = keypoints2[matches12[:, 1]][:, ::-1]
                dst = keypoints1[matches12[:, 0]][:, ::-1]

                print('ransac')        
                model_robust, inliers = ransac((src, dst), EuclideanTransform,
                                   min_samples=2, residual_threshold=1, max_trials=200)

                print('warping')
                image2 = warp(data[channel], model_robust.inverse, preserve_range=True).astype(data.dtype)
                print(np.mean(image2))
                print('mean done')
                rotated.append(image2)
                

        rotated = np.array(rotated)
        
        
        if remove_c0_DAPI == True:
            rotated = np.hstack((dapi,rotated))
        rotated = rotated.astype(data.dtype)
        print(rotated.shape)
        print(rotated.type)
        return rotated
        # if remove_c0_DAPI == True:
        #     dapi = data[0]
        #     data = data[1:]

        # print(len(data))


        # data = np.array(data)#
        # assert data.ndim == 3, 'Input data must have dimensions CHANNEL, I, J'
       
        # from skimage.feature import ORB, match_descriptors
        # from skimage.transform import EuclideanTransform, warp
        # from skimage.measure import ransac


        # orb = ORB(n_keypoints=200, fast_threshold=0.05) #https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
        
        # data_window = Align.apply_window(data, window=window)

        # rotated = []
        # for channel in range(data.shape[1]):
        #     if (channel == 0):
        #         image1 = data_window[channel]
        #         orb.detect_and_extract(image1)
        #         keypoints1 = orb.keypoints
        #         descriptors1 = orb.descriptors
        #         rotated.append(data[cycle, channel])
        #     else:
        #         image2 = data_window[channel]
        #         orb.detect_and_extract(image2)   
        #         keypoints2 = orb.keypoints
        #         descriptors2 = orb.descriptors
        #         print('match descriptors')
        #         matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
        #         src = keypoints2[matches12[:, 1]][:, ::-1]
        #         dst = keypoints1[matches12[:, 0]][:, ::-1]

        #         print('ransac')        
        #         model_robust, inliers = ransac((src, dst), EuclideanTransform,
        #                            min_samples=2, residual_threshold=1, max_trials=200)

        #         print('warping')
        #         image2 = warp(data[channel], model_robust.inverse, preserve_range=True).astype(data.dtype)
        #         print(np.mean(image2))
        #         print('mean done')
        #         rotated.append(image2)
                

        # rotated = np.array(rotated)
        
        
        # if remove_c0_DAPI == True:
        #     rotated = np.hstack((dapi,rotated))

        # print(rotated.shape)
        # return rotated



    @staticmethod
    def _crop_files(data, nsites = 1):
        import math
        data = np.array(data)
        print(data.shape)
        stitched_size = data.shape[-1]
        cropped = []
        for i in range(int(math.sqrt(nsites))):
            for j in range(int(math.sqrt(nsites))):
                cropped.append(data[:,int(i*stitched_size/math.sqrt(nsites)):int(stitched_size/math.sqrt(nsites)*(i+1)),
                                 int(j*stitched_size/math.sqrt(nsites)):int(stitched_size/math.sqrt(nsites)*(j+1))])

        #print(cropped[0].shape)
        print(len(cropped))
        return cropped


    @staticmethod
    def _align_general(data, channel_index=0, channel_offsets=None):
        """Align data using first channel. If data is a list of stacks with different 
        IJ dimensions, the data will be piled first. Optional channel offset.
        Images are aligned to the image at `channel_index`.

        """

        # shapes might be different if stitched with different configs
        # keep shape consistent with DO
        shape = data[0].shape
        data = ops.utils.pile(data)
        data = data[..., :shape[-2], :shape[-1]]

        indices = list(range(len(data)))
        indices.pop(channel_index)
        indices_fwd = [channel_index] + indices
        indices_rev = np.argsort(indices_fwd)
        aligned = ops.process.Align.register_and_offset(data[indices_fwd], registration_images=data[indices_fwd,0])
        aligned = aligned[indices_rev]
        if channel_offsets:
            aligned = fix_channel_offsets(aligned, channel_offsets)

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
    def _align_by_DAPI_m36(data_1, data_2, channel_index=0, upsample_factor=1):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """
        images = data_1[channel_index], data_2[channel_index]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        aligned = np.stack((aligned[1],aligned[2],data_1[0],data_1[1]),axis=0) #dapi,mavs,mito,pex
        return aligned

    @staticmethod
    def _align_by_channel_downsample(data_1, data_2, channel_index1=0, channel_index2=0, upsample_factor=2, downsample_index=0, upsample_index=1):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """

        # add new axis to single-channel images
        if data_1.ndim == 2:
            data_1 = data_1[np.newaxis,:]
        if data_2.ndim == 2:
            data_2 = data_2[np.newaxis,:]



        print('aligning')

        images = [data_1[channel_index1], data_2[channel_index2]]
        print('resizing')
        print(images[downsample_index].shape,images[upsample_index].shape)
        images[downsample_index] = skimage.transform.resize(images[downsample_index], images[upsample_index].shape,
                        anti_aliasing=True, mode = 'constant')
        
        print(images[downsample_index].shape,images[upsample_index].shape)

        images = tuple(images)

        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        print(images[downsample_index][np.newaxis,:].shape)
        print(aligned.shape)
        aligned = np.vstack((images[downsample_index][np.newaxis,:], aligned[1:]))
        return aligned

    @staticmethod
    def _align_by_channel(data_1, data_2, channel_index1=0, channel_index2=0, upsample_factor=1, data_1_keepchannels=None, data_2_keepchannels=None):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.
        """

        # add new axis to single-channel images
        if data_1.ndim == 2:
            data_1 = data_1[np.newaxis,:]
        if data_2.ndim == 2:
            data_2 = data_2[np.newaxis,:]

        print('aligning')
        images = data_1[channel_index1], data_2[channel_index2]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)

        if (data_1_keepchannels == None) & (data_2_keepchannels != None):
            aligned = aligned[data_2_keepchannels,:]
            if aligned.ndim == 2:
                aligned = aligned[np.newaxis,:]
            print(data_1.shape)
            print(aligned.shape)

        elif (data_1_keepchannels != None) & (data_2_keepchannels == None):
            data_1 =data_1[data_1_keepchannels,:]
            if data_1.ndim == 2:
                data_1 = data_1[np.newaxis,:]
#        elif (data_1_keepchannels == None) & (data_2_keepchannels == None):

        else:
            data_1 =data_1[data_1_keepchannels,:]
            aligned = aligned[data_2_keepchannels,:]
            if aligned.ndim == 2:
                aligned = aligned[np.newaxis,:]
            if data_1.ndim == 2:
                data_1 = data_1[np.newaxis,:]

        aligned = np.vstack((data_1, aligned))
        print(aligned.shape)
        return aligned

    @staticmethod
    def _align_by_channel_3ch(data_1, data_2, data_3, channel_index1=0, channel_index2=0, 
        channel_index3=0, channel_index4=0, upsample_factor=1):
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
 
        print('aligning')
        images = data_1[channel_index1], data_2[channel_index2]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned2 = ops.process.Align.apply_offsets(data_2, offsets)

        images = data_1[channel_index1], data_3[channel_index3]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_3)
        aligned3 = ops.process.Align.apply_offsets(data_3, offsets)

        aligned = np.vstack((data_1, aligned2, aligned3))
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

    # 
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max, max = None, channel = None):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J).
        """
        #print('99th percentile nuc: ' + str(np.percentile(data[0],99)))
        print(data.shape)
        if max != None:
            if max == 'special':
                z = data.shape[0]
                data = data.reshape(-1,1024,1024).reshape(5,z,1024,1024).max(axis=1)     
            else:
                data = data.max(axis=1)

        if np.percentile(data[0],99) < 700:
            print('no nuclei')
            print(np.max(data[0]))
            return np.zeros_like(data[0])
        
        if channel != None:
            if isinstance(data, list):
                dapi = data[channel]
            elif data.ndim == 3:
                dapi = data[channel]
            elif data.ndim == 4:
                dapi = data[1][channel]
            print(dapi)
            kwargs = dict(threshold=lambda x: threshold, 
                area_min=area_min, area_max=area_max)

            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nuclei = ops.process.find_nuclei(dapi, **kwargs)
            return nuclei.astype(np.uint16)

        else:
            if isinstance(data, list):
                dapi = data[0]
            elif data.ndim == 3:
                dapi = data[0]
            elif data.ndim == 4:
                dapi = data[1][0]
            kwargs = dict(threshold=lambda x: threshold, 
                area_min=area_min, area_max=area_max)

            print('type, ndim:')
            print(data.dtype)
            print(data.ndim)

            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nuclei = ops.process.find_nuclei(dapi, **kwargs)
            return nuclei.astype(np.uint16)


    @staticmethod
    def _segment_nuclei_nondapi(data, threshold, area_min, area_max, channel):
        """Find nuclei from channel specified by channel argument (not DAPI). Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J).
        """
        #print('99th percentile nuc: ' + str(np.percentile(data[0],99)))
        if np.percentile(data[channel],99.999999999) < 700:
            print('no nuclei')
            print(np.max(data[channel]))
            return data[channel]
        else:
            if isinstance(data, list):
                dapi = data[channel]
            elif data.ndim == 3:
                dapi = data[channel]
            elif data.ndim == 4:
                dapi = data[1][channel]
            print(dapi)
            kwargs = dict(threshold=lambda x: threshold, 
                area_min=area_min, area_max=area_max)

            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nuclei = ops.process.find_nuclei(dapi, **kwargs)
            return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_masked(data, threshold, area_min, area_max, mask, threshold_mask):
        """Find nuclei but ignore values in provided mask >= 'threshold_mask'. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape (CHANNEL, I, J)."""
        mask = mask >= threshold_mask
        #print('99th percentile nuc: ' + str(np.percentile(data[0],99)))
        if np.percentile(data[0],99) < 700:
            print('no nuclei')
            print(np.max(data[0]))
            return data[0]
        else:
            if isinstance(data, list):
                dapi = data[0]
            elif data.ndim == 3:
                dapi = data[0]
            elif data.ndim == 4:
                dapi = data[1][0]
            #print(dapi)
            kwargs = dict(threshold=lambda x: threshold, 
                area_min=area_min, area_max=area_max)
 
        #print(sum(sum(dapi)))

        

    # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        
        print(sum(sum(nuclei)))

        nuclei[mask==True] = 0
        print(sum(sum(nuclei)))

        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_cells(data, nuclei, threshold, chstart, chend, max = None, cycle=None):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        """
        print(data.ndim)
        if max != None:
            if max == 'special':
                z = data.shape[0]
                data = data.reshape(-1,1024,1024).reshape(5,z,1024,1024).max(axis=1)     
            else:
                data = data.max(axis=1)
        print(data.shape)
        if data.ndim == 4:
            print(threshold, chstart, chend)
            # no DAPI, min over cycles, mean over channels
            if cycle == None:
                mask = data[:, chstart:chend].min(axis=0).mean(axis=0)
            else:
                mask = np.median(data[cycle, chstart:chend], axis=0)
        else:
            #mask = np.median(data[1:], axis=0)
            print(threshold, chstart, chend)

            mask = np.median(data[chstart:chend], axis=0)

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



    ##### BECCA ADDITIONAL

    @staticmethod
    def _segment_cells_constrainedpixeldistance(data, nuclei, max_pixel_distance, threshold, chstart, chend):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.
        """
        print(data.ndim)
        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, chstart:chend].min(axis=0).mean(axis=0)
            #mask = data[:, 1:].min(axis=0).mean(axis=0)
        else:
            #mask = np.median(data[1:], axis=0)

            mask = np.median(data[chstart:chend], axis=0)

        distance = ndimage.distance_transform_cdt(nuclei == 0)

        mask = (mask > threshold) & (distance <= max_pixel_distance)

        print(mask)


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
    def _find_cytoplasm(nuclei, cells):

        mask = (nuclei == 0) & (cells > 0)
        
        cytoplasm = cells.copy()
        cytoplasm[mask == False] = 0

        return cytoplasm

    @staticmethod
    def _find_ecm(data, cells, ecm_channel, ecm_thresh):
        print(data[0,ecm_channel].shape)
        print(cells.shape)
        print(sum(sum(data[0,ecm_channel] > ecm_thresh)))
        print(sum(sum(cells == 0)))

        mask = (data[0,ecm_channel] > ecm_thresh) & (cells == 0)
        print(mask.shape)
        ecm = data[0,ecm_channel].copy()
        print(sum(sum(mask)))
        ecm[mask == False] = 0
        ecm[mask == True] = 1

        return ecm

    # @staticmethod
    # def _find_tissue(data, channel, thresh):
    #     print(data[0,channel].shape)
    #     print(cells.shape)
    #     print(sum(sum(data[0,ecm_channel] > ecm_thresh)))
    #     print(sum(sum(cells == 0)))

    #     mask = (data[0,ecm_channel] > ecm_thresh) & (cells == 0)
    #     print(mask.shape)
    #     ecm = data[0,ecm_channel].copy()
    #     print(sum(sum(mask)))
    #     ecm[mask == False] = 0
    #     ecm[mask == True] = 1

    #     return ecm


    @staticmethod
    def _find_multichannel_cellmask(cells, cytoplasm):

        mask = np.isin(cells, np.unique(cytoplasm))
        cells[mask == False] = 0

        return cells
    



    #################
    @staticmethod
    def _extract_phenotype_general_withcorr_cyto(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,corrchannel):
        

        def correlate_channel_corrchannel(region):

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

        features_nuclear = {
            'channel_corrch_nuclear_corr' : correlate_channel_corrchannel,
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'cell'               : lambda r: r.label
        }

        features_cell = {
            'channel_corrch_cell_corr' : correlate_channel_corrchannel,
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'cell'            : lambda r: r.label
        }

        features_cytoplasm = {
            'channel_corrch_cyto_corr' : correlate_channel_corrchannel,
            'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
            'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
            'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
            'channel_cyto_max'    : lambda r: masked(r, channel).max(),
            'dapi_cyto_median'    : lambda r: np.median(masked(r, 0)),
            'area_cyto'       : lambda r: r.area,
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

        
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
                .reset_index())
        df = df.loc[:, ~df.columns.duplicated()]

  
       
        
        return df
    
    #########

    @staticmethod
    def _extract_phenotype_general_withcorr_multicycle(data_phenotype, nuclei, cells, wildcards, channel, corrchannel, cycles):
        
        print(cycles)
        def correlate_channel_corrchannel(region):

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

        features_nuclear = {
            'channel_corrch_nuclear_corr' : correlate_channel_corrchannel,
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'cell'               : lambda r: r.label
        }

        features_cell = {
            'channel_corrch_cell_corr' : correlate_channel_corrchannel,
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'cell'            : lambda r: r.label
        }
        
        
        dfs = []
        for cycle in range(cycles):
            print('cycle: ', cycle)
            df_n =  Snake._extract_features(data_phenotype[cycle,:], nuclei, wildcards, features_nuclear)
            df_c =  Snake._extract_features(data_phenotype[cycle,:], cells, wildcards, features_cell) 

            
            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())
            df = df.loc[:, ~df.columns.duplicated()]
            print(df.columns)
            df.columns = [str(col) + '_c' + str(cycle) for col in df.columns]

            print(df.head())
            print(df.shape)
            dfs.append(df)
        
        dfs = pd.concat(dfs,axis=1)
        print(dfs.shape)
        return dfs

    #################
    @staticmethod
    def _extract_phenotype_general_withcorr(data_phenotype, nuclei, cells, wildcards,channel,corrchannel):
        

        def correlate_channel_corrchannel(region):

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

        features_nuclear = {
            'channel_corrch_nuclear_corr' : correlate_channel_corrchannel,
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'cell'               : lambda r: r.label
        }

        features_cell = {
            'channel_corrch_cell_corr' : correlate_channel_corrchannel,
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'cell'            : lambda r: r.label
        }
        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

        
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())
        df = df.loc[:, ~df.columns.duplicated()]

  
       
        
        return df

    #################
    @staticmethod
    def _extract_phenotype_general_peaksmask(data_phenotype, nuclei, cells, peaks, channel, peaksthresh, wildcards):
        
        
        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

        features_nuclear = {
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'cell'               : lambda r: r.label
        }

        features_cell = {
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'cell'            : lambda r: r.label
        }
        

        mask = (peaks > peaksthresh) 
        print(mask.shape, data_phenotype[channel].shape, np.mean(data_phenotype[channel]))
        data_phenotype[channel][mask == False] = 0
        print(mask.shape, data_phenotype[channel].shape, np.mean(data_phenotype[channel]))

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

        
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())
        df = df.loc[:, ~df.columns.duplicated()]

  
       
        
        return df
    

    @staticmethod
    def _extract_phenotype_extended(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,corrchannel1=None,
         corrchannel2=None, corrchannel3=None, corrchannel4=None, corrchannel5=None, granularity = False, skeleton = False):
 

        def correlate_channel_corrchannel1(region):

            if corrchannel1 == None:
                return 0
            else:

                corrch = region.intensity_image_full[corrchannel1]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
    
        def correlate_channel_corrchannel2(region):

            if corrchannel2 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel2]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel3(region):

            if corrchannel3 == None:
                return 0
            else:

                corrch = region.intensity_image_full[corrchannel3]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
    
        def correlate_channel_corrchannel4(region):


            if corrchannel4 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel4]
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

        def mahotas_zernike(region):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            # names = ['zernike' + str(c) for c in range(1,31)]
            # t = [(names[i],mfeat[i]) for i in range(0,30)]
            # feat_dict = dict((x, y) for x, y in t)
            return mfeat

        def mahotas_pftas(region):
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat



        features_nuclear = {
            'channel_corrch1_nuclear_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_nuclear_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_nuclear_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_nuclear_corr' : correlate_channel_corrchannel4,
            'dapi_nuclear_median': lambda r: np.median(masked(r, 0)),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'dapi_nuclear_int'   : lambda r: masked(r, 0).sum(),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'dapi_nuclear_max'   : lambda r: masked(r, 0).max(),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'area_nuclear'       : lambda r: r.area,
            'perimeter_nuclear' : lambda r: r.perimeter,
            'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
            'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
            'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
            'orientation_nuclear' : lambda r: r.orientation,
            'zernike_features_nuclear': mahotas_zernike,
            'hu_moments_nuclear': lambda r: r.moments_hu,
            'solidity_nuclear': lambda r: r.solidity,
            'extent_nuclear': lambda r: r.extent,
            'pftas_features_nuclear': mahotas_pftas,
            'dapi_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'dapi_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'dapi_nuclear_mean': lambda r: masked(r, 0).mean(),
            'dapi_nuclear_25': lambda r: np.percentile(masked(r, 0),25),
            'dapi_nuclear_75': lambda r: np.percentile(masked(r, 0),75),
            'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
            'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        

        }


        features_cell = {
            'channel_corrch1_cell_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cell_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cell_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cell_corr' : correlate_channel_corrchannel4,
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'dapi_cell_median'    : lambda r: np.median(masked(r, 0)),
            'area_cell'       : lambda r: r.area,
            'perimeter_cell' : lambda r: r.perimeter,
            'cell'            : lambda r: r.label,
            'euler_cell' : lambda r: r.euler_number,
            'eccentricity_cell' : lambda r: r.eccentricity, #cell
            'major_axis_cell' : lambda r: r.major_axis_length, #cell
            'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
            'orientation_cell' : lambda r: r.orientation,
            'zernike_features_cell': mahotas_zernike,
            'hu_moments_cell': lambda r: r.moments_hu,
            'solidity_cell': lambda r: r.solidity,
            'extent_cell': lambda r: r.extent,
            'pftas_features_cell': mahotas_pftas,
            'channel_cell_min': lambda r: np.min(masked(r,channel)),
            'channel_cell_sd': lambda r: np.std(masked(r,channel)),
            'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }

        features_cytoplasm = {
            'channel_corrch1_cyto_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cyto_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cyto_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cyto_corr' : correlate_channel_corrchannel4,
            'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
            'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
            'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
            'channel_cyto_max'    : lambda r: masked(r, channel).max(),
            'dapi_cyto_median'    : lambda r: np.median(masked(r, 0)),
            'area_cyto'       : lambda r: r.area,
            'perimeter_cyto' : lambda r: r.perimeter,
            'euler_cyto' : lambda r: r.euler_number,
            'channel_cyto_sd': lambda r: np.std(masked(r,channel)),
            'channel_cyto_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cyto_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cyto_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

        ## calculate radius distances, edge intensity values, mass displacement
        # dist = []
        # for i in np.unique(cells):
        #     if i == 0:
        #         continue
        #     if (i % 50 == 0):
        #         print(i)
        #     mask = cells == i
        #     tmp = data_phenotype[channel].copy()
        #     tmp[mask == False] = 0
        #     dists = distance_transform_edt(tmp, sampling=None, return_distances=True)
            
        #     tmp[dists != 1] = 0 #only consider perimeter intensities
        #     integ_inten = np.sum(tmp)
        #     mean_inten = np.average(tmp[tmp!=0])
        #     sd_inten = np.std(tmp[tmp!=0])
        #     max_inten = np.max(tmp[tmp!=0])
        #     min_inten = np.min(tmp[tmp!=0])
            
        #     max_rad = np.max(dists)
        #     mean_rad = np.mean(dists)
        #     median_rad = np.median(dists)
        #     rad_std = np.std(dists)

        #     comgray = ndimage.measurements.center_of_mass(tmp)
        #     binary = tmp.copy()
        #     binary[binary!=0] = 1
        #     combin = ndimage.measurements.center_of_mass(binary) #different from skimage centroid (i,j) in resulting .pkl
        #     euclid_dist_cell = ((comgray[0] - combin[0]) ** 2 +
        #                 (comgray[1] - combin[1]) ** 2) ** 0.5

        #     tmp_msd = []
        #     nonzero = np.nonzero(tmp) #inds of nonzero objects
        #     for x in range(len(nonzero[0])):
        #         tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
        #     msd_cell = np.mean(tmp_msd)

        #     mask = nuclei == i
        #     tmp = data_phenotype[channel].copy()
        #     tmp[mask == False] = 0
        #     comgray = ndimage.measurements.center_of_mass(tmp)
        #     binary = tmp.copy()
        #     binary[binary!=0] = 1
        #     combin = ndimage.measurements.center_of_mass(binary) #technically no need to recalc, this is just skimage centroid
        #     euclid_dist_nuc = ((comgray[0] - combin[0]) ** 2 +
        #                 (comgray[1] - combin[1]) ** 2) ** 0.5

        #     tmp_msd = []
        #     nonzero = np.nonzero(tmp) #inds of nonzero objects
        #     for x in range(len(nonzero[0])):
        #         tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
        #     msd_nuc = np.mean(tmp_msd)

        #     dist.append((i, max_rad, mean_rad, rad_std, integ_inten, mean_inten, sd_inten, max_inten, min_inten, 
        #         euclid_dist_cell, euclid_dist_nuc, msd_cell, msd_nuc))

        #     df_radii = pd.DataFrame(dist, columns = ['cell', 'max_rad','mean_rad', 'rad_std' 
        #                                             'integrated_edge_intensity', 'mean_edge_intensity', 'edge_intensity_SD',
        #                                     'max_edge_intensity', 'min_edge_intensity', 'euclid_dist_cell', 'euclid_dist_nuc'
        #                                     'msd_cell', 'msd_nuc'])
   


            # # adapted from CP: https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/measuregranularity.py
            # # Downsample the image and mask
            # #
            # subsample_size = .25 #decrease for higher-res images
            # sample_size = .125 #usually .125-.25
            # radius = 1 #must be at least 1.
            # # This radius should correspond to the radius of the textures of interest
            # # *after* subsampling; i.e., if textures in the original image scale have
            # # a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
            # # structuring element size should be 10 or slightly smaller, and the range
            # # of the spectrum defined below will cover more sizes."""))
            # granular_spectrum_length = 8
            # # You may need a trial run to see which granular
            # # spectrum range yields informative measurements. Start by using a wide spectrum and
            # # narrow it down to the informative range to save time."""))
            # ##

            # new_shape = np.array(data_phenotype[channel].shape)

            # if subsample_size < 1:
            #     new_shape = new_shape * subsample_size
            #     i, j = (np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / subsample_size)
            #     pixels = ndimage.map_coordinates(data_phenotype[channel], (i, j), order=1)
            #     small_cells = ndimage.map_coordinates(cells, (i, j), order=1)
            #     small_cyto = ndimage.map_coordinates(cytoplasm, (i, j), order=1)
            #     small_nuc = ndimage.map_coordinates(nuclei, (i, j), order=1)

            # else:
            #     pixels = data_phenotype[channel]
            #     small_cells = cells
            #     small_nuc = nuclei
            # # Remove background pixels using a greyscale tophat filter
            # #
            # if sample_size < 1:
            #     back_shape = new_shape * sample_size
            #     i, j = (np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / sample_size)
            #     back_pixels = ndimage.map_coordinates(pixels, (i, j), order=1)
            #     back_mask = ndimage.map_coordinates(mask.astype(float), (i, j)) > .9
            # else:
            #     back_pixels = pixels
            #     back_mask = mask
            #     back_shape = new_shape
                


            # selem = skimage.morphology.disk(radius, dtype=bool)
                
            # back_pixels_mask = np.zeros_like(back_pixels)
            # back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
            # back_pixels = skimage.morphology.erosion(back_pixels_mask, selem=selem)
            # back_pixels_mask = np.zeros_like(back_pixels)
            # back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
            # back_pixels = skimage.morphology.dilation(back_pixels_mask, selem=selem)

            # if sample_size < 1:
            #     i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float)
            #     #
            #     # Make sure the mapping only references the index range of
            #     # back_pixels.
            #     #
            #     i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            #     j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            #     back_pixels = ndimage.map_coordinates(back_pixels, (i, j), order=1)
            # pixels -= back_pixels
            # pixels[pixels < 0] = 0

            # ng = granular_spectrum_length

            # all_measurements_cell = []
            # all_measurements_nuc = []

            # for i in np.unique(cells):
            #     if i == 0:
            #         continue
            #     mask = small_cells == i
            #     startmean = np.mean(pixels[mask])
            #     ero = pixels.copy()
            #     # Mask the test image so that masked pixels will have no effect
            #     # during reconstruction
            #     #
            #     ero[~mask] = 0
            #     currentmean = startmean
            #     startmean = max(startmean, np.finfo(float).eps)

            #     footprint = skimage.morphology.disk(1, dtype=bool)
            #     measurements = []
            #     measurements.append(i)
            #     for x in range(1, ng + 1):
            #         prevmean = currentmean
            #         ero_mask = np.zeros_like(ero)
            #         ero_mask[mask == True] = ero[mask == True]
            #         ero = skimage.morphology.erosion(ero_mask, selem=footprint)
            #         rec = skimage.morphology.reconstruction(ero, pixels, selem=footprint)
            #         currentmean = np.mean(rec[mask])
            #         gs = (prevmean - currentmean) * 100 / startmean
            #         measurements.append(gs)
            #     all_measurements_cell.append(measurements)
            #         #
            #     mask2 = small_nuc == i
            #     startmean = np.mean(pixels[mask2])
            #     ero = pixels.copy()
            #     # Mask the test image so that masked pixels will have no effect
            #     # during reconstruction
            #     #
            #     ero[~mask2] = 0
            #     currentmean = startmean
            #     startmean = max(startmean, np.finfo(float).eps)

            #     footprint = skimage.morphology.disk(1, dtype=bool)
            #     measurements = []
            #     measurements.append(i)
            #     for x in range(1, ng + 1):
            #         prevmean = currentmean
            #         ero_mask = np.zeros_like(ero)
            #         ero_mask[mask == True] = ero[mask == True]
            #         ero = skimage.morphology.erosion(ero_mask, selem=footprint)
            #         rec = skimage.morphology.reconstruction(ero, pixels, selem=footprint)
            #         currentmean = np.mean(rec[mask])
            #         gs = (prevmean - currentmean) * 100 / startmean
            #         measurements.append(gs)
            #     all_measurements_nuc.append(measurements)


            #     df_gran_cell = pd.DataFrame(all_measurements_cell, columns = ['cell', 'cell_granularity_1', 'cell_granularity_2', 
            #         'cell_granularity_3', 'cell_granularity_4', 'cell_granularity_5', 
            #         'cell_granularity_6', 'cell_granularity_7', 'cell_granularity_8'])
 
            #     df_gran_nuc = pd.DataFrame(all_measurements_nuc, columns = ['cell', 'nuc_granularity_1', 'nuc_granularity_2', 
            #         'nuc_granularity_3', 'nuc_granularity_4', 'nuc_granularity_5', 
            #         'nuc_granularity_6', 'nuc_granularity_7', 'nuc_granularity_8'])

        if skeleton == True:
            from skan.pre import threshold
            from skimage.morphology import skeletonize
            from skan import csr
            from itertools import compress


            spacing_nm = 15#change as needed
            smooth_radius = 5 / spacing_nm  # float OK
            threshold_radius = int(np.ceil(50 / spacing_nm))
            binary0 = threshold(data_phenotype[channel], sigma=smooth_radius,
                                 radius=threshold_radius)


            vals_all = []
            for i in np.unique(cytoplasm):
                if i == 0:
                    continue
                if (i % 50 == 0):
                    print(i)
                
                mask = cytoplasm == i

                tmp = binary0.copy()
                tmp[mask == False] = 0
                skeleton0 = skeletonize(tmp)

                table = csr.summarise(skeleton0, spacing=spacing_nm)
                n_branches = table.groupby(['branch-type'])['branch-type'].count()
                med_dist = table.groupby(['branch-type'])['branch-distance'].median()

                dic = dict(n_branches)
                dic2 = dict(med_dist)
                branch_types = [0,1,2,3] 
                ind = [s not in list(dic.keys()) for s in branch_types]
    
                for z in range(len(list(compress(branch_types,ind)))):
                    dic.update({(list(compress(branch_types,ind)))[z]: 0})
                    dic2.update({(list(compress(branch_types,ind)))[z]: 0})
                    sort = sorted(dic.items(), key=lambda kv: kv[0])
                    sort2 = sorted(dic2.items(), key=lambda kv: kv[0])
                    n_branches_ls = [s[1] for s in sort]
                    med_dist_ls = [s[1] for s in sort2]
                
                vals = [[i],n_branches_ls, med_dist_ls]
                vals_all.append([v for sublist in vals for v in sublist])
                df_skeleton = pd.DataFrame(vals_all, columns = ['cell', 'n_endpt_to_endpt_branches', 'n_junc_to_endpt_branches', 'n_junc_to_junc_branches',
             'n_isolated_cycle','ee_branch_med_dist', 'je_branch_med_dist', 'jj_branch_med_dist','is_cyc_med_dist'])

            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell'), df_radii.set_index('cell'),
                df_skeleton.set_index('cell'), df_gran_cell.set_index('cell'), df_gran_nuc.set_index('nuc')], axis=1, join='inner')
                .reset_index())

        
        else: 
            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
                .reset_index())

       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    #########



    @staticmethod
    def _extract_phenotype_ecm(data_phenotype, ecm, wildcards, channel):

        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

        def mahotas_zernike(region):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            # names = ['zernike' + str(c) for c in range(1,31)]
            # t = [(names[i],mfeat[i]) for i in range(0,30)]
            # feat_dict = dict((x, y) for x, y in t)
            return mfeat

        def mahotas_pftas(region):
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat


        features_ecm = {
            'channel_ecm_median' : lambda r: np.median(masked(r, channel)),
            'channel_ecm_mean' : lambda r: masked(r, channel).mean(),
            'channel_ecm_int'    : lambda r: masked(r, channel).sum(),
            'channel_ecm_max'    : lambda r: masked(r, channel).max(),
            'zernike_features_ecm': mahotas_zernike,
            'hu_moments_ecm': lambda r: r.moments_hu,
            'pftas_features_ecm': mahotas_pftas,
            'channel_ecm_min': lambda r: np.min(masked(r,channel)),
            'channel_ecm_sd': lambda r: np.std(masked(r,channel)),
            'channel_ecm_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_ecm_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_ecm_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }

        df =  Snake._extract_features(data_phenotype, ecm, wildcards, features_ecm)
        
        # else: 
        #     df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
        #         .reset_index())

       
        # df = df.loc[:, ~df.columns.duplicated()]
        
        return df

        ####

    # @staticmethod
    # def _extract_phenotype_extended_channel(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,corrchannel1,
    #      corrchannel2, corrchannel3, corrchannel4, corrchannel5, corrchannel6):
 

    #     def correlate_channel_corrchannel1(region, channel, corrchannel):
    #         if corrchannel == None:
    #             return 0
    #         corrch = region.intensity_image_full[corrchannel]
    #         refch = region.intensity_image_full[channel]

    #         filt = corrch > 0
    #         if filt.sum() == 0:
    #             # assert False
    #             return np.nan

    #         corrch = corrch[filt]
    #         refch  = refch[filt]
    #         corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

    #         return corr.mean()    
    
    #     def masked(region, index):
    #         return region.intensity_image_full[index][region.filled_image]

    #     def mahotas_zernike(region, channel):
    #         mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
    #         return mfeat

    #     def mahotas_pftas(region, channel):
    #         mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
    #         ### according to this, at least as good as haralick/zernike and much faster:
    #         ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
    #         return mfeat



    #     features_nuclear = {
    #         'channel_corrch1_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel1),
    #         'channel_corrch2_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel2),
    #         'channel_corrch3_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel3),
    #         'channel_corrch4_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel4),   
    #         'channel_corrch5_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel5),
    #         'channel_corrch6_nuclear_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel6),           
    #         'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
    #         'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
    #         'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
    #         'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
    #         'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
    #         'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
    #         'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
    #         'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
    #         'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
    #         'channel_zernike_nuclear': lambda r: mahotas_zernike(r, channel),
    #         'channel_pftas_nuclear': lambda r: mahotas_pftas(r, channel),
    #         'cell'               : lambda r: r.label

    #     }

    #     features_cell = {
    #         'channel_corrch1_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel1),
    #         'channel_corrch2_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel2),
    #         'channel_corrch3_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel3),
    #         'channel_corrch4_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel4),   
    #         'channel_corrch5_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel5),
    #         'channel_corrch6_cell_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel6),
    #         'channel_cell_mean' : lambda r: masked(r, channel).mean(),
    #         'channel_cell_median' : lambda r: np.median(masked(r, channel)),
    #         'channel_cell_int'    : lambda r: masked(r, channel).sum(),
    #         'channel_cell_min': lambda r: np.min(masked(r,channel)),
    #         'channel_cell_max'    : lambda r: masked(r, channel).max(),
    #         'channel_cell_sd': lambda r: np.std(masked(r,channel)),
    #         'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
    #         'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
    #         'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
    #         'channel_zernike_cell': lambda r: mahotas_zernike(r, channel),
    #         'channel_pftas_cell': lambda r: mahotas_pftas(r, channel),
    #         'cell'               : lambda r: r.label
    #     }

    #     features_cytoplasm = {
    #         'channel_corrch1_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel1),
    #         'channel_corrch2_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel2),
    #         'channel_corrch3_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel3),
    #         'channel_corrch4_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel4),   
    #         'channel_corrch5_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel5),
    #         'channel_corrch6_cyto_corr' : lambda r: correlate_channel_corrchannel1(r, channel, corrchannel6),
    #         'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
    #         'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
    #         'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
    #         'channel_cyto_min'    : lambda r: masked(r, channel).min(),
    #         'channel_cyto_max'    : lambda r: masked(r, channel).max(),
    #         'channel_cyto_sd': lambda r: np.std(masked(r,channel)),
    #         'channel_cyto_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
    #         'channel_cyto_25': lambda r: np.percentile(masked(r, channel),25),
    #         'channel_cyto_75': lambda r: np.percentile(masked(r, channel),75),
    #         'channel_zernike_cyto': lambda r: mahotas_zernike(r, channel),
    #         'channel_pftas_cyto': lambda r: mahotas_pftas(r, channel),
    #         'cell'            : lambda r: r.label
    #     }

    #     df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
    #     df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
    #     df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 
        
    #     df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
    #         .reset_index())

       
    #     df = df.loc[:, ~df.columns.duplicated()]
        
    #     return df


    @staticmethod
    def _extract_phenotype_extended_channel(data_phenotype, nuclei, cells, channel, wildcards, max = None, cytoplasm=None, corrchannel1=None,
         corrchannel2=None, corrchannel3=None, corrchannel4=None, corrchannel5=None, corrchannel6=None):

        data_phenotype = data_phenotype.astype('uint16') 
        print(data_phenotype.shape)

        if max != None:
            if max == 'special':
                z = data_phenotype.shape[0]
                data_phenotype = data_phenotype.reshape(-1,1024,1024).reshape(5,z,1024,1024).max(axis=1)   
         
            else:
                data_phenotype = data_phenotype.max(axis=1)

        nuclei = nuclei.astype('uint16') 
        cells = cells.astype('uint16') 
        print(data_phenotype[channel].shape,nuclei.shape,cells.shape)

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
        print(data_phenotype.shape, nuclei.astype('uint16').shape)
        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        if cytoplasm != None:
            df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 
        
        
            if 'cell' not in list(set(df_n.columns) & set(df_c.columns) & set(df_cyto.columns)):
                print('no cells')
                return None
           
            else:
                df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')], axis=1, join='inner')
                    .reset_index())

        else:
            if 'cell' not in list(set(df_n.columns) & set(df_c.columns)):
                print('no cells')
                return None
            else:
                df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

        df = df.loc[:, ~df.columns.duplicated()]
    
        return df

    #########

    @staticmethod
    def _extract_phenotype_extended_channel_noclassical(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,corrchannel1,
         corrchannel2, corrchannel3, corrchannel4, skeleton = False):

        def correlate_channel_corrchannel1(region):

            corrch = region.intensity_image_full[corrchannel1]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    
    
        def correlate_channel_corrchannel2(region):

            corrch = region.intensity_image_full[corrchannel2]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    

        def correlate_channel_corrchannel3(region):

            corrch = region.intensity_image_full[corrchannel3]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    
    
        def correlate_channel_corrchannel4(region):

            corrch = region.intensity_image_full[corrchannel4]
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

        def mahotas_zernike(region):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            return mfeat

        def mahotas_pftas(region):
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel])
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat


        features_nuclear = {
            'channel_zernike_nuclear': mahotas_zernike,
            'channel_pftas_nuclear': mahotas_pftas,
            'cell'               : lambda r: r.label
        
        }


        features_cell = {
            'channel_zernike_cell': mahotas_zernike,
            'channel_pftas_cell': mahotas_pftas,
            'cell'               : lambda r: r.label
        }

        features_cytoplasm = {
            'channel_corrch1_cyto_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cyto_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cyto_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cyto_corr' : correlate_channel_corrchannel4,
            'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
            'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
            'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
            'channel_cyto_min'    : lambda r: masked(r, channel).min(),
            'channel_cyto_max'    : lambda r: masked(r, channel).max(),
            'channel_cyto_sd': lambda r: np.std(masked(r,channel)),
            'channel_cyto_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cyto_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cyto_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 


        # # adapted from CP: https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/measuregranularity.py
        # # Downsample the image and mask
        # #
        # subsample_size = .25 #decrease for higher-res images
        # sample_size = .125 #usually .125-.25
        # radius = 2 #must be at least 1.
        # # This radius should correspond to the radius of the textures of interest
        # # *after* subsampling; i.e., if textures in the original image scale have
        # # a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
        # # structuring element size should be 10 or slightly smaller, and the range
        # # of the spectrum defined below will cover more sizes."""))
        # granular_spectrum_length = 4
        # # You may need a trial run to see which granular
        # # spectrum range yields informative measurements. Start by using a wide spectrum and
        # # narrow it down to the informative range to save time."""))
        # ##

        # new_shape = np.array(data_phenotype[channel].shape)

        # if subsample_size < 1:
        #     new_shape = new_shape * subsample_size
        #     i, j = (np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / subsample_size)
        #     pixels = ndimage.map_coordinates(data_phenotype[channel], (i, j), order=1)
        #     small_cells = ndimage.map_coordinates(cells, (i, j), order=1)
        #     small_cyto = ndimage.map_coordinates(cytoplasm, (i, j), order=1)
        #     small_nuc = ndimage.map_coordinates(nuclei, (i, j), order=1)

        # else:
        #     pixels = data_phenotype[channel]
        #     small_cells = cells
        #     small_nuc = nuclei
        # # Remove background pixels using a greyscale tophat filter
        # #
        # if sample_size < 1:
        #     back_shape = new_shape * sample_size
        #     i, j = (np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / sample_size)
        #     back_pixels = ndimage.map_coordinates(pixels, (i, j), order=1)
        #     back_mask = ndimage.map_coordinates(small_cells.astype(float), (i, j)) > .9
        # else:
        #     back_pixels = pixels
        #     back_mask = mask
        #     back_shape = new_shape
            


        # selem = skimage.morphology.disk(radius, dtype=bool)
            
        # back_pixels_mask = np.zeros_like(back_pixels)
        # back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
        # back_pixels = skimage.morphology.erosion(back_pixels_mask, selem=selem)
        # back_pixels_mask = np.zeros_like(back_pixels)
        # back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
        # back_pixels = skimage.morphology.dilation(back_pixels_mask, selem=selem)

        # if sample_size < 1:
        #     i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float)
        #     #
        #     # Make sure the mapping only references the index range of
        #     # back_pixels.
        #     #
        #     i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        #     j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        #     back_pixels = ndimage.map_coordinates(back_pixels, (i, j), order=1)
        # pixels -= back_pixels
        # pixels[pixels < 0] = 0

        # ng = granular_spectrum_length

        # all_measurements_cell = []
        # all_measurements_nuc = []

        # for i in np.unique(cells):
        #     if i == 0:
        #         continue
        #     if (i % 50 == 0):
        #         print(i)
        #     mask = small_cells == i
        #     startmean = np.mean(pixels[mask])
        #     ero = pixels.copy()
        #     # Mask the test image so that masked pixels will have no effect
        #     # during reconstruction
        #     #
        #     ero[~mask] = 0
        #     currentmean = startmean
        #     startmean = max(startmean, np.finfo(float).eps)

        #     footprint = skimage.morphology.disk(1, dtype=bool)
        #     measurements = []
        #     measurements.append(i)
        #     for x in range(1, ng + 1):
        #         prevmean = currentmean
        #         ero_mask = np.zeros_like(ero)
        #         ero_mask[mask == True] = ero[mask == True]
        #         ero = skimage.morphology.erosion(ero_mask, selem=footprint)
        #         rec = skimage.morphology.reconstruction(ero, pixels, selem=footprint)
        #         currentmean = np.mean(rec[mask])
        #         gs = (prevmean - currentmean) * 100 / startmean
        #         measurements.append(gs)
        #     all_measurements_cell.append(measurements)
        #         #
        #     mask2 = small_nuc == i
        #     startmean = np.mean(pixels[mask2])
        #     ero = pixels.copy()
        #     # Mask the test image so that masked pixels will have no effect
        #     # during reconstruction
        #     #
        #     ero[~mask2] = 0
        #     currentmean = startmean
        #     startmean = max(startmean, np.finfo(float).eps)

        #     footprint = skimage.morphology.disk(1, dtype=bool)
        #     measurements = []
        #     measurements.append(i)
        #     for x in range(1, ng + 1):
        #         prevmean = currentmean
        #         ero_mask = np.zeros_like(ero)
        #         ero_mask[mask == True] = ero[mask == True]
        #         ero = skimage.morphology.erosion(ero_mask, selem=footprint)
        #         rec = skimage.morphology.reconstruction(ero, pixels, selem=footprint)
        #         currentmean = np.mean(rec[mask])
        #         gs = (prevmean - currentmean) * 100 / startmean
        #         measurements.append(gs)
        #     all_measurements_nuc.append(measurements)


        #     df_gran_cell = pd.DataFrame(all_measurements_cell, columns = ['cell', 'channel_cell_granularity_1', 'channel_cell_granularity_2', 
        #         'channel_cell_granularity_3', 'channel_cell_granularity_4'])

        #     df_gran_nuc = pd.DataFrame(all_measurements_nuc, columns = ['cell', 'channel_nuc_granularity_1', 'channel_nuc_granularity_2', 
        #         'channel_nuc_granularity_3', 'channel_nuc_granularity_4'])

        if skeleton == True:
            from skan.pre import threshold
            from skimage.morphology import skeletonize
            from skan import csr
            from itertools import compress


            spacing_nm = 15#change as needed
            smooth_radius = 5 / spacing_nm  # float OK
            threshold_radius = int(np.ceil(50 / spacing_nm))
            binary0 = threshold(data_phenotype[channel], sigma=smooth_radius,
                                 radius=threshold_radius)


            vals_all = []
            if len(np.unique(cytoplasm)) < 2:
                print('no cells')
                return
            for i in np.unique(cytoplasm):
                if i == 0:
                    continue
                if (i % 50 == 0):
                    print(i)
                
                mask = cytoplasm == i

                tmp = binary0.copy()
                tmp[mask == False] = 0
                skeleton0 = skeletonize(tmp)
                print(sum(sum(skeleton0)))
                if (sum(sum(skeleton0))) < 6:
                    print('no skeleton')
                    vals = [[i],[0,0,0,0],[0,0,0,0]]
                    vals_all.append([v for sublist in vals for v in sublist])

                else:
                    table = csr.summarise(skeleton0, spacing=spacing_nm)
                    n_branches = table.groupby(['branch-type'])['branch-type'].count()
                    med_dist = table.groupby(['branch-type'])['branch-distance'].median()

                    dic = dict(n_branches)
                    dic2 = dict(med_dist)
                    branch_types = [0,1,2,3] 
                    ind = [s not in list(dic.keys()) for s in branch_types]
        
                    for z in range(len(list(compress(branch_types,ind)))):
                        dic.update({(list(compress(branch_types,ind)))[z]: 0})
                        dic2.update({(list(compress(branch_types,ind)))[z]: 0})
                        sort = sorted(dic.items(), key=lambda kv: kv[0])
                        sort2 = sorted(dic2.items(), key=lambda kv: kv[0])
                        n_branches_ls = [s[1] for s in sort]
                        med_dist_ls = [s[1] for s in sort2]
                
                    vals = [[i],n_branches_ls, med_dist_ls]
                    vals_all.append([v for sublist in vals for v in sublist])
                df_skeleton = pd.DataFrame(vals_all, columns = ['cell', 'n_endpt_to_endpt_branches', 'n_junc_to_endpt_branches', 'n_junc_to_junc_branches',
             'n_isolated_cycle','ee_branch_med_dist', 'je_branch_med_dist', 'jj_branch_med_dist','is_cyc_med_dist'])

            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell'),
                df_skeleton.set_index('cell')], axis=1, join='inner')
                .reset_index())

        
        else: 
            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell') 
                ], axis=1, join='inner')
                .reset_index())

    
       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df


#########

    @staticmethod
    def _extract_phenotype_extended_channel_noclassical_7ch(data_phenotype, nuclei, cells, cytoplasm, wildcards,channel,
        corrchannel1=None, corrchannel2=None, corrchannel3=None, corrchannel4=None, 
        corrchannel5=None, corrchannel6=None, skeleton = False):

        def correlate_channel_corrchannel1(region):
            if corrchannel1 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel1]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
        
        def correlate_channel_corrchannel2(region):

            if corrchannel2 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel2]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel3(region):

            if corrchannel3 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel3]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
    
        def correlate_channel_corrchannel4(region):

            if corrchannel4 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel4]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel5(region):

            if corrchannel5 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel5]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()

        def correlate_channel_corrchannel6(region):

            if corrchannel6 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel6]
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

        def mahotas_zernike(region):
            mfeat = mahotas.features.zernike_moments(region.intensity_image_full[channel], radius = 9, degree=9)
            return mfeat

        def mahotas_pftas(region):
            # print(channel)
            # print(region.intensity_image_full[channel])
            # print(region.intensity_image_full[channel].astype('uint64'))
            mfeat = mahotas.features.pftas(region.intensity_image_full[channel].astype('uint64'))
            ### according to this, at least as good as haralick/zernike and much faster:
            ### https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-110
            return mfeat


        features_nuclear = {
            'channel_zernike_nuclear': mahotas_zernike,
            'channel_pftas_nuclear': mahotas_pftas,
            'cell'               : lambda r: r.label
        
        }


        features_cell = {
            'channel_zernike_cell': mahotas_zernike,
            'channel_pftas_cell': mahotas_pftas,
            'cell'               : lambda r: r.label
        }

        features_cytoplasm = {
            'channel_corrch1_cyto_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cyto_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cyto_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cyto_corr' : correlate_channel_corrchannel4,
            'channel_corrch5_cyto_corr' : correlate_channel_corrchannel5,
            'channel_corrch6_cyto_corr' : correlate_channel_corrchannel6,
            'channel_cyto_mean' : lambda r: masked(r, channel).mean(),
            'channel_cyto_median' : lambda r: np.median(masked(r, channel)),
            'channel_cyto_int'    : lambda r: masked(r, channel).sum(),
            'channel_cyto_min'    : lambda r: masked(r, channel).min(),
            'channel_cyto_max'    : lambda r: masked(r, channel).max(),
            'channel_cyto_sd': lambda r: np.std(masked(r,channel)),
            'channel_cyto_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cyto_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cyto_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'            : lambda r: r.label
        }

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
        df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

        print(df_cyto.columns)

        if skeleton == True:
            from skan.pre import threshold
            from skimage.morphology import skeletonize
            from skan import csr
            from itertools import compress


            spacing_nm = 15#change as needed
            smooth_radius = 5 / spacing_nm  # float OK
            threshold_radius = int(np.ceil(50 / spacing_nm))
            binary0 = threshold(data_phenotype[channel], sigma=smooth_radius,
                                 radius=threshold_radius)


            vals_all = []
            if len(np.unique(cytoplasm)) < 2:
                print('no cells')
                return
            for i in np.unique(cytoplasm):
                if i == 0:
                    continue
                if (i % 50 == 0):
                    print(i)
                
                mask = cytoplasm == i

                tmp = binary0.copy()
                tmp[mask == False] = 0
                skeleton0 = skeletonize(tmp)
                print(sum(sum(skeleton0)))
                if (sum(sum(skeleton0))) < 6:
                    print('no skeleton')
                    vals = [[i],[0,0,0,0],[0,0,0,0]]
                    vals_all.append([v for sublist in vals for v in sublist])

                else:
                    table = csr.summarise(skeleton0, spacing=spacing_nm)
                    n_branches = table.groupby(['branch-type'])['branch-type'].count()
                    med_dist = table.groupby(['branch-type'])['branch-distance'].median()

                    dic = dict(n_branches)
                    dic2 = dict(med_dist)
                    branch_types = [0,1,2,3] 
                    ind = [s not in list(dic.keys()) for s in branch_types]
                    #print(ind, dic, dic2)
                    if sum(ind) == 0:
                            sort = sorted(dic.items(), key=lambda kv: kv[0])
                            sort2 = sorted(dic2.items(), key=lambda kv: kv[0])
                            n_branches_ls = [s[1] for s in sort]
                            med_dist_ls = [s[1] for s in sort2]
                    else:
                        for z in range(len(list(compress(branch_types,ind)))):
                            dic.update({(list(compress(branch_types,ind)))[z]: 0})
                            dic2.update({(list(compress(branch_types,ind)))[z]: 0})
                            sort = sorted(dic.items(), key=lambda kv: kv[0])
                            sort2 = sorted(dic2.items(), key=lambda kv: kv[0])
                            n_branches_ls = [s[1] for s in sort]
                            med_dist_ls = [s[1] for s in sort2]
                
                    vals = [[i],n_branches_ls, med_dist_ls]
                    vals_all.append([v for sublist in vals for v in sublist])
                    #print([v for sublist in vals for v in sublist])
                df_skeleton = pd.DataFrame(vals_all, columns = ['cell', 'n_endpt_to_endpt_branches', 'n_junc_to_endpt_branches', 'n_junc_to_junc_branches',
             'n_isolated_cycle','ee_branch_med_dist', 'je_branch_med_dist', 'jj_branch_med_dist','is_cyc_med_dist'])

            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell'),
                df_skeleton.set_index('cell')], axis=1, join='inner')
                .reset_index())

        
        else: 
            df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell') 
                ], axis=1, join='inner')
                .reset_index())

    
       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    #########

    @staticmethod
    def _extract_phenotype_classical_7ch(data_phenotype, nuclei, cells, wildcards,channel,corrchannel1=None,
         corrchannel2=None, corrchannel3=None, corrchannel4=None, corrchannel5=None, corrchannel6=None):
 

        def correlate_channel_corrchannel1(region):

            if corrchannel1 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel1]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
        
        def correlate_channel_corrchannel2(region):

            if corrchannel2 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel2]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel3(region):

            if corrchannel3 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel3]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    
    
        def correlate_channel_corrchannel4(region):

            if corrchannel4 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel4]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel5(region):

            if corrchannel5 == None:
                return 0

            else:
                corrch = region.intensity_image_full[corrchannel5]
                refch = region.intensity_image_full[channel]

                filt = corrch > 0
                if filt.sum() == 0:
                    # assert False
                    return np.nan

                corrch = corrch[filt]
                refch  = refch[filt]
                corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

                return corr.mean()    

        def correlate_channel_corrchannel6(region):

            if corrchannel6 == None:
                return 0
            else:
                corrch = region.intensity_image_full[corrchannel6]
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


        features_nuclear = {
            'channel_corrch1_nuclear_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_nuclear_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_nuclear_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_nuclear_corr' : correlate_channel_corrchannel4,
            'channel_corrch5_nuclear_corr' : correlate_channel_corrchannel5,
            'channel_corrch6_nuclear_corr' : correlate_channel_corrchannel6,
            'area_nuclear'       : lambda r: r.area,
            'perimeter_nuclear' : lambda r: r.perimeter,
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }


        features_cell = {
            'channel_corrch1_cell_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cell_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cell_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cell_corr' : correlate_channel_corrchannel4,
            'channel_corrch5_cell_corr' : correlate_channel_corrchannel5,
            'channel_corrch6_cell_corr' : correlate_channel_corrchannel6,
            'area_cell' : lambda r: r.area,
            'perimeter_cell' : lambda r: r.perimeter,
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_min': lambda r: np.min(masked(r,channel)),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'channel_cell_sd': lambda r: np.std(masked(r,channel)),
            'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }

        
        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())

       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df


 #########

    @staticmethod
    def _extract_phenotype_classical_5ch(data_phenotype, nuclei, cells, wildcards,channel,corrchannel1,
         corrchannel2, corrchannel3, corrchannel4, granularity = False, skeleton = False):
 

        def correlate_channel_corrchannel1(region):

            corrch = region.intensity_image_full[corrchannel1]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    
    
        def correlate_channel_corrchannel2(region):

            corrch = region.intensity_image_full[corrchannel2]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    

        def correlate_channel_corrchannel3(region):

            corrch = region.intensity_image_full[corrchannel3]
            refch = region.intensity_image_full[channel]

            filt = corrch > 0
            if filt.sum() == 0:
                # assert False
                return np.nan

            corrch = corrch[filt]
            refch  = refch[filt]
            corr = (corrch - corrch.mean()) * (refch - refch.mean()) / (corrch.std() * refch.std())

            return corr.mean()    
    
        def correlate_channel_corrchannel4(region):

            corrch = region.intensity_image_full[corrchannel4]
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


        features_nuclear = {
            'channel_corrch1_nuclear_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_nuclear_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_nuclear_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_nuclear_corr' : correlate_channel_corrchannel4,
            'area_nuclear'       : lambda r: r.area,
            'perimeter_nuclear' : lambda r: r.perimeter,
            'channel_nuclear_mean' : lambda r: masked(r, channel).mean(),
            'channel_nuclear_median' : lambda r: np.median(masked(r, channel)),
            'channel_nuclear_int'    : lambda r: masked(r, channel).sum(),
            'channel_nuclear_min': lambda r: np.min(masked(r,channel)),
            'channel_nuclear_max'    : lambda r: masked(r, channel).max(),
            'channel_nuclear_sd': lambda r: np.std(masked(r,channel)),
            'channel_nuclear_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_nuclear_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_nuclear_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }


        features_cell = {
            'channel_corrch1_cell_corr' : correlate_channel_corrchannel1,
            'channel_corrch2_cell_corr' : correlate_channel_corrchannel2,
            'channel_corrch3_cell_corr' : correlate_channel_corrchannel3,
            'channel_corrch4_cell_corr' : correlate_channel_corrchannel4,
            'area_cell' : lambda r: r.area,
            'perimeter_cell' : lambda r: r.perimeter,
            'channel_cell_mean' : lambda r: masked(r, channel).mean(),
            'channel_cell_median' : lambda r: np.median(masked(r, channel)),
            'channel_cell_int'    : lambda r: masked(r, channel).sum(),
            'channel_cell_min': lambda r: np.min(masked(r,channel)),
            'channel_cell_max'    : lambda r: masked(r, channel).max(),
            'channel_cell_sd': lambda r: np.std(masked(r,channel)),
            'channel_cell_mad': lambda r: median_absolute_deviation(masked(r,channel)),    
            'channel_cell_25': lambda r: np.percentile(masked(r, channel),25),
            'channel_cell_75': lambda r: np.percentile(masked(r, channel),75),
            'cell'               : lambda r: r.label
        }

        
        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())

       
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df


    ########

    @staticmethod
    def _extract_phenotype_flow_5ch(data_phenotype, cells, wildcards, channels):
        ## will already yield area so no need to include


        if len(np.unique(cells)) < 2:
            print('no cells')
            

            return pd.DataFrame()

        else:
            def masked(region, index):
                return region.intensity_image_full[index][region.filled_image]


            features_cell = {
                'channel1_cell_int'    : lambda r: masked(r, channels[0]).sum(),
                'channel2_cell_int'    : lambda r: masked(r, channels[1]).sum(),
                'channel3_cell_int'    : lambda r: masked(r, channels[2]).sum(),
                'channel4_cell_int'    : lambda r: masked(r, channels[3]).sum(),
                'channel5_cell_int'    : lambda r: masked(r, channels[4]).sum()
                }

            df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

            ## calculate radius distances, edge intensity values, mass displacement
           
        
            df = (pd.concat([df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

           
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df

########

    @staticmethod
    def _extract_phenotype_flow_4ch_withmedian(data_phenotype, cells, wildcards, channels,cycle=None):
        ## will already yield area so no need to include

        if cycle != None:
            print(data_phenotype.shape)
            data_phenotype = data_phenotype[cycle]
            print(data_phenotype.shape)


        if len(np.unique(cells)) < 2:
            print('no cells')
            

            return pd.DataFrame()

        else:
            def masked(region, index):
                return region.intensity_image_full[index][region.filled_image]


            features_cell = {
                'channel1_cell_int'    : lambda r: masked(r, channels[0]).sum(),
                'channel2_cell_int'    : lambda r: masked(r, channels[1]).sum(),
                'channel3_cell_int'    : lambda r: masked(r, channels[2]).sum(),
                'channel4_cell_int'    : lambda r: masked(r, channels[3]).sum(),
                'channel1_cell_median' : lambda r: np.median(masked(r, 0)),
                'channel2_cell_median' : lambda r: np.median(masked(r, 1)),
                'channel3_cell_median' : lambda r: np.median(masked(r, 2)),
                'channel4_cell_median' : lambda r: np.median(masked(r, 3)),
                }

            df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

            ## calculate radius distances, edge intensity values, mass displacement
           
        
            df = (pd.concat([df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

           
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df


########

    @staticmethod
    def _extract_phenotype_flow_1ch(data_phenotype, cells, wildcards):
        ## will already yield area so no need to include


        if len(np.unique(cells)) < 2:
            print('no cells')
            

            return pd.DataFrame()

        else:
            def masked(region, index):
                return region.intensity_image_full[index][region.filled_image]


            features_cell = {
                'channel1_cell_int'    : lambda r: masked(r, 0).sum(),
                }

            df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

            ## calculate radius distances, edge intensity values, mass displacement
           
        
            df = (pd.concat([df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

           
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df

 ########

    @staticmethod
    def _extract_phenotype_flow_7ch(data_phenotype, cells, wildcards, channels):
        ## will already yield area so no need to include


        if len(np.unique(cells)) < 2:
            print('no cells')
            

            return pd.DataFrame()

        else:
            def masked(region, index):
                return region.intensity_image_full[index][region.filled_image]


            features_cell = {
                'channel1_cell_int'    : lambda r: masked(r, channels[0]).sum(),
                'channel2_cell_int'    : lambda r: masked(r, channels[1]).sum(),
                'channel3_cell_int'    : lambda r: masked(r, channels[2]).sum(),
                'channel4_cell_int'    : lambda r: masked(r, channels[3]).sum(),
                'channel5_cell_int'    : lambda r: masked(r, channels[4]).sum(),
                'channel6_cell_int'    : lambda r: masked(r, channels[5]).sum(),
                'channel7_cell_int'    : lambda r: masked(r, channels[6]).sum()
                }

            df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 

            ## calculate radius distances, edge intensity values, mass displacement
           
        
            df = (pd.concat([df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

           
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df

    ########

    @staticmethod
    def _extract_phenotype_extended_morphology(data_phenotype, nuclei, cells, wildcards, cytoplasm = None, max = None):
 
        if max != None:
            data_phenotype = data_phenotype.max(axis=0)

        def masked(region, index):
            return region.intensity_image_full[index][region.filled_image]

     

        features_nuclear = {
            'area_nuclear': lambda r: r.area,
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
            'area_cell': lambda r: r.area,
            'perimeter_cell': lambda r: r.perimeter,
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
        if cytoplasm != None:
            df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

        ## calculate radius distances, mass displacement
#        dist = []
#        for i in np.unique(cells):
#            if i == 0:
#                continue
#            if (i % 100 == 0):
#                print(i)
#            mask = cells == i
#            tmp = data_phenotype[1].copy()
#            tmp[mask == False] = 0
#            dists = distance_transform_edt(tmp, sampling=None, return_distances=True)
#                   
#            max_rad = np.max(dists)
#            mean_rad = np.mean(dists)
#            median_rad = np.median(dists)
#            rad_std = np.std(dists)
#
#            comgray = ndimage.measurements.center_of_mass(tmp)
#            binary = tmp.copy()
#            binary[binary!=0] = 1
#            combin = ndimage.measurements.center_of_mass(binary) #different from skimage centroid (i,j) in resulting .pkl
#            euclid_dist_cell = ((comgray[0] - combin[0]) ** 2 +
#                        (comgray[1] - combin[1]) ** 2) ** 0.5
#
#            tmp_msd = []
#            nonzero = np.nonzero(tmp) #inds of nonzero objects
#            for x in range(len(nonzero[0])):
#                tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
#            msd_cell = np.mean(tmp_msd)
#
#            mask = nuclei == i
#            tmp = data_phenotype[1].copy()
#            tmp[mask == False] = 0
#            comgray = ndimage.measurements.center_of_mass(tmp)
#            binary = tmp.copy()
#            binary[binary!=0] = 1
#            combin = ndimage.measurements.center_of_mass(binary) #technically no need to recalc, this is just skimage centroid
#            euclid_dist_nuc = ((comgray[0] - combin[0]) ** 2 +
#                        (comgray[1] - combin[1]) ** 2) ** 0.5
#
#            tmp_msd = []
#            nonzero = np.nonzero(tmp) #inds of nonzero objects
#            for x in range(len(nonzero[0])):
#                tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
#            msd_nuc = np.mean(tmp_msd)
#
#            dist.append((i, max_rad, mean_rad, rad_std, 
#                euclid_dist_cell, euclid_dist_nuc, msd_cell, msd_nuc))
#
#        df_radii = pd.DataFrame(dist, columns = ['cell', 'max_cell_radius', 'mean_cell_radius', 'radius_cell_SD',
#                                                'cell_mass_displacement', 'nuc_mass_displacement',
#                                            'mean_sq_dist_cell', 'mean_sq_dist_nuc'])
#    
#        
#        #print(list(set(df_n.columns) | set(df_c.columns) | set(df_cyto.columns)))

        if cytoplasm != None:
            if 'cell' not in list(set(df_n.columns) & set(df_c.columns) & set(df_cyto.columns)):
                print('no cells')
                return None
            else: 
                df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell')],axis=1, join='inner')
                    .reset_index())

        else:

            if 'cell' not in list(set(df_n.columns) & set(df_c.columns)):
                print('no cells')
                return None
            else: 
                df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                    .reset_index())

       
            df = df.loc[:, ~df.columns.duplicated()]
        
            return df



    # @staticmethod
    # def _extract_phenotype_extended_morphology(data_phenotype, nuclei, cells, cytoplasm, wildcards):
 

    #     def masked(region, index):
    #         return region.intensity_image_full[index][region.filled_image]

     

    #     features_nuclear = {
    #         'eccentricity_nuclear' : lambda r: r.eccentricity, #cell
    #         'major_axis_nuclear' : lambda r: r.major_axis_length, #cell
    #         'minor_axis_nuclear' : lambda r: r.minor_axis_length, #cell
    #         'orientation_nuclear' : lambda r: r.orientation,
    #         'hu_moments_nuclear': lambda r: r.moments_hu,
    #         'solidity_nuclear': lambda r: r.solidity,
    #         'extent_nuclear': lambda r: r.extent,
    #         'cell'               : lambda r: r.label
    #     }

    #     features_cell = {
    #         'euler_cell' : lambda r: r.euler_number,
    #         'eccentricity_cell' : lambda r: r.eccentricity, #cell
    #         'major_axis_cell' : lambda r: r.major_axis_length, #cell
    #         'minor_axis_cell' : lambda r: r.minor_axis_length, #cell
    #         'orientation_cell' : lambda r: r.orientation,
    #         'hu_moments_cell': lambda r: r.moments_hu,
    #         'solidity_cell': lambda r: r.solidity,
    #         'extent_cell': lambda r: r.extent,
    #         'cell'               : lambda r: r.label
    #     }

    #     features_cytoplasm = {
    #         'euler_cyto' : lambda r: r.euler_number,
    #         'area_cyto'       : lambda r: r.area,
    #         'perimeter_cyto' : lambda r: r.perimeter,
    #         'cell'            : lambda r: r.label
    #     }

    #     df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_nuclear)
    #     df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_cell) 
    #     df_cyto =  Snake._extract_features(data_phenotype, cytoplasm, wildcards, features_cytoplasm) 

    #     ## calculate radius distances, mass displacement
    #     dist = []
    #     for i in np.unique(cells):
    #         if i == 0:
    #             continue
    #         mask = cells == i
    #         tmp = data_phenotype[1].copy()
    #         tmp[mask == False] = 0
    #         dists = distance_transform_edt(tmp, sampling=None, return_distances=True)
                   
    #         max_rad = np.max(dists)
    #         mean_rad = np.mean(dists)
    #         median_rad = np.median(dists)
    #         rad_std = np.std(dists)

    #         comgray = ndimage.measurements.center_of_mass(tmp)
    #         binary = tmp.copy()
    #         binary[binary!=0] = 1
    #         combin = ndimage.measurements.center_of_mass(binary) #different from skimage centroid (i,j) in resulting .pkl
    #         euclid_dist_cell = ((comgray[0] - combin[0]) ** 2 +
    #                     (comgray[1] - combin[1]) ** 2) ** 0.5

    #         tmp_msd = []
    #         nonzero = np.nonzero(tmp) #inds of nonzero objects
    #         for x in range(len(nonzero[0])):
    #             tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
    #         msd_cell = np.mean(tmp_msd)

    #         mask = nuclei == i
    #         tmp = data_phenotype[1].copy()
    #         tmp[mask == False] = 0
    #         comgray = ndimage.measurements.center_of_mass(tmp)
    #         binary = tmp.copy()
    #         binary[binary!=0] = 1
    #         combin = ndimage.measurements.center_of_mass(binary) #technically no need to recalc, this is just skimage centroid
    #         euclid_dist_nuc = ((comgray[0] - combin[0]) ** 2 +
    #                     (comgray[1] - combin[1]) ** 2) ** 0.5

    #         tmp_msd = []
    #         nonzero = np.nonzero(tmp) #inds of nonzero objects
    #         for x in range(len(nonzero[0])):
    #             tmp_msd.append(((nonzero[0][x]-combin[0])**2+(nonzero[1][x]-combin[1])**2)) # find mean squared distance from centroid
    #         msd_nuc = np.mean(tmp_msd)

    #         dist.append((i, max_rad, mean_rad, rad_std, 
    #             euclid_dist_cell, euclid_dist_nuc, msd_cell, msd_nuc))

    #     df_radii = pd.DataFrame(dist, columns = ['cell', 'max_cell_radius', 'mean_cell_radius', 'radius_cell_SD',
    #                                             'cell_mass_displacement', 'nuc_mass_displacement',
    #                                         'mean_sq_dist_cell', 'mean_sq_dist_nuc'])
    
    #     df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell'), df_cyto.set_index('cell'), df_radii.set_index('cell')], axis=1, join='inner')
    #             .reset_index())

       
    #     df = df.loc[:, ~df.columns.duplicated()]
        
    #     return df

    @staticmethod
    def _bg_subtract(data, ecm):
        """subtract median of ecm in tile
        """

        mask = ecm > 0
        bg = data[:,mask==True]
        print(np.mean(data,axis=0))
        bg_means = np.mean(bg,axis=0)
        
        data = np.stack((data[0]-bg_means[0],data[1]-bg_means[1],

            data[2]-bg_means[2],data[3]-bg_means[3],data[4]-bg_means[4]))
        print(data.shape, np.mean(data,axis = 0))
        data[data < 0] = 0
      
        return data

    @staticmethod
    def _bg_subtract_cell(data, cells):
        """subtract median of ecm in tile
        """
        data_subtracted = data.copy()
        print(data_subtracted.shape)
        for cell in np.unique(cells):
            #print(cell)
            if cell == 0:
                continue
            else:
                mask = cells == cell
                bg = data[:,mask==True]
                bg_means = np.mean(bg,axis = 1)
                print(bg_means.shape,bg_means)
                #print(mask.shape, data_subtracted.shape, data.shape,data_subtracted[0,mask].shape)
                data_subtracted[0,mask] = data_subtracted[0,mask==True]-bg_means[0]
                data_subtracted[1,mask] = data_subtracted[1,mask==True]-bg_means[1]
                data_subtracted[2,mask] = data_subtracted[2,mask==True]-bg_means[2]
                data_subtracted[3,mask] = data_subtracted[3,mask==True]-bg_means[3]
                data_subtracted[4,mask] = data_subtracted[4,mask==True]-bg_means[4]
                # data_subtracted[1,cells == cell]-bg_means[1],
                # data_subtracted[2,cells == cell]-bg_means[2],
                # data_subtracted[3,cells == cell]-bg_means[3],
                # data_subtracted[4,cells == cell]-bg_means[4]))

        # mask = ecm > 0
        # bg = data[:,mask==True]
        # print(np.mean(data,axis=0))
        # bg_means = np.mean(bg,axis=0)
        
        # data = np.stack((data[0]-bg_means[0],data[1]-bg_means[1],

        #     data[2]-bg_means[2],data[3]-bg_means[3],data[4]-bg_means[4]))
        # print(data.shape, np.mean(data,axis = 0))
        # data[data < 0] = 0
      
        return data_subtracted


    ########

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
        #print(sum(sum(data[0,3])))



        def log_ndi(data, sigma=1, *args, **kwargs):
            import scipy
            print('inner log ndi')
            f = scipy.ndimage.filters.gaussian_laplace
            print(sigma, data.shape)
            arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
            arr_ = np.clip(arr_, 0, 65535) / 65535
            # arr_[arr_ < 0] = 0
            # arr_ /= arr_.max()
            print(arr_.shape)
            # skimage precision warning 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.img_as_uint(arr_)

        cycles = data.shape[1]
        channels = data.shape[2]
        
        loged = []
        for cycle in range(cycles):
            tmp = []
            for channel in range(channels):
                print(cycle, channel)
                print(data.shape)
                print(data[0,cycle,channel].shape)
                tmp.append(log_ndi(data[0,cycle, channel], sigma = sigma))
                
            loged.append(tmp)
        
        loged = np.array(loged)[np.newaxis, :]
        print(loged.shape)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        #print(sum(sum(loged[0,3])))
        return loged

    @staticmethod
    def _transform_log_bychannel(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        #print(sum(sum(data[0,3])))
        #print(data.shape,data.dtype)
        def log_ndi(data, sigma=1, *args, **kwargs):
            import scipy
            print('inner log ndi')
            f = scipy.ndimage.filters.gaussian_laplace
            print(sigma, data.shape)
            arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
            arr_ = np.clip(arr_, 0, 65535) / 65535
            # arr_[arr_ < 0] = 0
            # arr_ /= arr_.max()
            print(arr_.shape)
            # skimage precision warning 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.img_as_uint(arr_)

        print(sigma,data.shape)
        channels = data.shape[2]
        loged = []
        for channel in range(channels):
            tmp = data[:,:,channel]
            print(tmp.shape)
            loged.append(log_ndi(tmp, sigma=sigma))

        #print(np.array(loged).reshape(data.shape).shape)

        loged = np.array(loged).reshape(data.shape[0],data.shape[2],data.shape[1],data.shape[3],data.shape[4])
        print(loged.shape)
        loged=np.swapaxes(loged,1,2)
        print(loged.shape)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        #print(sum(sum(loged[0,3])))
        return loged


    @staticmethod
    def _transform_log_bycycle(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip transforming a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        #print(sum(sum(data[0,3])))
        #print(data.shape,data.dtype)
        def log_ndi(data, sigma=1, *args, **kwargs):
            import scipy
            print('inner log ndi')
            f = scipy.ndimage.filters.gaussian_laplace
            print(sigma, data.shape)
            arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
            arr_ = np.clip(arr_, 0, 65535) / 65535
            # arr_[arr_ < 0] = 0
            # arr_ /= arr_.max()
            print(arr_.shape)
            # skimage precision warning 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.img_as_uint(arr_)

        print(sigma,data.shape)
        cycles = data.shape[1]
        loged = []
        for cycle in range(cycles):
            tmp = data[:,cycle]
            print(tmp.shape)
            loged.append(log_ndi(tmp, sigma=sigma))

        #print(np.array(loged).reshape(data.shape).shape)

        loged = np.array(loged).reshape(data.shape[0],data.shape[1],data.shape[2],data.shape[3],data.shape[4])
        print(loged.shape)
        #loged=np.swapaxes(loged,1,2)
        #print(loged.shape)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        #print(sum(sum(loged[0,3])))
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation to estimate sequencing read locations.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        leading_dims = tuple(range(0, data.ndim - 2))
        # consensus = np.std(data, axis=leading_dims)
        consensus = np.std(data, axis=0).mean(axis=0)

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

    
    # @staticmethod
    # def _find_peaks(data, remove_index=None, data_index=None):
    #     """Find local maxima and label by difference to next-highest neighboring
    #     pixel.
    #     """
    #     if remove_index is not None:
    #         data = remove_channels(data, remove_index)

    #     if data_index is not None:
    #         data = data[data_index]

    #     if data.ndim == 2:
    #         data = [data]

    #     peaks = [ops.process.find_peaks(x) 
    #                 if x.max() > 0 else x 
    #                 for x in data]
    #     peaks = np.array(peaks).squeeze()
    #     return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None, remove_index2=None):
        """Apply a maximum filter in a window of `width`.
        """
        import scipy.ndimage.filters

        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        if remove_index2 is not None:
            data = remove_channels(data, remove_index2)

        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (typically well and tile) and 
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
    def _extract_bases_downsample_cells(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (typically well and tile) and 
        label at that position in integer mask `cells`.
        """

        if maxed.ndim == 3:
            maxed = maxed[None]

        print(np.unique(cells))
        print(cells.dtype)
        print(cells.shape,maxed[0,0,:].shape)

        from skimage.util import img_as_uint, img_as_float
        # cells = img_as_uint(skimage.transform.resize(img_as_cells, maxed[0,0,:].shape,
        #                 anti_aliasing=True, mode = 'constant'))#, preserve_range=True)
        
        cells = np.resize((cells), maxed[0,0,:].shape)
        print(cells.shape,maxed.shape)
        print(np.unique(cells))

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
    def _extract_bases_DO(log, cells, threshold_bases, wildcards, channel_DO = 1, bases='G'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (typically well and tile) and 
        label at that position in integer mask `cells`.
        """
        print(channel_DO)

        if log.ndim == 3:
            log = log[None]

        # if len(bases) != log.shape[:,1:]:
        #     error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
        #     raise ValueError(error.format(len(bases), bases, log.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, log.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity_DO(log, cells, threshold_bases, channel_DO))

        df_bases = ops.in_situ.format_bases_DO(values, labels, positions, bases)

        for k,v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases, peaks=None, correction_only_in_cells=True, position_column = 'site'):
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

        print(df_bases.head())
        if position_column != 'site':
            df_bases['site'] = df_bases[position_column]

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles=cycles, channels=channels, correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads


    @staticmethod
    def _call_reads_mbc(df_bases, peaks=None, correction_only_in_cells=True, position_column = 'site'):
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        median call is by cycle
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return


        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        if position_column != 'site':
            df_bases['site'] = df_bases[position_column]

        print(df_bases.head())
        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call_bycycle, cycles=cycles, channels=channels, correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_reads_firstncycles(df_bases, ncycles, peaks=None, correction_only_in_cells=True):
        """Median correction performed independently for each tile.
        Use the `correction_only_in_cells` flag to specify if correction
        is based on reads within cells, or all reads.
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return

        #ncycles = int(ncycles)
        df_bases = df_bases[df_bases.cycle <= ncycles]
        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles=cycles, channels=channels, correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

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

    @staticmethod
    def _call_reads_nomedian(df_bases, peaks=None, correction_only_in_cells=True):

        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return


        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))
        print("")
        print(channels, cycles)
        print("")
        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call_nomedian, cycles=cycles, channels=channels, correction_only_in_cells=correction_only_in_cells)
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
    def _call_cells_highmoi(df_reads, q_min=0):
        """Median correction performed independently for each tile.
        """
        if df_reads is None:
            return
        
        return (df_reads
            .query('Q_min >= @q_min')
            .pipe(ops.in_situ.call_cells_highmoi))


    @staticmethod
    def _extract_features(data, nuclei, wildcards, features=None):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        from ops.features import features_cell
        features = features.copy() if features else dict()
        features.update(features_cell)

        df = feature_table(data, nuclei, features)

        for k,v in sorted(wildcards.items()):
            df[k] = v
        
        return df

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift_haonly
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift_haonly)       

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features)     

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards, channel0 = None, channel1 = None):
        import ops.features

        print(data_phenotype[0])
        if channel0 != None:
            data_phenotype[0] = data_phenotype[channel0]
        if channel1 != None:
            data_phenotype[1] = data_phenotype[channel1]

        print(data_phenotype[0])

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_c) 

        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())

        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    @staticmethod
    def _extract_phenotype_translocation_nuclear(data_phenotype, nuclei, wildcards, max = None, channel0 = None, channel1 = None):
        import ops.features

        if max != None:
            data_phenotype = data_phenotype.max(axis=1)
        print(data_phenotype.shape)
        print(data_phenotype[0],data_phenotype[1])
        if channel0 != None:
            data_phenotype[0] = data_phenotype[channel0]
        if channel1 != None:
            data_phenotype[1] = data_phenotype[channel1]

        print(data_phenotype[0],data_phenotype[1])

        features_n = ops.features.features_translocation_nuclear

        features_n = {k + '_nuclear': v for k,v in features_n.items()}

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)

        # inner join discards nuclei without corresponding cells
        # df = (pd.concat([df_n.set_index('cell'), axis=1, join='inner')
        #         .reset_index())

        #df = df.loc[:, ~df.columns.duplicated()]
        
        return df_n

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)

    @staticmethod
    def _extract_minimal_phenotype(data_phenotype, nuclei, wildcards):
        return Snake._extract_features(data, nuclei, wildcards, dict())

    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], call_from_snakemake(f))


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

        result = f(**input_kwargs)

        if 'output' in output_kwargs:
            save_output(output_kwargs['output'], result, **output_kwargs)

    return functools.update_wrapper(g, f)


def remove_channels(data, remove_index, remove_index2 = None):
    """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
    """
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    if remove_index2 != None:
        channels_mask[remove_index2] = False

    data = data[..., channels_mask, :, :]
    return data


def load_arg(x):
    """Try loading data from `x` if it is a string or list of strings.
    If that fails just return `x`.
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


def load_well_tile_list(filename):
    wells, tiles = pd.read_pickle(filename)[['well', 'tile']].values.T
    return wells, tiles


def load_csv(filename):
    with open(filename, 'r') as fh:
        txt = fh.readline()
    sep = ',' if ',' in txt else '\s+'
    return pd.read_csv(filename, sep=sep)


def load_pkl(filename):
    return pd.read_pickle(filename)


def load_tif(filename):
    return ops.io.read_stack(filename)

def load_nd2(filename):
    return ops.io.read_nd2_stack(filename)

def save_csv(filename, df):
    df.to_csv(filename, index=None)


def save_pkl(filename, df):
    df.to_pickle(filename)


def save_tif(filename, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # make sure `data` doesn't come from the Snake method since it's an
    # argument name for the save function, too
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
    if filename.endswith('.nd2'):
        return load_nd2(filename)
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



Snake.load_methods()
