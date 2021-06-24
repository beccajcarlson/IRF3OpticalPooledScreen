import warnings
from collections import defaultdict
from itertools import product

import skimage
import skimage.feature
import skimage.filters
import numpy as np
import pandas as pd
import scipy.stats

from scipy import ndimage

import ops.io
import ops.utils 


# FEATURES
def feature_table(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data 
    specified by integer labels. If provided, the global feature
    dictionary is applied to the full input data and labels. 

    Results are combined in a dataframe with one row per label and
    one column per feature.
    """
    regions = ops.utils.regionprops(labels, intensity_image=data)
    results = defaultdict(list)
    for region in regions:
        for feature, func in features.items():
            results[feature].append(func(region))
    if global_features:
        for feature, func in global_features.items():
            results[feature] = func(data, labels)
    return pd.DataFrame(results)


def build_feature_table(stack, labels, features, index):
    """Iterate over leading dimensions of stack, applying `feature_table`. 
    Results are labeled by index and concatenated.

        >>> stack.shape 
        (3, 4, 511, 626)
        
        index = (('round', range(1,4)), 
                 ('channel', ('DAPI', 'Cy3', 'A594', 'Cy5')))
    
        build_feature_table(stack, labels, features, index) 

    """
    index_vals = list(product(*[vals for _, vals in index]))
    index_names = [x[0] for x in index]
    
    s = stack.shape
    results = []
    for frame, vals in zip(stack.reshape(-1, s[-2], s[-1]), index_vals):
        df = feature_table(frame, labels, features)
        for name, val in zip(index_names, vals):
            df[name] = val
        results += [df]
    
    return pd.concat(results)


def find_cells(nuclei, mask, remove_boundary_cells=False):
    """Convert binary mask to cell labels, based on nuclei labels.

    Expands labeled nuclei to cells, constrained to where mask is >0.
    """
    distance = ndimage.distance_transform_cdt(nuclei == 0)
    cells = skimage.morphology.watershed(distance, nuclei, mask=mask)
    # remove cells touching the boundary
    if remove_boundary_cells:
        cut = np.concatenate([cells[0,:], cells[-1,:], 
                              cells[:,0], cells[:,-1]])
        cells.flat[np.in1d(cells, np.unique(cut))] = 0

    return cells.astype(np.uint16)


def find_peaks(data, n=5): #normal cells 5
    """Finds local maxima. At a maximum, the value is max - min in a 
    neighborhood of width `n`. Elsewhere it is zero.
    """
    from scipy.ndimage import filters
    neighborhood_size = (1,)*(data.ndim-2) + (n,n)
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    peaks = data_max - data_min
    peaks[data != data_max] = 0
    
    # remove peaks close to edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0
    
    return peaks

@ops.utils.applyIJ
def log_ndi(data, sigma=1, *args, **kwargs):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J). 
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    Inverts output and converts back to uint16.
    """
    f = scipy.ndimage.filters.gaussian_laplace
    print(sigma, data.shape)
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
    arr_[arr_ < 0] = 0
    arr_ /= arr_.max()
    print(arr_.shape)
    print(sum(sum(arr_[3])))
    # skimage precision warning 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)
    

class Align:
    """Alignment redux, used by snakemake.
    """
    @staticmethod
    def normalize_by_percentile(data_, q_norm=70):
        shape = data_.shape
        shape = shape[:-2] + (-1,)
        p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
        normed = data_ / p
        return normed
    
    @staticmethod
    def calculate_offsets(data_, upsample_factor):
        target = data_[0]
        offsets = []
        for i, src in enumerate(data_):
            if i == 0:
                offsets += [(0, 0)]
            else:
                offset, _, _ = skimage.feature.register_translation(
                                src, target, upsample_factor=upsample_factor)
                offsets += [offset]
        return offsets

    # @staticmethod
    # def calculate_offsets_rigidbody(data_, upsample_factor):
    #     from pystackreg import StackReg
    #     sr = StackReg(StackReg.RIGID_BODY)

    #     target = data_[0]
    #     offsets = []
    #     for i, src in enumerate(data_):
    #         if i == 0:
    #             offsets += [(0, 0)]

    #         out_rot = sr.register_transform(src, target)
    #         else:
    #             offset, _, _ = skimage.feature.register_translation(
    #                             src, target, upsample_factor=upsample_factor)
    #             offsets += [offset]
    #     return offsets

    @staticmethod
    def apply_offsets(data_, offsets):
        print(data_.shape[0])
        # if sum(sum(np.abs(offsets))) > 100*data_.shape[0]:
        #     print ('alignment failed')
        #     return np.zeros(data_.shape)

        #else:
        warped = []
        print(sum(sum(np.abs(offsets))))
        for frame, offset in zip(data_, offsets):
            if offset[0] == 0 and offset[1] == 0:
                warped += [frame]
            else:
                # skimage has a weird (i,j) <=> (x,y) convention
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_ = skimage.transform.warp(frame, st, preserve_range=True)
                warped += [frame_.astype(data_.dtype)]


        return np.array(warped)
    
    @staticmethod
    def register_and_offset(images, registration_images=None, verbose=True):
        """Wrapper around `register_images` and `offset`.
        """
        if registration_images is None:
            registration_images = images
        offsets = Align.register_images(registration_images)
        # if verbose:
        #     print(np.array(offsets))
        aligned = [ops.utils.offset(d, o) for d,o in zip(images, offsets)]

        if verbose:
            print(sum(sum(np.abs(offsets)))) #>0, < 75


        if sum(sum(np.abs(offsets))) > 500:
            print ('alignment failed')
            return np.zeros(images.shape)#np.array((0))

        return np.array(aligned)

    @staticmethod
    def register_images(images, index=None, window=(500, 500), upsample=1., verbose=True): 
        #window default (500,500), upsample 1.
        """Register image stacks to pixel accuracy.
        :param images: list of N-dim image arrays, height and width may differ
        :param index: image[index] should yield 2D array with which to perform alignment
        :param window: centered window in which to perform registration, smaller is faster
        :param upsample: align to sub-pixels of width 1/upsample
        :return list[(int)]: list of offsets
        """
        if index is None:
            index = ((0,) * (images[0].ndim - 2) + (slice(None),) * 2)

        sz = [image[index].shape for image in images]
        sz = np.array([max(x) for x in zip(*sz)])

        origin = np.array(images[0].shape) * 0.

        center = tuple([slice(s / 2 - min(s / 2, rw), s / 2 + min(s / 2, rw))
                        for s, rw in zip(sz, window)])

        def pad(img):
            pad_width = [(s // 2, s - s // 2) for s in (sz - img.shape)]
            print(pad_width)
            img = np.pad(img, pad_width, 'constant')
            return img[center], np.array([x[0] for x in pad_width]).astype(float)

        image0, pad_width = pad(images[0][index])
        offsets = [origin.copy()]
        offsets[0][-2:] += pad_width
        for image in [x[index] for x in images[1:]]:
            padded, pad_width = pad(image)
            shift, error, _ = register_translation(image0, padded, upsample_factor=upsample)

            offsets += [origin.copy()]
            offsets[-1][-2:] = shift + pad_width  # automatically cast to uint64

        

        return offsets


    @staticmethod
    def align_within_cycle(data_, upsample_factor=4, window=1):
        normed = Align.normalize_by_percentile(Align.apply_window(data_, window))
        offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)

        return Align.apply_offsets(data_, offsets)

    @staticmethod
    def align_between_cycles(data, channel_index, upsample_factor=4, window=1):
        # offsets from target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor)

        # apply to all channels
        warped = []
        for data_ in data.transpose([1, 0, 2, 3]):
            warped += [Align.apply_offsets(data_, offsets)]

        return np.array(warped).transpose([1, 0, 2, 3])

    @staticmethod
    def apply_window(data, window):
        print(data.shape, window)
        find_border = lambda x: int((x/2.) * (1 - 1/float(window)))
        i, j = find_border(data.shape[-2]), find_border(data.shape[-1])
        return data[..., i:-i, j:-j]


# SEGMENT
def find_nuclei(dapi, threshold, radius=15, area_min=50, area_max=500,
                score=lambda r: r.mean_intensity,
                smooth=1.35):

# find_nuclei(dapi, threshold, radius=150, area_min=50, area_max=500,
#                 score=lambda r: r.mean_intensity,
#                 smooth=20): 
    """
    """
    print(radius, area_min, area_max)
    print('THRESH ')

    mask = binarize(dapi, radius, area_min)
    print('bin done')
    print(sum(sum(mask)))
    labeled = skimage.measure.label(mask)
    labeled = filter_by_region(labeled, score, threshold, intensity_image=dapi) > 0
    print('lab')
    print(np.min(dapi),np.max(dapi))

    # only fill holes below minimum area
    filled = ndimage.binary_fill_holes(labeled)
    difference = skimage.measure.label(filled!=labeled)


    change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    nuclei = apply_watershed(labeled, smooth=smooth)

    result = filter_by_region(nuclei, lambda r: area_min < r.area < area_max, threshold)
    print(np.max(result))
    print(result.shape)

    return result


def binarize(image, radius, min_size):
    """Apply local mean threshold to find outlines. Filter out
    background shapes. Otsu threshold on list of region mean intensities will remove a few
    dark cells. Could use shape to improve the filtering.
    """
    print(np.max(image))
    print(np.min(image))
    dapi = skimage.img_as_ubyte(image/100000)
    # slower than optimized disk in ImageJ
    # scipy.ndimage.uniform_filter with square is fast but crappy
    selem = skimage.morphology.disk(radius)
    mean_filtered = skimage.filters.rank.mean(dapi, selem=selem)
    mask = dapi > mean_filtered
    print((sum(sum(mask))))
    print('min size ',min_size)
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)
    print((sum(sum(mask))))

    return mask

@ops.utils.applyIJ
def log_ndi(data, sigma=1, *args, **kwargs):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J). 
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    Inverts output and converts back to uint16.
    """
    f = scipy.ndimage.filters.gaussian_laplace
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)
    arr_ = np.clip(arr_, 0, 65535) / 65535
    
    # skimage precision warning 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)

        
def filter_by_region(labeled, score, threshold, intensity_image=None, relabel=True):
    """Apply a filter to label image. The `score` function takes a single region 
    as input and returns a score. 
    If scores are boolean, regions where the score is false are removed.
    Otherwise, the function `threshold` is applied to the list of scores to 
    determine the minimum score at which a region is kept.
    If `relabel` is true, the regions are relabeled starting from 1.
    """
    labeled = labeled.copy().astype(int)
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image)
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        t = threshold(scores)
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0
    
    if relabel:
        labeled, _, _ = skimage.segmentation.relabel_sequential(labeled)

    return labeled


def apply_watershed(img, smooth=4):
    distance = ndimage.distance_transform_edt(img)
    if smooth > 0:
        distance = skimage.filters.gaussian(distance, sigma=smooth)
    local_max = skimage.feature.peak_local_max(
                    distance, indices=False, footprint=np.ones((3, 3)), 
                    exclude_border=False)

    markers = ndimage.label(local_max)[0]
    result = skimage.morphology.watershed(-distance, markers, mask=img)
    return result.astype(np.uint16)

