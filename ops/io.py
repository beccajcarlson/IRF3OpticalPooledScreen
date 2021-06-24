import numpy as np
import os
import pandas as pd
import six
import struct
import warnings

import ops.utils
import ops.constants
from ops.external.tifffile_old import imsave
from ops.external.tifffile_new import imread

####
from nd2reader import ND2Reader

###


imagej_description = ''.join(['ImageJ=1.49v\nimages=%d\nchannels=%d\nslices=%d',
                              '\nframes=%d\nhyperstack=true\nmode=composite',
                              '\nunit=\\u00B5m\nspacing=8.0\nloop=false\n',
                              'min=764.0\nmax=38220.0\n'])

ramp = list(range(256))
ZERO = [0]*256
RED     = ramp + ZERO + ZERO
GREEN   = ZERO + ramp + ZERO
BLUE    = ZERO + ZERO + ramp
MAGENTA = ramp + ZERO + ramp
GRAY    = ramp + ramp + ramp
CYAN    = ZERO + ramp + ramp

DEFAULT_LUTS = GRAY, GREEN, RED, MAGENTA, CYAN, GRAY, GRAY


def read_lut(lut_string):
	return pd.read_csv(six.StringIO(lut_string), sep='\s+', header=None).values

GLASBEY = read_lut(ops.constants.GLASBEY_INVERTED)


def grid_view(files, bounds, padding=40, with_mask=False):
    """Mask is 1-indexed, zero indicates background.
    """
    padding = int(padding)

    arr = []
    for filename, bounds_ in zip(files, bounds):
        # some memory issue right now
        I = read_stack(filename, memmap=False, copy=False) 
        I_cell = ops.utils.subimage(I, bounds_, pad=padding)
        arr.append(I_cell.copy())

    if with_mask:
        arr_m = []
        for i, (i0, j0, i1, j1) in enumerate(bounds):
            shape = i1 - i0 + padding, j1 - j0 + padding
            img = np.zeros(shape, dtype=np.uint16) + i + 1
            arr_m += [img]
        return ops.utils.pile(arr), ops.utils.pile(arr_m)

    return ops.utils.pile(arr)


# @ops.utils.memoize
# def read_stack(filename, copy=True):
#     """Read a .tif file into a numpy array, with optional memory mapping.
#     """
#     print(filename)
#     print(imread(filename,multifile=False).shape)
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", message="not an ome-tiff master file")
#         data = imread(filename, multifile=False)
#     # preserve inner singleton dimensions
#     while data.shape[0] == 1:
#         data = np.squeeze(data, axis=(0,))

#     if copy:
#     	data = data.copy()
#     return data

@ops.utils.memoize(active=False)
def read_stack(filename, copy=True):
    """Read a .tif file into a numpy array, with optional memory mapping.
    """
    data = imread(filename, multifile=False, is_ome=False)
    # preserve inner singleton dimensions
    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))

    if copy:
        data = data.copy()
    return data

@ops.utils.memoize(active=False)
def read_nd2_stack(filename, copy=True):
    """Read a .tif file into a numpy array, with optional memory mapping.
    """

    with ND2Reader(filename) as images:
        print(images.sizes)
        if 't' in images.sizes.keys():
            print('time-series')
            if 'c' in images.sizes.keys():
                images.bundle_axes = 'cxy'
            else:
                images.bundle_axes = 'xy'

            imlist = []
            print(len(images))
            print(images.bundle_axes)
            i = 0
            # for fov in images:
            #     print(i)
            #     i += 1
            #     print(fov.shape)
            #     imlist.append(fov)
            #     print(len(imlist))
            # print(len(imlist))
            data = np.array(images)
            print(data.shape)

        elif 'z' in images.sizes.keys():
            print('z-stack')
            if 'c' in images.sizes.keys():
                images.bundle_axes = 'cxy'
            else:
                images.bundle_axes = 'xy'

            imlist = []
            for fov in images:
                imlist.append(fov)
            data = np.array(imlist).max(axis = 0)
        
        else: 
            data = np.array(images)
            print(np.array(images),data.shape)

    return data


# TODO fix extension of luts and display_ranges when not provided
def save_stack(name, data, luts=None, display_ranges=None, 
               resolution=1., compress=0, save_multi_output=False):
    """
    Saves `data`, an array with 5, 4, 3, or 2 dimensions [TxZxCxYxX]. `resolution` 
    can be specified in microns per pixel. Setting `compress` to 1 saves a lot of 
    space for integer masks. The ImageJ lookup table for each channel can be set
    with `luts` (e.g, ops.io.GRAY, ops.io.GREEN, etc). The display range can be set
    with a list of (min, max) pairs for each channel.

    >>> random_data = np.random.randint(0, 2**16, size=(3, 100, 100), dtype=np.uint16)
    >>> luts = ops.io.GREEN, ops.io.RED, ops.io.BLUE
    >>> display_ranges = (0, 60000), (0, 40000), (0, 20000)
    >>> save('random_image', random_data, luts=luts, display_ranges=display_ranges)

    Compatible array data types are:
    	bool (converted to uint8 0, 255)
    	uint8 
    	uint16 
    	float32 
    	float64 (converted to float32)
    """
    
    if save_multi_output == True:
        print(type(data))
        print(data[0].shape)
        # save list of data to output
        name = name.split('.tif ')
        for i in range(len(name)):
            if name[i].split('.')[-1] != 'tif':
                name[i] += '.tif'
            name[i] = os.path.abspath(name[i])
            imsave(name[i], data[i], photometric='minisblack')

    else:
        if name.split('.')[-1] != 'tif':
            name += '.tif'
        name = os.path.abspath(name)

        if isinstance(data, list):
            data = np.array(data)

        if data.ndim == 2:
            data = data[None]

        if (data.dtype == np.int64):
            if (data>=0).all() and (data<2**16).all():
                data = data.astype(np.uint16)
                print('Cast int64 to int16 without loss')
            else:
                data = data.astype(np.float32)
                print('Cast int64 to float32')
        if data.dtype == np.float64:
            data = data.astype(np.float32)
            print('Cast float64 to float32')

        if data.dtype == np.bool:
            data = 255 * data.astype(np.uint8)

        if not data.dtype in (np.uint8, np.uint16, np.float32):
            raise ValueError('Cannot save data of type %s' % data.dtype)

        nchannels = data.shape[-3]

        resolution = (1./resolution,)*2

        if luts is None:
            luts = DEFAULT_LUTS + (GRAY,) * nchannels

        if display_ranges is None:
            display_ranges = [None] * data.shape[-3]
        for i, dr in enumerate(display_ranges):
            if dr is None:
                x = data[..., i, :, :]
                display_ranges[i] = x.min(), x.max()

        try:
            luts = luts[:nchannels]
            display_ranges = display_ranges[:nchannels]
        except IndexError:
            raise IndexError('Must provide at least %d luts and display ranges' % nchannels)

        # metadata encoding LUTs and display ranges
        # see http://rsb.info.nih.gov/ij/developer/source/ij/io/TiffEncoder.java.html
        description = ij_description(data.shape)
        tag_50838 = ij_tag_50838(nchannels)
        tag_50839 = ij_tag_50839(luts, display_ranges)

        if not os.path.isdir(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))


        imsave(name, data, photometric='minisblack',
               description=description, resolution=resolution, compress=compress,
               extratags=[(50838, 'I', len(tag_50838), tag_50838, True),
                          (50839, 'B', len(tag_50839), tag_50839, True),
                         ])


def ij_description(shape):
    """Format ImageJ description for hyperstack.
    :param shape:
    :return:
    """
    s = shape[:-2]
    if not s:
        return imagej_description % (1, 1, 1, 1)
    n = np.prod(s)
    if len(s) == 3:
        return imagej_description % (n, s[2], s[1], s[0])
    if len(s) == 2:
        return imagej_description % (n, s[1], s[0], 1)
    if len(s) == 1:
        return imagej_description % (n, s[0], 1, 1)
    # bad shape
    assert False


def ij_tag_50838(nchannels):
    """ImageJ uses tag 50838 to indicate size of metadata elements (e.g., 768 bytes per ROI)
    :param nchannels:
    :return:
    """
    info_block = (20,)  # summary of metadata fields
    display_block = (16 * nchannels,)  # display range block
    luts_block = (256 * 3,) * nchannels  #
    return info_block + display_block + luts_block


def ij_tag_50839(luts, display_ranges):
    """ImageJ uses tag 50839 to store metadata. Only range and luts are implemented here.
    :param tuple luts: tuple of 255*3=768 8-bit ints specifying RGB, e.g., constants io.RED etc.
    :param tuple display_ranges: tuple of (min, max) pairs for each channel
    :return:
    """
    d = struct.pack('<' + 'd' * len(display_ranges) * 2, *[y for x in display_ranges for y in x])
    # insert display ranges
    tag = ''.join(['JIJI',
                   'gnar\x01\x00\x00\x00',
                   'stul%s\x00\x00\x00' % chr(len(luts)),
                   ]).encode('ascii') + d
    tag = struct.unpack('<' + 'B' * len(tag), tag)
    return tag + tuple(sum([list(x) for x in luts], []))
    