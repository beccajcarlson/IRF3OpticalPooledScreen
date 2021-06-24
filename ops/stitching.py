import cv2
import numpy as np


# FUNCTIONS
def preprocess(img1, img2, overlap_w):

    shape = np.concatenate((img1,img2[:,:-overlap_w]),axis=1).shape
  
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    
    subA = np.zeros(shape)
    subB = np.zeros(shape)
    subA[:, :w1] = img1
         
    subB = np.zeros(shape)
    subB[:, w1 - overlap_w:] = img2
    
    mask = np.zeros(shape)
    mask[:, :w1 - int(overlap_w / 2)] = 1
    
    return subA, subB, mask

def preprocess_subset(img1, img2, overlap_w, extra_blend_w):

    shape = (img1.shape[0],extra_blend_w*2+overlap_w)
    
    subA = np.zeros(shape)
    subA[:, :-extra_blend_w] = img1[:, -(overlap_w + extra_blend_w):]
         
    subB = np.zeros(shape)
    subB[:, extra_blend_w:] = img2[:,:(overlap_w+extra_blend_w)]
    
    mask = np.zeros(shape)
    mask[:, :extra_blend_w + int(overlap_w / 2)] = 1
     
    return subA, subB, mask

def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP

def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP

def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended

def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img += lev_img
    return img

def multi_band_blending(img1, img2, overlap_w, leveln=2):
    # left-right blending of two images at right edge of img1, left edge img2 with given overlap_width
    # images do not need to be the same size
    
    if overlap_w < 0:
        print("error: overlap_w should be a positive integer")
        sys.exit()

    subA, subB, mask = preprocess(img1, img2, overlap_w)

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)

    return result

def multi_band_blending_subset(img1, img2, overlap_w, leveln=2, extra_blend_w = 100):
    # left-right blending of two images at right edge of img1, left edge img2 with given overlap_width
    # images do not need to be the same width
    # extra blend w refers to width in pixels on each side of overlap to be blended
    
    # only blend regions near overlap, then add back remaining part of image, unblended
    
    if overlap_w < 0:
        print("error: overlap_w should be a positive integer")
        sys.exit()

    subA, subB, mask = preprocess_subset(img1, img2, overlap_w, extra_blend_w)

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    left = img1[:,:-(overlap_w + extra_blend_w)]
    right = img2[:,(overlap_w + extra_blend_w):]

    
    result = np.concatenate((left, result, right),axis=1)
    
    return result


        
        ###
        ###
        