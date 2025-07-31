import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import scipy
import skimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import io
from .shapes import make_shape_mask,clip_center,circle_deform,rectangle_array

def find_grid_dist(image,plot=False):
    """
    Determines the pixel calibration distance between lines of a calibration grid.
    Optionally plots proof of function.

    args:
        im (arr[N,M]) : A 2D array of an image of a calibration grid
        plot (bool)   : Whether to plot proof in-line

    returns: 
        float : pixels per grid unit calibration distance
    """
    if type(image)==str:
        image = np.asarray(Image.open(image)).sum(axis=-1)
    im = skimage.filters.gaussian(image, sigma=10)
    # filtering by dxdy sobel
    im = skimage.filters.sobel(im,axis=0)*skimage.filters.sobel(im,axis=1)
    # refiltering for magnitude
    im = skimage.filters.sobel(im)
    # blurring 
    im = skimage.filters.gaussian(im,sigma=10)
    im = im > (im.max() * 0.5)
    basins = skimage.measure.label(im)
    pts = []
    for i in range(basins.max()):
        # incrementing for boolean test reasons
        i += 1
        # Find centroid of each dot
        pts.append( np.asarray(np.where(basins == i)).mean(axis=-1) )
    pts = np.asarray(pts)
    # Find nearest neighbors of each point
    dists = scipy.spatial.distance_matrix(pts,pts)
    dists_sorted = np.sort(dists,axis=-1)
    # Diagnostic plotting as proof
    if plot:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(image,cmap='gray')
        ax.scatter(pts[:,1],pts[:,0],color='red',s=5)
        plt.show()
    return np.median(dists_sorted[:,1:3].flatten())

def norm_arr(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def normalize_to_percentiles(arr,lower,upper):
    ### TODO: Cleanup when proofed. Weird manually calculated quantiles from earlier. - TW 20250515
    # s_arr = np.sort(arr.flatten())
    # lower_val = s_arr[int(lower*len(s_arr))]
    # upper_val = s_arr[int(upper*len(s_arr))]
    
    # Casting to float
    arr = arr.astype(float)
    # Sanity checking values to make sure they fit in function
    lower = max(lower,0)
    upper = min(upper,1)

    # Getting quantile values
    lower_val = np.quantile(arr,lower)
    upper_val = np.quantile(arr,upper)
    
    # Normalizing to quantiles
    arr = (arr - lower_val) / (upper_val - lower_val)
    
    # Trimming values to [0,1] range
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    return arr

def estimate_hist_range(arr):
    # Getting histogram of image
    vals,bins = np.histogram(arr.flatten(),bins=np.arange(257))
    # Normalizing
    nvals = np.cumsum(vals) / vals.sum()
    # Taking derivative of normalized values
    dv = nvals[1:] - nvals[:-1]
    # Smoothing with gaussian filter
    sdv = scipy.ndimage.gaussian_filter1d(dv,sigma=5)
    # Getting watershed basins
    basins = skimage.segmentation.watershed(sdv)
    peaks = skimage.segmentation.watershed(1-sdv)

    mins = []
    for i in np.unique(basins):
        mins.append(sdv[basins == i].min())
    mins = np.asarray(mins)    
    minbasin = np.where(sdv == mins[np.where(mins < 5e-4)[0].min()])[0][0]
    maxbasin = np.where(sdv == mins[np.where(mins < 5e-4)[0].max()])[0][0]

    maxs = []
    for i in np.unique(peaks):
        maxs.append(sdv[peaks == i].max())
    maxs = np.asarray(maxs)    
    minpeak = np.where(sdv == np.sort(maxs)[-1])[0][0]
    maxpeak = np.where(sdv == np.sort(maxs)[-2])[0][0]
    
    return nvals[minbasin],nvals[maxpeak]

def fill_mask(arr,min_threshold):
    arr_fill = (arr > min_threshold).astype(int)
    i = 1
    while np.any(arr_fill == 1):
        i += 1
        seedpt = np.asarray(np.where(arr_fill == 1)).T[0]
        arr_fill = skimage.segmentation.flood_fill(arr_fill,(seedpt[0],seedpt[1]),i)
    return arr_fill

def remove_edgeclip(arr):
    mask = np.ones_like(arr)
    mask[1:-2,1:-2] = 0
    to_remove = np.unique(mask * arr)
    return np.invert(np.isin(arr,to_remove)) * arr

def border_mean(arr):
    return np.mean(np.concatenate([arr[:,0],arr[:,-1],arr[0,:],arr[-1,:]]))


def make_aligned_shape(pars,random_state):
    """
    TODO: document this, clean up, bring in line with shapes outputs
    """
    np.random.seed(random_state)
    arr = make_shape_mask(**pars['shape_pars'])
    margin = pars['hole_margin']
    shape_range = pars['hole_range']
    holes = pars['hole_count']
    for i in range(holes):
        # Generate random sized hole
        mask_size = np.random.randint(pars['hole_range'][0],pars['hole_range'][1])
        # Make negative boolean mask
        mask = make_shape_mask(**pars['hole_pars'],shape=mask_size,invert=False)
    
        # Find bounding box of main shape to try to meaningfully overlap random hole location
        x_idx = np.where(arr.sum(axis=0))[0]
        xmin = x_idx[0] - margin
        xmax = x_idx[-1] + margin - mask_size
        xind = np.random.randint(xmin,xmax)
        y_idx = np.where(arr.sum(axis=1))[0]
        ymin = y_idx[0] - margin
        ymax = y_idx[-1] + margin - mask_size
        yind = np.random.randint(ymin,ymax)
    
        # Take boolean difference in target region 
        subset = arr[xind:xind+mask_size,yind:yind+mask_size]
        subset = subset * mask
        arr[xind:xind+mask_size,yind:yind+mask_size] = subset
    # NEW: shape processing
    arr = clip_center(arr,pars)

def align_array_points(arr):
    """
    Rotates array and returns an aligned set of points.

    args:
        arr[N,M](bool) : A boolean array specifying a shape of interest

    returns:
        arr[N+n,M+m]   : Rotated array
        arr[N,2]       : Aligned, PCA-transformed point cloud of input array
    """
    idx = get_subsampled_points(arr).T
    pca = PCA(n_components=2).fit(idx)
    # pca_rt = vp.images.get_shape_pca(arr)
    arr = rotate_array(arr,pca,as_bool=True)
    # pc_stats = PCA(n_components=2)
    pts = pca.transform(idx)
    # Recalculating point distribution from rotated array.
    idx2 = get_subsampled_points(arr).T
    # Aligning point cloud to transformed index
    idxn = idx2 - idx2.mean(0)[None,...]
    alignment_code = score_overlap(idxn.T,pts.T).argmax()
    if alignment_code > 3:
        pts = np.flip(pts,axis=1)
    pts = pts*mirrors()[:,0,alignment_code%4][None,...]
    return pca,pts

def get_shape_pca(arr):
    """
    Gets principal component analysis of a boolean-parseable array for pointcloud alignment

    args:
        arr[N,M] : a boolean-parseable array

    returns:
        pca      : an sklearn principal component analysis object
    """
    pca = PCA(n_components=2)
    pca.fit(np.asarray(np.where(arr)).T)
    return pca

def rotate_array(arr, pca, as_bool = True):
    """
    Rotates array to align with pca fit. Optionally thresholds to parse as boolean.

    args:
        arr[N,M]       : a 2D numeric or boolean array
        pca            : a fitted sklearn PCA object
        as_bool (bool) : whether to output rotated arr as boolean

    returns:
        arr[N+n,M+m]   : a 2D numeric or boolean array
    """
    angle = np.arctan2(pca.components_[0,1], pca.components_[0,0])
    angle = np.degrees(angle)

    if as_bool:
        arr = arr.astype(float)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = scipy.ndimage.rotate(arr, -angle)
    if as_bool:
        arr = arr > 0.5
    return arr

def get_shape_out(pars, random_state=None, arr=None):
    
    # Getting shape mask and point cloud from generator
    if type(arr)==type(None):
        if pars['shape_method'] == 'rectangle_array':
            arr = rectangle_array(pars,random_state)
        else:
            arr = circle_deform(pars,random_state)
    
    if 'align_method' in pars.keys():
        if pars['align_method'] == 'pca':
            pca,pts = align_array_points(arr)
            arr = rotate_array(arr,pca,as_bool=True)
            # Scaling to unit distance
            unit_radius = np.sqrt((pts**2).sum(axis=1)).max()
            pts = pts / unit_radius
            # Transposing for later convenience
            pts = pts.T
        else:
            pts = None
    else:
        pca,pts = align_array_points(arr)
        arr = rotate_array(arr,pca,as_bool=True)
        # Scaling to unit distance
        unit_radius = np.sqrt((pts**2).sum(axis=1)).max()
        pts = pts / unit_radius
        # Transposing for later convenience
        pts = pts.T
    
    return dict(zip(['points','radius','array'],[pts,pars['obj_radius'],arr]))
    

def get_subsampled_points(arr, target_points = 1000):
    """
    Subsamples array to generate scaled pointcloud representing object.
    
    args: 
        arr (bool)[N,M]     : a boolean parseable array
        target_points (int) : minimum number of points for point cloud
        scale (float)       : conversion factor, pixels per unit of interest

    returns: 
        array (float)[2,N]    : Array of scaled 2D points from array
    """

    # Get subsampling factor estimated geometrically 
    subsample = int(arr.sum()**0.5 / target_points**0.5)
    if subsample < 1:
        subsample = 1
    # Creating calibration-adjusted point cloud of subsampled object
    return np.asarray(np.where(arr[::subsample,::subsample])).astype(float) * subsample
    
def process_image(image_name,pars):
    hsv = np.asarray(Image.open(image_name).convert('HSV')).astype(float)
    hsv[...,2] = 255 - hsv[...,2]
    coeffs = [2,1,1]
    kmeans = KMeans(n_clusters = 2, n_init = 10)
    kmeans.fit(hsv.reshape((np.prod(hsv.shape[:-1]),hsv.shape[-1])))
    # Assigns group as of interest based on distance from specified median group
    
    # ### TODO: Refine process of cluster selection
    # kgroup = scipy.spatial.distance_matrix( kmeans.cluster_centers_, np.asarray(pars['kmean_group'])[None,...] ).argmin()
    # # Labels pixels aligning with cluster of interest
    # kmask = np.asarray(kmeans.labels_).reshape(hsv.shape[:-1]) == kgroup
    ### TW 20241011: Swapping over to tophat method, should be more robust (hopefully)
    if not 'tophat' in pars.keys():
        pars['tophat'] = True
    if pars['tophat']:
        mask = skimage.morphology.white_tophat(hsv[...,1:].prod(-1),
                                                footprint=np.ones((int(pars['image_cal'] * 2),
                                                                   int(pars['image_cal'] * 2))))
    else:
        mask = hsv[...,2]
    mask = mask > (mask.max() * 0.5)
    mask = skimage.morphology.binary_dilation(skimage.morphology.binary_erosion(mask, 
                                                                                np.ones((int(pars['image_cal'] * 0.1),
                                                                                         int(pars['image_cal'] * 0.1)))),
                                                                                np.ones((int(pars['image_cal'] * 0.1),
                                                                                         int(pars['image_cal'] * 0.1))))
    coeffs = [2,1,1]
    arr = np.asarray([hsv[...,i]*coeffs[i] for i in range(3)]).sum(axis=0) * mask
    arr = (arr - arr.min())*255/(arr.max() - arr.min())
    
    arr = normalize_to_percentiles(arr,
                                   lower = 1 - 1.*mask.sum() / np.prod(mask.shape),
                                   upper = 1 - 0.05*mask.sum() / np.prod(mask.shape))
    basins = skimage.measure.label(mask)
    # May be buggy, removing for now - TW 2023-09-22
    # basins = remove_edgeclip(basins)
    n,c = np.unique(basins,return_counts=True)
    c = c[n!=0]
    n = n[n!=0]
    id = n[c.argmax()]
    obj = basins==id
    arr = obj*arr

    # Rotating image based on object alignment
    if 'align_method' in pars.keys():
        if pars['align_method'] == 'pca':
            pca,pts = align_array_points(obj)
            # Getting maximum radius from centroid
            unit_radius = np.sqrt((pts**2).sum(axis=1)).max()
            obj_radius = unit_radius / pars['image_cal']
            # Normalizing to unit
            pts = pts / unit_radius
            # Transposing for later convenience
            pts = pts.T
            arr = rotate_array(arr.astype(float),pca,as_bool=False)
        else:
            arr = arr.astype(float)
            pts = np.where(arr > (arr.max() - arr.min())*0.1)
            obj_radius = 1
    else:
        pca,pts = align_array_points(obj)
        # Getting maximum radius from centroid
        unit_radius = np.sqrt((pts**2).sum(axis=1)).max()
        obj_radius = unit_radius / pars['image_cal']
        # Normalizing to unit
        pts = pts / unit_radius
        # Transposing for later convenience
        pts = pts.T
        arr = rotate_array(arr.astype(float),pca,as_bool=False)
    # Normalizing image array
    arr[arr < 1e-6] = 0
    arr[arr > 1] = 1
    return dict(zip(['name','points','radius','array'],[image_name.split('/')[-1],pts,obj_radius,arr]))
    
def get_fit_pars_pca(img,outs):
    im_fit = {}
    overlap_score = []
    for out in outs:
        overlap_score.append(score_overlap(img['points'],out['points']))
    bestfit = np.asarray(overlap_score).argmax() // len(outs)
    im_fit['best_fit'] = bestfit
    im_fit['mirror_key'] = (np.asarray(overlap_score).argmax() - (len(outs)*bestfit)) % 4
    im_fit['transpose'] = (np.asarray(overlap_score).argmax() - (len(outs)*bestfit)) // 4
    im_fit['fit_score'] = np.asarray(overlap_score).max()
    return im_fit

def get_fit_pars_diff(img,outs):
    im_fit = {}
    adiff = np.abs(outs - img[None,None,...]).sum(axis=(-2,-1))
    asim = 1. - adiff / img.shape[-1]**2
    bestshape = asim.max(axis=1).argmax()
    align1 = asim[bestshape].argmax()
    im_fit['best_fit'] = bestshape
    im_fit['mirror_key'] = align1 // 360
    im_fit['transpose'] = None
    im_fit['fit_score'] = align1 % 360
    return im_fit
    

def centered_crop(arr):
    """
    args:
        arr (numeric, [N,M]): Array where higher values correlate to feature of interest. Assumes only one feature.
    returns:
        PIL.Image object corresponding to crop allowing for full range of rotation without cropping.

    TODO: Auto-pad image if bounding box exceeds image size.
    """
    arr = arr.astype(float)
    idx = np.array(np.where(arr > 0.1*(arr.max()-arr.min())))
    centroid = idx.mean(axis=1)
    scale = (((idx - centroid[...,None])**2).sum(axis=0)**0.5).max()
    # Pad array if necessary
    while np.any((centroid - scale) < 0) or np.any((centroid + scale) >= arr.shape):
        idx = np.array(np.where(arr > 0.1*(arr.max()-arr.min())))
        centroid = idx.mean(axis=1)
        scale = (((idx - centroid[...,None])**2).sum(axis=0)**0.5).max()
        pad_size = (np.array(arr.shape)*1.4).astype(int)
        pad_delta = (pad_size - arr.shape) // 2
        centroid = centroid + pad_delta
        pad_arr = np.zeros(pad_size).astype(arr.dtype)
        pad_arr[pad_delta[0]:pad_delta[0]+arr.shape[0],pad_delta[1]:pad_delta[1]+arr.shape[1]] = arr
        arr = pad_arr
    cropbox = np.concatenate([np.floor(centroid - scale).astype(int),np.ceil(centroid + scale).astype(int)])
    crop = arr[cropbox[0]:cropbox[2],cropbox[1]:cropbox[3]]
    crop = (255*(crop - crop.min()) / (crop.max() - crop.min())).astype(np.uint8)
    return Image.fromarray(crop,mode='L')
    

def score_overlap(pin,pout):    
    # TODO: Debug input. Should be calculated in process_images, but is failing due to an unknown scaling issue. Object buffering?
    mean_dist_in = np.sort(scipy.spatial.distance_matrix(pin.T,pin.T),axis=0)[1,:].mean()
    # Making mirrored and transposed permutations of point clouds
    mean_dist_out = np.sort(scipy.spatial.distance_matrix(pout.T,pout.T),axis=0)[1,:].mean()
    mout = get_point_permutations(pout)
    # Get distances between in and out point set
    dists = scipy.spatial.distance_matrix(pin.T,mout.reshape((2,np.prod(mout.shape[1:]))).T).reshape(pin.shape[1],mout.shape[1],8)
    # Get sum of non-overlapping points
    in_overlap = (dists.min(axis=1) < 1.5*mean_dist_in).sum(axis=0) / pin.shape[1]
    out_overlap = (dists.min(axis=0) < 1.5*mean_dist_out).sum(axis=0) / mout.shape[1]
    return (in_overlap + out_overlap) / 2
        

def get_point_permutations(points):
    """
    Returns 8 permutations of [2,N] point set, as based on flipping on each axis as well as transposing axes.

    args:
        points (arr)[2,N] : an array of points

    returns:
        array [2,N,8]     : array of permutations of points 
    """
    return np.concatenate([points[...,None] * mirrors(),np.flip(points[...,None] * mirrors(),axis=0)],axis=-1)

def mirrors():
    """
    Returns an array for mirroring permutations of a set of 2D points of shape [2,N,None]
    """
    return np.asarray([[1,1],[-1,1],[1,-1],[-1,-1]]).T[:,None,:]

def norm_to_uint8(arr):
    return (255*(arr.astype(float) - arr.astype(float).min()) / (arr.astype(float).max() - arr.astype(float).min())).astype(np.uint8)

    