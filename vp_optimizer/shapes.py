# from stl import mesh
import os
import shutil
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import meshlib
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from itertools import combinations
import io
import scipy
import skimage
import cv2

def make_shape_mask(n,r,shape,cardinality=None,invert=True):
    '''
    n (int)          :   Number of key vertices around unit circle
    r (float)[0,1]   :   Magnitude of perturbation
    shape (int)      :   x,y dimensions of image to make, in pixels
    invert (bool)    :   If true, produces False shape on True background
    '''
    shape = (shape,shape)
    if type(n) != int:
        n_use = np.random.randint(n[0],n[1])
    else:
        n_use = n
    if (type(r) != int) & (type(r) != float):
        r_use = np.random.random(1)[0]*(r[1]-r[0]) + r[0]
    else:
        r_use = r
    
    N = n_use*3+1 # number of points in the Path
    angles = np.linspace(0,2*np.pi,N)
    codes = np.full(N,Path.CURVE4)
    codes[0] = Path.MOVETO
    
    verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r_use*np.random.random(N)+1-r_use)[:,None]
    verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    path = Path(verts, codes)
    
    fig = plt.figure(figsize=shape,dpi=1)
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='k', lw=2)
    ax.add_patch(patch)
    
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off') # removes the axis to leave only the shape
    
    io_buf = io.BytesIO()
    plt.close(fig)
    fig.savefig(io_buf, format='raw', dpi=1)
    io_buf.seek(0)
    arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0].astype(bool)
    io_buf.close()
    if invert:
        arr = np.invert(arr)
    return arr
    
# def plotSTL(filename):
#     """
#     Renders a simple 3D plot of an STL file for debugging purposes.
    
#     args:
#         filename (str) : The filename+directory pointing to the STL to plot.
        
#     returns:
#         n/a. Plots inline.
#     """
    
#     # Create a new plot
#     figure = pyplot.figure()
#     axes = mplot3d.Axes3D(figure)

#     # Load the STL mesh
#     stlmesh = mesh.Mesh.from_file(filename)
#     polymesh = mplot3d.art3d.Poly3DCollection(stlmesh.vectors)

#     # Create light source
#     ls = LightSource(azdeg=225, altdeg=45)

#     # Darkest shadowed surface, in rgba
#     dk = np.array([0.2, 0.0, 0.0, 1])
#     # Brightest lit surface, in rgba
#     lt = np.array([0.7, 0.7, 1.0, 1])
#     # Interpolate between the two, based on face normal
#     shade = lambda s: (lt-dk) * s + dk

#     # Set face colors 
#     sns = ls.shade_normals(stlmesh.get_unit_normals(), fraction=1.0)
#     rgba = np.array([shade(s) for s in sns])
#     polymesh.set_facecolor(rgba)

#     axes.add_collection3d(polymesh)

#     # Adjust limits of axes to fill the mesh, but keep 1:1:1 aspect ratio
#     pts = stlmesh.points.reshape(-1,3)
#     ptp = max(np.ptp(pts, 0))/2
#     ctrs = [(min(pts[:,i]) + max(pts[:,i]))/2 for i in range(3)]
#     lims = [[ctrs[i] - ptp, ctrs[i] + ptp] for i in range(3)]
#     axes.auto_scale_xyz(*lims)

#     pyplot.show()

def overlapping_tris(arr,simplices,points,scan=1,threshold=2):
    """
    Finds all triangles who have significant positive feature map overlap, based on centroid proximity sum.

    args:
        arr (array)      : a 2d feature map where higher values correspond to positive features (ie image/mask).
        simplices (list) : a list of vertex index sets describing triangles.
        points (array)   : an [N,2] array of vertices falling within the shape of arr.
        scan (int)       : window size to scan around simplex centroid, must be >= 0.
        threshold (num)  : minimum value in scan window to accept simplex

    returns:
        list [N,3]: a list of triangle simplices that significantly overlap with feature map.
    """
    tris = []
    x = points[:,0]
    y = points[:,1]
    for simplex in simplices:
        cx = np.sum([x[i] for i in simplex])//3
        cy = np.sum([y[i] for i in simplex])//3
        check = np.sum(arr[cx-scan:cx+scan+1,cy-scan:cy+scan+1])
        if check >= threshold:
            tris.append(simplex)
    return np.asarray(tris)

def find_borders(edges,polys):
    """
    Finds all edges that are borders of a polygon set.
    
    args:
        edges (array,int) : An [N,2] array of vertex index pairs describing edges where N is number of edges.
        polys (array,int) : An [N,n] array of vertex index sets describing polygons where N is number of edges
                                and n is number of vertices per polygon.
    
    returns:
        array(int)[N,2]: An array of vertex index pairs of border edges.
    """
    
    borders = []
    for edge in edges:
        # Checks whether tris contain both vectors of an edge. 
        # If only 1 triangle in tris has edge, then it's a border edge.
        if (((polys == edge[0]).astype(int) + (polys == edge[1]).astype(int)).sum(axis=-1)==2).sum() == 1:
            borders.append(edge)
    
    return np.asarray(borders)

def find_borders_array(edges,tris):
    """
    Finds edges which 
    
    ## TODO: rebuild algorithm to work with only triangles input.
    Concept: 
    - Sort simplices
    - Get all permutations as tuples
    - Get unique tuple counts
    
    """
    # Build test array to compare all tris vs edges
    edgetest = np.repeat(tris[None,...],edges.shape[0],axis=0)
    # Find all tris containing both vectors of all edges
    borders = (edges[...,0][...,None,None] == edgetest) + (edges[...,1][...,None,None] == edgetest)
    # Tris that contain 2 true values contain an edge. Find number of tris that contain edge.
    # If only 1 tri contains an edge, that means it is a border edge.
    borders = (borders.sum(axis=-1)==2).sum(axis=-1) == 1
    
    return edges[borders,:]

def find_edges(tris):
    """
    Gets all edges present in a set of triangle simplices

    args:
        tris (arr) : An [N,3] array of vertex indices describing triangles.

    returns:
        arr(int)[N,2] : An array describing all edges of all triangles input.
    """
    edges = []
    for i,j in combinations(range(3),2):
        edges.append(np.concatenate([tris[:,i][...,None],tris[:,j][...,None]],axis=1))
    return np.concatenate(edges,axis=0)

def make_random_shape(pars,random_state):
    """
    TODO: document this
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
    return arr

# def arr_to_stl(arr,pars):
#     """
#     TODO: document, cleanup
#     """
#     pca_rt = PCA(n_components=2)
#     pca_rt.fit(np.asarray(np.where(arr)).T)
    
#     angle = np.arctan2(pca_rt.components_[0,1], pca_rt.components_[0,0])
#     angle = np.degrees(angle)
    
#     arr = scipy.ndimage.rotate(arr.astype(float), -angle) > 0.5
    
#     pc_stats = PCA(n_components=2)
#     pc_idx = np.asarray(np.where(arr)).T
#     pc_stats = pca_rt.fit_transform(pc_idx)
    
#     # Getting maximum distance from PCA centroid, transforming as unit radius
#     unit_radius = np.sqrt((pc_stats**2).sum(axis=1)).max()
#     pc_stats = pc_stats * pars['obj_radius'] / unit_radius

#     # Find relevant pixels with sobel dxdy
#     edgearr = scipy.ndimage.sobel(arr,0) * scipy.ndimage.sobel(arr,1) * arr
#     points = np.asarray(np.where(edgearr)).T
#     # Delaunay triangulation to generate polys
#     tri = scipy.spatial.Delaunay(points)
    
#     tris = overlapping_tris(arr,tri.simplices,points)
#     # Find borders to stitch 2D to 3D
#     edges = find_edges(tris)
#     borders = find_borders(edges,tris)

#     data = np.zeros(tris.shape[0]*2 + borders.shape[0]*2, dtype=mesh.Mesh.dtype)
    
#     verts = []
#     # Converting point indices to actual positions
#     for point in points:
#         idx = np.where((pc_idx[:,0] == point[0]) & (pc_idx[:,1] == point[1]))[0][0]
#         verts.append(pc_stats[idx])
#     verts = np.asarray(verts)
    
#     # Adding z element and doubling to simulate extrusion
#     verts = np.concatenate([np.concatenate([verts,np.ones(len(verts))[:,None]*pars['obj_thickness']/2],axis=-1),
#                             np.concatenate([verts,np.ones(len(verts))[:,None]*pars['obj_thickness']/-2],axis=-1)],
#                             axis=0)
    
#     faces = np.concatenate([tris,
#                             tris+len(points),
#                             np.concatenate([borders,borders[:,0][:,None]+len(points)],axis=-1),
#                             np.concatenate([borders+len(points),borders[:,1][:,None]],axis=-1)],
#                             axis=0)
    
#     data['vectors'] = verts[faces]
    
#     return mesh.Mesh(data)

def clip_center(arr,pars):
    px_radius = pars['working_diameter'] / 2 / pars['voxel_size']
    frame_size = pars['shape_pars']['shape']
    mask = ((np.indices((frame_size,frame_size)) - frame_size/2)**2).sum(axis=0)**0.5 < px_radius
    return arr*mask

def continuity_test(arr):
    return skimage.measure.label(arr).max() == 1

def circle_deform(pars,random_state):
    arr = make_random_shape(pars,random_state)
    arr = clip_center(arr,pars)
    return arr

def rectangle_array(pars,random_state):
    rng = np.random.default_rng(random_state)
    px_radius = pars['working_diameter'] / 2 / pars['voxel_size']
    min_width = pars['min_support'] / pars['voxel_size']
    centroid = pars['shape_pars']['shape']//2
    squares = []
    for i in range(pars['shape_pars']['n']):
        idx = np.array([-1000,-1000,-1000,-1000])
        while np.any((centroid - idx)**2 > px_radius**2) | np.any(np.abs(np.diff(idx,axis=0)) < min_width):
            start = rng.integers(centroid, centroid+px_radius, 2)[None,...]
            end = start - rng.integers(0, px_radius*2 * pars['shape_pars']['r'], 2)[None,...]
            idx = np.concatenate([start,end])
        idx.sort(axis=0)
        rotation = rng.integers(0,pars['shape_pars']['cardinality'])
        
        idx = rotate_point(idx,centroid,360*rotation/pars['shape_pars']['cardinality']).astype(int)
        idx.sort(axis=0)
        squares.append(idx)
    
    arr = np.zeros((pars['shape_pars']['shape'],pars['shape_pars']['shape']),dtype = bool)
    for idx in squares:
        arr[idx[0,0]:idx[1,0]+1,idx[0,1]:idx[1,1]+1] = True
    return arr

def rotate_point(v, origin, angle):
    angle = angle * np.pi / 180.0
    x = np.cos(angle) * (v[...,0]-origin) - np.sin(angle) * (v[...,1]-origin) + origin
    y = np.sin(angle) * (v[...,0]-origin) + np.cos(angle) * (v[...,1]-origin) + origin
    return np.array([x,y]).T

def min_thickness_test(arr,pars):
    # Todo: correct for kernel radius
    min_pixels = int(pars['min_thickness'] / 2 / pars['voxel_size'])
    dil_arr = cv2.erode(arr.astype(np.uint8),np.ones((min_pixels,min_pixels),dtype=np.uint8))
    return skimage.measure.label(np.invert(dil_arr.astype(bool))).max() == skimage.measure.label(np.invert(arr.astype(bool))).max()

def min_support_test(arr,pars):
    # Todo: correct for kernel radius
    min_pixels = int(pars['min_support'] / 2 / pars['voxel_size'])
    dil_arr = cv2.erode(arr.astype(np.uint8),np.ones((min_pixels,min_pixels),dtype=np.uint8))
    return skimage.measure.label(dil_arr.astype(bool)).max() == 1

class shape_generator:
    def __init__(self,pars,seed):
        self.pars = pars
        self.seed = seed