from time import time
import vp_optimizer as vp
import importlib
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from PIL import Image
from stl import mesh
import scipy
import vp_optimizer.shapes as shapes
import vp_optimizer.images as images
from alive_progress import alive_bar
from skimage.filters import gaussian


default_pars = {
                    "align_method": "diff",
                    "hole_count": 2,
                    "hole_margin": 0,
                    "hole_pars": {
                        "n": 4,
                        "r": 0.4
                    },
                    "hole_range": [
                        10,
                        60
                    ],
                    "kmean_group": [
                        155.40568757,
                        153.75507468,
                        111.35401764
                    ],
                    "mask_shape": "circle",
                    "min_diff": 0.25,
                    "min_support": 1.0,
                    "min_thickness": 0.5,
                    "obj_radius": 3,
                    "obj_thickness": 2,
                    "shape_method": "circle_deform",
                    "shape_pars": {
                        "n": 6,
                        "r": 0.7,
                        "cardinality":4,
                        "shape": 256
                    },
                    "skip_images": [],
                    "voxel_size": 0.04,
                    "working_diameter": 6.4
                }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_keys_to_int(d: dict):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = convert_keys_to_int(v)
        new_dict[new_key] = v
    return new_dict

class VPExperiment:
    def __init__(self,
                 base_dir = None,
                 expt_tag = None,
                 img_dir = 'images',
                 pars = None,
                 random_seed = None,
                 from_array = False,
                ):
        self.base_dir = base_dir
        self.from_array = from_array
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        self.img_dir = os.path.join(self.base_dir,img_dir)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        if pars == None:
            self.pars = default_pars
        elif pars == True:
            try_pars = [i for i in os.listdir(base_dir) if 'pars.json' in i]
            if len(try_pars)>0:
                with open(os.path.join(base_dir,try_pars[0]),'r') as f:
                    self.pars = json.load(f, object_hook = convert_keys_to_int)
        else:
            with open(pars,'r') as f:
                self.pars = json.load(f, object_hook = convert_keys_to_int)
        if expt_tag == True:
            self.pars['expt'] = [i for i in base_dir.split(os.path.sep) if i != ''][-1]            
        elif type(expt_tag) == str:
            self.pars['expt'] = expt_tag
        if (random_seed == None) & (self.pars['random_seed'] == None):
            self.pars['random_seed'] = int(time())
        if self.from_array:
            with open(os.path.join(base_dir,'arrays/model_out.npy'),'rb') as f:
                self.shape_arrs = np.load(f)

    def new_seed(self,seed = None):
        """
        Sets new random seed for shape generation.

        args:
            seed (int) (optional) : Random seed for numpy generator
        """
        if type(seed) != int:
            seed = int(time())
        self.pars['random_seed'] = seed
    
    def save_pars(self):
        """
        Saves experimental parameters as json file to base directory.
        """
        pars_name = ' '.join([str(i) for i in [self.pars['expt'],'pars.json'] if i != None])
        with open(os.path.join(self.base_dir, pars_name), 'w') as f:
            json.dump(obj = self.pars, fp = f, sort_keys = True, indent = 4, cls = NpEncoder)

    def generate_shapes(self,batch_size=8,new_seed=True):
        """
        Generates new set of shapes based on parameters and random seed. Shapes are validated for chance of cross-conformity by PCA overlap.

        args:
            batch_size (int) : Number of shapes to validate
            new_seed (bool)  : Whether to generate new seed when run
        """
        if new_seed:
            self.new_seed()
        rng = np.random.default_rng(self.pars['random_seed'])
        arrs = []
        ptcs = []
        min_dists = []
        seeds = []
        if self.pars['align_method'] == 'diff':
            with alive_bar(1, title='Batch generated') as bar:
                crops = []
                # Prelim 
                for i in range(batch_size):
                    seed = rng.integers(0,2**32,1)[0]
                    arr = self.get_shape(seed,i)
                    while not (shapes.continuity_test(arr) & shapes.min_thickness_test(arr,self.pars) & shapes.min_support_test(arr,self.pars)):
                        i = rng.integers(0,2**32,1)[0]
                        arr = self.get_shape(seed,i)
                    crop = images.centered_crop(arr)
                    crop = crop.resize((self.pars['rescale_size'],self.pars['rescale_size']),resample=Image.Resampling.LANCZOS)
                    arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC)) for i in np.arange(0,360,5)])
                    arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0)
                    crops.append(arrs)
                    seeds.append(seed)
                crops = np.array(crops).astype(float)
                # Normalizing
                crops_out = (crops - crops.min()) / (crops.max() - crops.min())
                ### TODO: Finish this thing
                fits = np.ones((batch_size,batch_size),dtype=float)
                # Setting initial difference score matrix. Lower means more similar.
                for i in range(batch_size):
                        crop_in = crops_out[i][0]
                        scores = np.abs(crops_out - crop_in[None,None,...]).sum((2,3)) 
                        scores /= (crops_out[:,0,...] + crop_in[None,...]).sum((1,2))[...,None]
                        fits[i] = scores.min(1) - self.pars['min_diff']
                while np.any((fits < 0).sum(0)>1):
                    
                    if np.all(fits == 0):
                        break
                    # Choosing best re-fit candidate by lowest difference score.
                    i = (fits * (fits < 0)).sum(0).argmin()
                    fit_old = (fits[i] * (fits[i] < 0)).sum()
                    fit_new = fit_old
                    while fit_new <= fit_old:
                        seed = rng.integers(0,2**32,1)[0]
                        arr = self.get_shape(seed,i)
                        while not (shapes.continuity_test(arr) & shapes.min_thickness_test(arr,self.pars) & shapes.min_support_test(arr,self.pars)):
                            seed = rng.integers(0,2**32,1)[0]
                            arr = self.get_shape(seed,i)
                        crop = images.centered_crop(arr)
                        crop = crop.resize((self.pars['rescale_size'],self.pars['rescale_size']),resample=Image.Resampling.LANCZOS)
                        crop_in = np.array(crop)
                        crop_in = (crop_in - crop_in.min()) / (crop_in.max() - crop_in.min())
                        scores = np.abs(crops_out - crop_in[None,None,...]).sum((2,3)) 
                        scores /= (crops_out[:,0,...] + crop_in[None,...]).sum((1,2))[...,None]
                        scores = scores.min(1) - self.pars['min_diff']
                        scores[i] = 0 - self.pars['min_diff']
                        fit_new = (scores * (scores < 0)).sum()
                    arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC)) for i in np.arange(0,360,5)])
                    arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0).astype(float)
                    arrs = (arrs - arrs.min()) / (arrs.max() - arrs.min())
                    crops_out[i] = arrs
                    fits[i] = scores
                    fits[:,i] = scores
                    seeds[i] = seed
                bar()

        else:
            with alive_bar(batch_size, title='Shapes generated') as bar:
                while len(arrs) < batch_size:
                    i = rng.integers(0,2**32,1)[0]
                    arr = self.get_shape(i,len(seeds))
                    if not shapes.continuity_test(arr):
                        continue
                    if not shapes.min_thickness_test(arr,self.pars):
                        continue
                    if not shapes.min_support_test(arr,self.pars):
                        continue
                    pca = PCA(n_components=2)
                    pts = pca.fit_transform(
                        images.get_subsampled_points(arr)
                        .T)
                    pts_r = ((pts**2).sum(axis=1)**0.5).max()
                    pts = pts / pts_r
                    ptc = images.get_point_permutations(pts.T)
                    
                    ptr = ptc.swapaxes(1,2).reshape(2,np.prod(ptc.shape[1:]))
                    
                    
                    dists = scipy.spatial.distance_matrix(pts,ptc.reshape((2,np.prod(ptc.shape[1:]))).T).reshape(pts.T.shape[1],ptc.shape[1],8)
                    
                    min_dist = np.sort(dists[...,0],axis=0)[1,:].max()
                    
                    diffs = ((np.sort(dists,axis=0)[0,...] > min_dist).sum(axis=0)/pts.shape[0] + (np.sort(dists,axis=1)[:,0,...] > min_dist).sum(axis=0)/pts.shape[0])/2
                    
                    if np.sort(diffs)[1] > self.pars['min_diff']:
                        if len(ptcs) > 0:
                            for p in range(len(ptcs)):
                                dists = scipy.spatial.distance_matrix(pts,ptc.reshape((2,np.prod(ptc.shape[1:]))).T).reshape(pts.T.shape[1],ptc.shape[1],8)
                                min_dist_c = np.max([min_dist,min_dists[p]])
                                diffs = ((np.sort(dists,axis=0)[0,...] > min_dist).sum(axis=0)/pts.shape[0] + 
                                         (np.sort(dists,axis=1)[:,0,...] > min_dist).sum(axis=0)/pts.shape[0]) / 2
                                if np.sort(diffs)[0] < self.pars['min_diff']:
                                    continue    
                        
                        arrs.append(arr)
                        ptcs.append(ptc)
                        min_dists.append(min_dist)
                        seeds.append(i)
                        bar()
        self.pars['shape_seeds'] = dict(zip(range(batch_size),seeds))

    def get_shape(self,seed,i=0):
        if self.pars['shape_method'] == 'circle_deform':
            return shapes.circle_deform(self.pars,seed)
        elif self.pars['shape_method'] == 'rectangle_array':
            return shapes.rectangle_array(self.pars,seed)
        elif self.pars['shape_method'] == 'mixed':
            if i % 2 == 0:
                return shapes.circle_deform(self.pars,seed)
            else:
                return shapes.rectangle_array(self.pars,seed)
            
    
    def get_shapes(self):
        shape_list = []
        for i,v in enumerate(self.pars['shape_seeds'].values()):
            shape_list.append(self.get_shape(v,i))
        return shape_list

    def process_images(self,cal = True):
        image_list = [i for i in os.listdir(self.img_dir) if 'ipynb' not in i] 
        
        if type(cal) == str:
            self.pars['image_cal'] = images.find_grid_dist(cal)
        elif any('cal' in s for s in image_list) & (cal == True):
            # Automatically process calibration image
            cal_image = image_list.pop(image_list.index([i for i in image_list if 'cal' in i][0]))
            self.pars['cal_image'] = cal_image
            self.pars['image_cal'] = images.find_grid_dist(os.path.join(self.img_dir,cal_image))
        elif (type(cal) == int) or (type(cal) == float):
            # Get input as specified number
            self.pars['image_cal'] = cal
        if 'image_cal' not in self.pars.keys():
            self.pars['image_cal'] = 1 / self.pars['voxel_size']
        image_list.sort()
        self.pars['image_list'] = dict(zip(range(len(image_list)),image_list))
        self.ins = []
        with alive_bar(len(image_list), title='Processing images') as bar: 
            for image_name in [os.path.join(self.img_dir,im) for im in image_list]:
                self.ins.append(images.process_image(image_name, self.pars))
                bar()
        shape_seeds = self.pars['shape_seeds'].values()
        self.outs = []
        with alive_bar(len(shape_seeds), title='Processing shapes') as bar:
            if self.from_array:
                for arr in self.shape_arrs:
                    self.outs.append(images.get_shape_out(self.pars,arr = arr))
                    bar()
            else:
                arrs = self.get_shapes() ### TW 2024-10-10 Patching this to account for mixed method
                for arr in arrs:
                    self.outs.append(images.get_shape_out(self.pars,arr = arr.astype(int)))
                    bar()

    def fit_images(self):
        if not hasattr(self,'image_fits'):
            self.pars['image_fits'] = {}
        if 'align_method' in self.pars.keys():
            if self.pars['align_method'] == 'diff':
                crops = []
                for out in self.outs:
                    arr = out['array']
                    crop = images.centered_crop(arr)
                    crop = crop.resize((self.pars['rescale_size'],self.pars['rescale_size']),resample=Image.Resampling.LANCZOS)
                    arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC)) for i in np.arange(0,360,1)])
                    arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0)
                    crops.append(arrs)
                crops = np.array(crops).astype(float)
                # Normalizing
                crops_out = (crops - crops.min()) / (crops.max() - crops.min())
        with alive_bar(len(self.ins), title='Aligning images') as bar:
            for i in range(len(self.ins)):
                if 'align_method' in self.pars.keys():
                    if self.pars['align_method'] == 'pca':
                        self.pars['image_fits'][i] = images.get_fit_pars_pca(self.ins[i],self.outs)
                    elif self.pars['align_method'] == 'diff':
                        crop_in = images.centered_crop(self.ins[i]['array'])
                        crop_in = np.array(crop_in.resize((self.pars['rescale_size'],self.pars['rescale_size']),resample=Image.Resampling.LANCZOS))
                        crop_in = crop_in.astype(float)
                        # Normalizing
                        crop_in = (crop_in - crop_in.min()) / (crop_in.max() - crop_in.min())
                        self.pars['image_fits'][i] = images.get_fit_pars_diff(crop_in,crops_out)
                bar()

    def shapes_to_voxels(self):
        arrs = self.get_shapes()
        arrstack = np.concatenate([np.zeros((1,256,256))] + 
                                  [np.concatenate([arr[None,...]]*50 + 
                                                  [np.zeros_like(arr)[None,...]]*25,axis=0) for arr in arrs],
                                  axis=0)
        arrstack = (arrstack.astype(bool).astype(np.uint8)*255)
        arrstack.tofile(os.path.join(self.base_dir,f'{self.pars["expt"]} voxels.dat'))

    def plot_alignments(self, adj_radius = False, save = False, export = False):
        """
        Plots pointclouds of generated shapes (blue) overlaid with processed image shapes (orange) to prove correct alignment.

        args:
            adj_radius (bool) : Multiplies shapes by actual radius rather than relative if True. Default False.
            save (bool)       : Saves figure in base_dir if True. Default False.
        """
        rows = int(np.ceil(len(self.ins)/4))
        fig,axs = plt.subplots(nrows=rows,ncols=4,figsize=(16,4*rows))
        for i in range(4*rows):
            if rows > 1:
                ax = axs[i//4][i%4]
            else:
                ax = axs[i]
            if i < len(self.pars['image_fits']):
                if 'align_method' in self.pars.keys():
                    if self.pars['align_method'] == 'pca':
                        pin = self.ins[i]['points']
                        pout = self.outs[self.pars['image_fits'][i]['best_fit']]['points']
                        if self.pars['image_fits'][i]['transpose'] == 1:
                            pout = np.flip(pout,axis=0)
                        pout = pout*images.mirrors()[...,self.pars['image_fits'][i]['mirror_key']]
                        if adj_radius:
                            pin = pin * self.ins[i]['radius']
                            pout = pout * self.outs[self.pars['image_fits'][i]['best_fit']]['radius']
                        ax.scatter(pout[0],pout[1],alpha=0.5)
                        ax.scatter(pin[0],pin[1],alpha=0.5)
                    elif self.pars['align_method'] == 'diff':
                        arr_in = self.ins[i]['array']
                        if self.pars['image_fits'][i]['mirror_key'] > 0:
                            arr_in = np.flip(arr_in, axis=-1)
                        crop_in = images.centered_crop(arr_in)
                        crop_in = crop_in.resize((self.pars['rescale_size']*4,self.pars['rescale_size']*4),resample=Image.Resampling.LANCZOS)
                        crop_in = crop_in.rotate(-self.pars['image_fits'][i]['fit_score'],Image.Resampling.BICUBIC)
                        crop_in = images.norm_to_uint8(np.array(crop_in))[...,None]
                        crop_out = images.centered_crop(self.outs[self.pars['image_fits'][i]['best_fit']]['array'])
                        crop_out = crop_out.resize((self.pars['rescale_size']*4,self.pars['rescale_size']*4),resample=Image.Resampling.LANCZOS)
                        crop_out = images.norm_to_uint8(np.array(crop_out))
                        im = np.concatenate([crop_in, crop_out[...,None], np.zeros_like(crop_in)], axis=-1)
                        ax.imshow(im)
                        if export:
                            im = Image.fromarray(im,mode='RGB')
                            if not os.path.isdir(os.path.join(self.base_dir,'alignments')):
                                os.mkdir(os.path.join(self.base_dir,'alignments'))
                            im.save(os.path.join(os.path.join(self.base_dir,'alignments'),self.pars['image_list'][i]),mode='RGB')

                        
                ax.set_title(i)
            ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.base_dir,f'{self.expt_tag} alignment plot.png'), bbox_inches='tight')
        else:
            plt.plot()

    def correct_alignment(self, image_number, shape_fit, mirror_key, transpose, fit_score):
        self.pars['image_fits'][image_number] = {'best_fit': shape_fit, 'mirror_key': mirror_key, 'transpose': transpose, 'fit_score': fit_score}
        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(4,4))
        if self.pars['align_method'] == 'pca':
            pin = self.ins[iimage_number]['points']
            pout = self.outs[self.pars['image_fits'][image_number]['best_fit']]['points']
            if self.pars['image_fits'][image_number]['transpose'] == 1:
                pout = np.flip(pout,axis=0)
            pout = pout*images.mirrors()[...,self.pars['image_fits'][image_number]['mirror_key']]
            if adj_radius:
                pin = pin * self.ins[image_number]['radius']
                pout = pout * self.outs[self.pars['image_fits'][image_number]['best_fit']]['radius']
            axs.scatter(pout[0],pout[1],alpha=0.5)
            axs.scatter(pin[0],pin[1],alpha=0.5)
        elif self.pars['align_method'] == 'diff':
            arr_in = self.ins[image_number]['array']
            if self.pars['image_fits'][image_number]['mirror_key'] > 0:
                arr_in = np.flip(arr_in, axis=-1)
            crop_in = images.centered_crop(arr_in)
            crop_in = crop_in.resize((self.pars['rescale_size']*4,self.pars['rescale_size']*4),resample=Image.Resampling.LANCZOS)
            crop_in = crop_in.rotate(-self.pars['image_fits'][image_number]['fit_score'],Image.Resampling.BICUBIC)
            crop_in = images.norm_to_uint8(np.array(crop_in))[...,None]
            crop_out = images.centered_crop(self.outs[self.pars['image_fits'][image_number]['best_fit']]['array'])
            crop_out = crop_out.resize((self.pars['rescale_size']*4,self.pars['rescale_size']*4),resample=Image.Resampling.LANCZOS)
            crop_out = images.norm_to_uint8(np.array(crop_out))
            im = np.concatenate([crop_in, crop_out[...,None], np.zeros_like(crop_in)], axis=-1)
            axs.imshow(im)
        axs.axis('off')
        plt.plot()

    def plot_shapes(self, save = False):
        if self.from_array:
            arrs = self.shape_arrs
        
        else:
            arrs = self.get_shapes()

        rows = int(np.ceil(len(arrs)/4))
        fig,axs = plt.subplots(nrows=rows,ncols=4,figsize=(8,2*rows))
        for i in range(4*rows):
            
            ax = axs[i//4][i%4]
            if i < len(arrs):
                ax.imshow(arrs[i],cmap='gray')
            ax.set_title(i)
            ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.base_dir,f'{self.expt_tag} shapes plot.png'), bbox_inches='tight')
        else:
            plt.plot()

    def skip_images(self,images=[]):
        """
        Defines image numbers to skip when exporting outputs.

        args:
            images (list) : A list of integers corresponding to image numbers as shown by plot_alignments.
        """
        if not 'skip_images' in self.pars.keys():
            self.pars['skip_images'] = []
        if type(images) != list:
            images = [images]
        for i in images:   
            self.pars['skip_images'].append(i)

    def export_alignments(self,threshold = 150):
        """
        Exports pngs of aligned shapes from input images (red) and source shape (green) to ./outputs directory
        """
        # Parameter to block in internal of image
        TRUE_IMAGE_THRESHOLD = threshold
        out_dir = os.path.join(self.base_dir,'outputs')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        with alive_bar(len(self.ins), title='Exporting data') as bar:
            for i in range(len(self.ins)):
                if i in self.pars['skip_images']:
                    continue
                px_size = self.pars['shape_pars']['shape']
                actual_size = self.pars['voxel_size'] * px_size
                scaled_size = int(self.pars['image_cal'] * actual_size)
                
                empty_in = np.zeros((scaled_size,scaled_size),dtype=float)
                empty_out = np.zeros((px_size,px_size),dtype=float)
                
                arr_in = self.ins[i]['array']
                arr_out = self.outs[self.pars['image_fits'][i]['best_fit']]['array']
                
                if self.pars['align_method'] == 'pca':
                    arr_in = arr_in - np.quantile(arr_in.flatten(),0.25)
                    in_idx = np.asarray(np.where(arr_in>0))
                    in_mins = in_idx.min(axis=1)
                    in_maxs = in_idx.max(axis=1)
                    in_transplant = arr_in[in_mins[0]:in_maxs[0],in_mins[1]:in_maxs[1]]
                    in_offset = (np.asarray(empty_in.shape) - np.asarray(in_transplant.shape))//2
                    empty_in[in_offset[0]:in_offset[0]+in_transplant.shape[0],in_offset[1]:in_offset[1]+in_transplant.shape[1]] = in_transplant
                    in_scaled = np.array(Image.fromarray(empty_in).resize((px_size,px_size)))
                    # Cleaning up floating point discrepancies
                    in_scaled[in_scaled<0] = 0
                    # Rescaling to fill in bulk of shape based on histogram analysis.
                    in_scaled = images.norm_to_uint8(in_scaled).astype(float)
                    in_scaled = in_scaled / TRUE_IMAGE_THRESHOLD
                    in_scaled[in_scaled>1.] = 1.
                    in_scaled = images.norm_to_uint8(in_scaled)
                    
                    out_idx = np.asarray(np.where(arr_out>arr_out.mean()))
                    out_mins = out_idx.min(axis=1)
                    out_maxs = out_idx.max(axis=1)
                    out_transplant = arr_out[out_mins[0]:out_maxs[0],out_mins[1]:out_maxs[1]]
                    out_offset = (np.asarray(empty_out.shape) - np.asarray(out_transplant.shape))//2
                    empty_out[out_offset[0]:out_offset[0]+out_transplant.shape[0],out_offset[1]:out_offset[1]+out_transplant.shape[1]] = out_transplant
                    if self.pars['image_fits'][i]['transpose']:
                        empty_out = empty_out.T
                    mirror_code = images.mirrors()[:,0,self.pars['image_fits'][i]['mirror_key']]
                    mirror_code = np.where(mirror_code < 0)[0]
                    if len(mirror_code) > 0:
                        empty_out = np.flip(empty_out,axis=mirror_code)
                    
                    image = np.concatenate([in_scaled[...,None],
                                            images.norm_to_uint8(empty_out)[...,None],
                                            np.zeros_like(empty_out).astype(np.uint8)[...,None]],axis=-1)
                elif self.pars['align_method'] == 'diff':
                    if self.pars['image_fits'][i]['mirror_key'] > 0:
                        arr_in = np.flip(arr_in, axis=-1)
                    crop_in = images.centered_crop(arr_in)
                    crop_in = crop_in.rotate(-self.pars['image_fits'][i]['fit_score'],Image.Resampling.BICUBIC)
                    crop_in = np.array(crop_in).astype(float)
                    in_offset = (np.asarray(empty_in.shape) - np.asarray(crop_in.shape))//2
                    empty_in[in_offset[0]:in_offset[0]+crop_in.shape[0],in_offset[1]:in_offset[1]+crop_in.shape[1]] = crop_in
                    in_scaled = np.array(Image.fromarray(empty_in).resize((px_size,px_size)))
                    # Cleaning up floating point discrepancies
                    in_scaled[in_scaled<0] = 0
                    #### Rescaling to fill in bulk of shape based on histogram analysis.
                    in_scaled = images.norm_to_uint8(in_scaled).astype(float)
                    ### Commented out for new method
                    in_scaled = in_scaled / TRUE_IMAGE_THRESHOLD
                    # distance = scipy.ndimage.distance_transform_edt(in_scaled>TRUE_IMAGE_THRESHOLD)#0.001) #Changed TW 20241103
                    # distmax = distance.max()
                    # bins = 50
                    # distance = (distance * bins / distance.max()).astype(int)
                    # idx = np.indices(np.array([i for i in distance.shape]+[bins]))[-1]
                    # idx = idx == distance[...,None]
                    # pivot = ((idx*in_scaled[...,None]).sum(axis=(0,1)) / (1+idx.sum(axis=(0,1)))).argmax()
                    # correction = (distance > pivot).astype(float)
                    
                    # correction = gaussian(correction,sigma=2*distmax/bins)*in_scaled.max() - in_scaled
                    # correction[correction < 0] = 0
                    # in_scaled = in_scaled + correction
                    in_scaled[in_scaled>1.] = 1.
                    in_scaled = images.norm_to_uint8(in_scaled)
                    
                    crop_out = images.centered_crop(arr_out)
                    # Patch to fix bug created after trying to fix centered_crop 2024-10-10
                    if np.any((np.array(empty_out.shape) - np.array(crop_out.size))<0):
                        crop_out = crop_out.resize(arr_out.shape)
                    crop_out = np.array(crop_out).astype(float)
                    # print(f'arr:{arr_out.shape},crop_out:{crop_out.shape},empty:{empty_out.shape}')
                    out_offset = (np.asarray(empty_out.shape) - np.asarray(crop_out.shape))//2
                    empty_out[out_offset[0]:out_offset[0]+crop_out.shape[0],out_offset[1]:out_offset[1]+crop_out.shape[1]] = crop_out
                    
                    image = np.concatenate([in_scaled[...,None],
                                            images.norm_to_uint8(empty_out)[...,None],
                                            np.zeros_like(empty_out).astype(np.uint8)[...,None]],axis=-1)
                
                Image.fromarray(image).save(os.path.join(out_dir,self.pars['image_list'][i]))
                bar()