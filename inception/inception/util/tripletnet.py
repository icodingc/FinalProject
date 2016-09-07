"""
copy from FaceNet rep.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
from scipy import misc
from tqdm import tqdm
from sklearn.utils import shuffle

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
    return dataset

def sample_random_people(dataset):
    # Create flattened dataset with image paths and labels
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
        
    # shuffle
    return shuffle(np.array(image_paths_flat),np.array(labels_flat))
  
    return image_paths, labels
def sample_class(dataset, people_per_batch, images_per_person):
    # 10 * 5000
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.max(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1.0/std_adj)
    return y  
def load_data(image_paths, do_random_flip, do_prewhiten=True):
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(image_paths[i])
        if do_prewhiten:
            img = prewhiten(img)
        img = flip(img, do_random_flip)
        img_list[i] = img
    images = np.stack(img_list)
    return images
def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float
def get_triplet_batch(triplets, batch_size, batch_index):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch
def select_triplets(embeddings, num_per_class, image_data, people_per_batch, alpha):

    def dist(emb1, emb2):
        x = np.square(np.subtract(emb1, emb2))
        return np.sum(x, 0)
  
    nrof_images = image_data.shape[0]
    nrof_triplets = nrof_images - people_per_batch
    shp = [nrof_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
    as_arr = np.zeros(shp)
    ps_arr = np.zeros(shp)
    ns_arr = np.zeros(shp)
    
    trip_idx = 0
    shuffle = np.arange(nrof_triplets)
    np.random.shuffle(shuffle)
    emb_start_idx = 0
    nrof_random_negs = 0
    for i in tqdm(xrange(people_per_batch)):
        #print('class %d' % (i))
        n = num_per_class[i]
        for j in xrange(1,n):
            a_idx = emb_start_idx
            p_idx = emb_start_idx + j
            as_arr[shuffle[trip_idx]] = image_data[a_idx]
            ps_arr[shuffle[trip_idx]] = image_data[p_idx]
      
            # Select a semi-hard negative that has a distance
            #  further away from the positive exemplar.
            pos_dist = dist(embeddings[a_idx][:], embeddings[p_idx][:])
            sel_neg_idx = emb_start_idx
            while sel_neg_idx>=emb_start_idx and sel_neg_idx<=emb_start_idx+n-1:
                sel_neg_idx = (np.random.randint(1, 2**32) % nrof_images) - 1
                #sel_neg_idx = np.random.random_integers(0, nrof_images-1)
            sel_neg_dist = dist(embeddings[a_idx][:], embeddings[sel_neg_idx][:])
      
            random_neg = True
            for k in range(nrof_images):
                if k<emb_start_idx or k>emb_start_idx+n-1:
                    neg_dist = dist(embeddings[a_idx][:], embeddings[k][:])
                    if pos_dist<neg_dist and neg_dist<sel_neg_dist and np.abs(pos_dist-neg_dist)<alpha:
                        random_neg = False
                        sel_neg_dist = neg_dist
                        sel_neg_idx = k
            
            if random_neg:
                nrof_random_negs += 1
              
            ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
            #print('Triplet %d: (%d, %d, %d), pos_dist=%2.3f, neg_dist=%2.3f, sel_neg_dist=%2.3f' % (trip_idx, a_idx, p_idx, sel_neg_idx, pos_dist, neg_dist, sel_neg_dist))
            trip_idx += 1
          
        emb_start_idx += n
    
    triplets = (as_arr, ps_arr, ns_arr)
    
    return triplets, nrof_random_negs, nrof_triplets
