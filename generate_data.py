#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
import os
import numpy as np
import random
import cv2

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb


# In[ ]:


def generator_batch(data_list, nbr_classes=3, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of (data, label).

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''

    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            if return_label:
                label = int(line[-1])
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                Y_batch[i - current_index, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if save_network_input:
            print('X_batch.shape: {}'.format(X_batch.shape))
            X_to_save = X_batch.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_to_save[:, :, 2], delimiter = ' ')

        img = X_batch[0,:,:,:]
        img = np.reshape(img, (-1))
        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch


# In[ ]:


def generator_batch_multitask(data_list, nbr_class_one=250, nbr_class_two=7, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):
    '''
    A generator that yields a batch of (data, class_one, class_two).

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y1_batch, Y2_batch)
    '''
    
    N = len(data_list)
    
    if shuffle:
        random.shuffle(data_list)
        
    batch_index = 0
    
    while True:
        current_index = (batch_index * batch_size) % N
        #when not get all the data, increate the batch_index
        if N > (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            #once get all data, reset batch_index and if need, shuffle data
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)
        
        #init the x_batch, y1_batch, y2_batch
        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y1_batch = np.zeros((current_batch_size, nbr_class_one))
        Y2_batch = np.zeros((current_batch_size, nbr_class_two))
        
        #now get each data in the batch 
        for i in range(current_index, (current_index + current_batch_size)):
            line = datalist[i].strip().split(' ')
            img_path = line[0]
            label_1 = line[1]
            label_2 = line[-1]
            
            #get image matrix
            if random_scale:
                scale_ratio = random.uniform(0.9,1.1)
            img_matrix = scale_byRatio(img_path, scale_ratio, img_width, crop_method)
            X_batch[i - current_index] = img_matrix
            Y1_batch[i - current_index, label_1] = 1
            Y2_batch[i - current_index, label_2] = 1
        
        #now alread get all batch data, and split it into X_batch, Y1_batch, Y2_batch
        #deal with the image data
        #1. need augment
        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        
        #if neet to save, save all the images in the batch
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_img_path = datalist[i].strip().split(' ')[0]
                base_dir = tmp_img_path.split(os.sep)[-2:]
                img_name = '_'.join(base_dir)
                img_save_path = os.path.join(save_to_dir, img_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_save_path)
        
        # if need preprocess the image data
        if preprocess:
            for i in range(current_batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image = True, result_size=(img_width, img_height))
        
        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)
        
        if return_label:
            yield (X_batch, Y1_batch, Y2_batch)
        else:
            yield X_batch        


# In[ ]:


def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of ([anchor, positive, negative], [class_one, class_two, pseudo_label]).

    Input:
        data_list  : a list of [img_path, vehicleID, modelID, colorID]
        dic_data_list: a dictionary: {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size
        mode       : generator used as for 'train', 'val'.
                     if mode is set 'train', dic_data_list has to be specified.
                     if mode is set 'val', dic_data_list could be a null dictionary: { }.
                     if mode is et 'feature_extraction', then return (X_anchor)


    Output:
        ([anchor, positive, negative], [class_one, class_two, pseudo_label]
    '''
    N = len(data_list)
    
    if shuffle:
        random.shuffle(data_list)
    
    batch_index = 0 
    pos_size = batch_size // 3
    while True:
        current_index = (batch_index * batch_size) % N
        if N > current_index:
            cur_batch_size = batch_size
            batch_index += 1
        else:
            cur_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)
                
        if  mode == 'feature_extraction':
            X_anchor = np.zeros((cur_batch_size, img_width, img_height, 3))
        else:
            simple_pos = 0
            hard_pos = 0
            ancor = np.zeros((cur_batch_size, img_width, img_height, 3))
            positive = np.zeros((cur_batch_size, img_width, img_height, 3))
            negative = np.zeros((cur_batch_size, img_width, img_height, 3))
            class_one = np.zeros((cur_batch_size, nbr_class_one))
            class_two = np.zeros((cur_batch_size, nbr_class_two))
            pseudo_label = np.zeros((cur_batch_size, 1))
            
        for i in (current_index, current_index + cur_batch_size):
            b_index = i - current_index
            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            if  mode == 'feature_extraction':
                X_anchor[b_index] = scale_byRatio(data_list[current_index][0], scale_ratio, img_width, crop_method)
            else:
                cur_data = data_list[current_index]
                vehicleID = cur_data[1]
                modelID = cur_data[2]
                colorID = cur_data[3]
                ancor[b_index]  = scale_byRatio(cur_data[0], scale_ratio, img_width, crop_method)
                class_one[b_index, modelID] = 1
                class_two[b_index, colorID] = 1
                if mode == 'train':
                    if cur_batch_size < batch_size:
                        vehicles_dict = dic_data_list[modelID][colorID]
                        if vehicles_dict:
                            vehicles = vehicles_dict.keys()
                            vehicle = vehicles[vehicles[random.randint(0, len(vehicles))]]
                            img_names = vehicles_dict[vehicle]
                            positive[b_index]  = scale_byRatio(img_names[random.randint(0, len(img_names))], scale_ratio, img_width, crop_method)
                        negative[b_index] = scale_byRatio(get_neg_img_name(dic_data_list, modelID, colorID, vehicleID, 2), scale_ratio, img_width, crop_method)
                    else:
                        if simple_pos < pos_size:
                            negative[b_index] = scale_byRatio(get_neg_img_name(dic_data_list, modelID, colorID, vehicleID, 0), scale_ratio, img_width, crop_method)
                            img_names = dic_data_list[modelID][colorID][vehicleID]
                            if img_names and len(img_names) > simple_pos:
                                positive[b_index]  = scale_byRatio(img_names[random.randint(0, len(img_names))], scale_ratio, img_width, crop_method)
                                simple_pos ++
                            else:
                                vehicles_dict = dic_data_list[modelID][colorID]
                                if vehicles_dict:
                                    vehicles = vehicles_dict.keys()
                                    vehicle = vehicles[vehicles[random.randint(0, len(vehicles))]]
                                    img_names = vehicles_dict[vehicle]
                                    positive[b_index]  = scale_byRatio(img_names[random.randint(0, len(img_names))], scale_ratio, img_width, crop_method)
                                    simple_pos ++
                                
                        else:
                            vehicles_dict = dic_data_list[modelID][colorID]
                            if vehicles_dict:
                                vehicles = vehicles_dict.keys()
                                vehicle = vehicles[vehicles[random.randint(0, len(vehicles))]]
                                img_names = vehicles_dict[vehicle]
                                positive[b_index]  = scale_byRatio(img_names[random.randint(0, len(img_names))], scale_ratio, img_width, crop_method)
                                hard_pos ++
                            if hard_pos < pos_size:
                                #get hard positive data and semi-hard negative data
                                negative[b_index] = scale_byRatio(get_neg_img_name(dic_data_list, modelID, colorID, vehicleID, 1), scale_ratio, img_width, crop_method)
                            else:
                                #get hard negative data
                                negative[b_index] = scale_byRatio(get_neg_img_name(dic_data_list, modelID, colorID, vehicleID, 2), scale_ratio, img_width, crop_method)
    
                else:
                    if current_index <= (N -3):
                        positive[b_index]  = scale_byRatio(data_list[current_index + 1][0], scale_ratio, img_width, crop_method)
                        negative[b_index]  = scale_byRatio(data_list[current_index + 2][0], scale_ratio, img_width, crop_method)
                    else if current_index == (N - 2) :
                        positive[b_index]  = scale_byRatio(data_list[current_index + 1][0], scale_ratio, img_width, crop_method)
                        negative[b_index]  = scale_byRatio(data_list[0][0], scale_ratio, img_width, crop_method)
                    else:
                        positive[b_index]  = scale_byRatio(data_list[0][0], scale_ratio, img_width, crop_method)
                        negative[b_index]  = scale_byRatio(data_list[1][0], scale_ratio, img_width, crop_method)
                
        if augment:
            if  mode == 'feature_extraction':
                X_anchor = X_anchor.astype(np.uint8)
                X_anchor = seq.augment_images(X_anchor)
            else:
                ancor = ancor.astype(np.uint8)
                positive = positive.astype(np.uint8)
                negative = negative.astype(np.uint8)
                anchor = seq.augment_images(anchor)
                positive = seq.augment_images(positive)
                negative = seq.augment_images(negative)
        
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_img_path = datalist[i][0]
                base_dir = tmp_img_path.split(os.sep)[-2:]
                img_name = '_'.join(base_dir)
                img_save_path = os.path.join(save_to_dir, img_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_save_path)   
        if save_network_input:
            #did smoe save action
        
        if  mode == 'feature_extraction':
                X_anchor = X_anchor.astype(np.float64)
                X_anchor = preprocess_input(X_anchor)
                yield X_anchor
            else:
                ancor = ancor.astype(np.float64)
                anchor = preprocess_input(anchor)
                positive = positive.astype(np.float64)
                positive = preprocess_input(positive)
                negative = negative.astype(np.float64)
                negative = preprocess_input(negative)
        
        if return_label:
            yield ([ancor, positive, negative], [class_one, class_two, pseudo_label])
        else:
            yield [ancor, positive, negative]         


# In[ ]:





# In[ ]:




