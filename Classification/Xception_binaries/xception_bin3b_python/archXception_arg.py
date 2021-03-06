#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:26:00 2020

@author: paulcalle
"""
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

from time import perf_counter 

from timeit import default_timer as timer

# tf.debugging.set_log_device_placement(True)

# class TimingCallback(keras.callbacks.Callback):
#     def __init__(self, logs={}):
#         self.logs=[]
#     def on_epoch_begin(self, epoch, logs={}):
#         self.starttime = perf_counter()
#     def on_epoch_end(self, epoch, logs={}):
#         self.logs.append(perf_counter()-self.starttime)

# only one value for paramameter
try:
    K_test = int(sys.argv[1])
    k_val = int(sys.argv[2])
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} <K_test>  <k_val>")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Reading inputs

with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_data/a_images_1D_5cat.npy', 'rb') as f:
    a_images_1D = np.load(f)
    
with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_data/a_label_5cat.npy', 'rb') as f:
    a_label = np.load(f)

with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_data/a_epidural_num_5cat.npy', 'rb') as f:
    a_epidural_num = np.load(f)

# Transforming labels to numerical categories

# a_label_num = np.copy(a_label)

# a_label_num[a_label_num == "fat"] = 0
# a_label_num[a_label_num == "flavum"] = 1
# a_label_num[a_label_num == "ligament"] = 2
# a_label_num[a_label_num == "spinalcord"] = 3

# a_label_num = a_label_num.astype(int)

# a_images_1D_float64 = a_images_1D.astype(float)

# a_images_3D = np.repeat(a_images_1D[..., np.newaxis], 3, -1) 
# images_resized = tf.image.resize_with_pad(a_images_3D, 224, 224, antialias=True)

## ResNet-34

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

from functools import partial  
    
import pickle

# a_kidneys_num = np.arange(1,11,1)
a_epidurals_num = np.unique(a_epidural_num)

a_selected_epidural_test = np.array([K_test])
# if all use : a_kidneys_num
# a_selected_kidneys_test = a_kidneys_num

print("a_selected_epidural_test = ", a_selected_epidural_test)

# a_selected_kidneys_val = np.array([4,7,9])
a_selected_epidural_val = np.array([k_val])
# if all use : a_kidneys_num
# a_selected_kidneys_val = a_kidneys_num

a_images_1D = a_images_1D[..., np.newaxis]

# flavum vs empty
bool_flavum_empty = np.logical_or(a_label == "flavum", a_label == "empty" )

a_label_num_bin_flavum_empty_pre = a_label[bool_flavum_empty]
a_label_num_bin_flavum_empty = np.copy(a_label_num_bin_flavum_empty_pre)

# Changing to numeric
a_label_num_bin_flavum_empty[a_label_num_bin_flavum_empty_pre == "flavum"] = 0
a_label_num_bin_flavum_empty[a_label_num_bin_flavum_empty_pre == "empty"] = 1
a_label_num_bin_flavum_empty = a_label_num_bin_flavum_empty.astype(int)

a_images_1D_bin_flavum_empty = a_images_1D[bool_flavum_empty]
a_epidural_num_bin_flavum_empty = a_epidural_num[bool_flavum_empty]

# for index in np.arange(1,11,1):
for index in a_selected_epidural_test:

    print("**Kidney test**: " + str(index) )

    a_epidural_num_val = np.delete(a_selected_epidural_val, np.where( a_selected_epidural_val == index))
    print("a_epidural_num_val = ", a_epidural_num_val)   
    
    bool_epidural_num = a_epidural_num_bin_flavum_empty != index
    a_images_1D_7_epidurals = a_images_1D_bin_flavum_empty[bool_epidural_num]
    a_label_num_7_epidurals = a_label_num_bin_flavum_empty[bool_epidural_num]
    a_epidural_num_7_epidurals = a_epidural_num_bin_flavum_empty[bool_epidural_num]
    
    print(len(a_images_1D_7_epidurals))
    print(len(a_label_num_7_epidurals))
    
    X_cv = a_images_1D_7_epidurals
    y_cv = a_label_num_7_epidurals
    
    n_epochs = 50
    # n_epochs = 1

    batch_size = 32

    # k fold validation  max:7 fold

    for index_val in a_epidural_num_val:

        print("Epidural_val: " + str(index_val) )
        
        bool_val_epidural = ( a_epidural_num_7_epidurals == index_val )
        bool_train_epidural = ~bool_val_epidural
        
        X_train_raw, X_val_raw = X_cv[bool_train_epidural], X_cv[bool_val_epidural]
        y_train, y_val = y_cv[bool_train_epidural], y_cv[bool_val_epidural]        
       
        # Preprocessing data
        # Substracting the mean pixel
        # (This is done normally per channel)
        mean_train_raw = np.mean(X_train_raw)
        
        X_train = X_train_raw - mean_train_raw
        X_val = X_val_raw - mean_train_raw
        
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        
        #ResNet50
        
        base_model_empty = keras.applications.Xception( include_top=False,
                                              weights=None,
                                              input_tensor=None,
                                              input_shape=(241,181,1),
                                              pooling=None)

        n_classes=2

        avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
        output = keras.layers.Dense(n_classes, activation="softmax")(avg)
        model_xception = keras.models.Model(inputs=base_model_empty.input, outputs=output)

        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov = True, decay=0.01)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True)

        # time_cb = TimingCallback()

        model_xception.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                      metrics=["accuracy"])

        t1_start = perf_counter() 

        history = model_xception.fit(X_train, y_train,
                               batch_size=batch_size,
                               validation_data=(X_val, y_val),
                               epochs=n_epochs,
                               callbacks=[early_stopping_cb])
        
        # RESNET 50

        t1_stop = perf_counter()
        time_lapse = t1_stop-t1_start

        model_xception.save("/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archXception_bin3b_results/model_outer_K%s_outer_k%s_val.h5"%(index, index_val))

        print( "Elapsed time during the whole program in seconds for K%s_outer_k%s_val: "%(index, index_val), time_lapse)
        
        with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archXception_bin3b_results/time_total_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, np.array(time_lapse))

        # with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archRESNET50_results/time_epoch_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
        #     np.save(f, np.array(time_cb.logs))
 
        y_proba = model_xception.predict(X_val)
                
        with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archXception_bin3b_results/pred_val_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, y_proba)
        
        with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archXception_bin3b_results/history_K%s_outer_k%s_val.pickle'%(index, index_val), 'wb') as handle:
            pickle.dump(history.history, handle)