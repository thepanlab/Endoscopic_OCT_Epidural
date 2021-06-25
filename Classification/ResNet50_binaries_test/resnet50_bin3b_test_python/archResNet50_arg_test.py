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

# only one value for paramameter
try:
    K_test = int(sys.argv[1])
    n_epochs = int(sys.argv[2])

except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} <K_test> <n_epochs>")

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

# a_label_num[a_label_num == "medulla"] = 0
# a_label_num[a_label_num == "cortex"] = 1
# a_label_num[a_label_num == "pelvis_calyx"] = 2

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

a_images_1D = a_images_1D[..., np.newaxis]

# flavum vs spinalcord
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
    
    bool_epidural_train = a_epidural_num_bin_flavum_empty != index
    bool_epidural_test = ~bool_epidural_train
    
    X_train_raw, X_test_raw = a_images_1D_bin_flavum_empty[bool_epidural_train], a_images_1D_bin_flavum_empty[bool_epidural_test]
    y_train, y_test = a_label_num_bin_flavum_empty[bool_epidural_train], a_label_num_bin_flavum_empty[bool_epidural_test]  

    # X_train_raw, X_test_raw = a_images_1D[bool_kidney_train], a_images_1D[bool_kidney_test]
    # y_train, y_test = a_label_num[bool_kidney_train], a_label_num[bool_kidney_test]
  

    # print("Epidural_val: " + str(index_val) )
    
    # bool_val_epidural = ( a_epidural_num_7_epidurals == index_val )
    # bool_train_epidural = ~bool_val_epidural
    
    # X_train_raw, X_val_raw = X_cv[bool_train_epidural], X_cv[bool_val_epidural]
    # y_train, y_val = y_cv[bool_train_epidural], y_cv[bool_val_epidural]        
    
    # Preprocessing data
    # Substracting the mean pixel
    # (This is done normally per channel)
    mean_train_raw = np.mean(X_train_raw)
    
    X_train = X_train_raw - mean_train_raw
    X_test = X_test_raw - mean_train_raw

    # Preprocessing data
    # Substracting the mean pixel
    # (This is done normally per channel)
    # mean_train_raw = np.mean(X_train_raw)
    
    # X_train = X_train_raw - mean_train_raw
    # X_test = X_test_raw - mean_train_raw

    batch_size = 32

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)
       
    #ResNet50
    
    base_model_empty = keras.applications.resnet50.ResNet50( include_top=False,
                                            weights=None,
                                            input_tensor=None,
                                            input_shape=(241,181,1),
                                            pooling=None)

    n_classes=2

    avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(avg)
    model_resnet50 = keras.models.Model(inputs=base_model_empty.input, outputs=output)

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov = True, decay=0.01)

    model_resnet50.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

    t1_start = perf_counter() 

    history = model_resnet50.fit(X_train, y_train,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            epochs=n_epochs)
    
    model_resnet50.save("/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archRESNET50_bin3b_test_results/model_outer_K%s.h5"%(index))

    # RESNET 50

    t1_stop = perf_counter()
    time_lapse = t1_stop-t1_start

    print( "Elapsed time during the whole program in seconds for K%s_outer: "%(index), time_lapse)
    
    with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archRESNET50_bin3b_test_results/time_total_K%s_outer.npy'%(index), 'wb') as f:
        np.save(f, np.array(time_lapse))

    y_proba = model_resnet50.predict(X_test)
            
    with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archRESNET50_bin3b_test_results/pred_val_K%s_outer.npy'%(index), 'wb') as f:
        np.save(f, y_proba)
    
    with open('/gpfs/alpine/bif121/proj-shared/epidural/epidural_results/archRESNET50_bin3b_test_results/history_K%s_outer.pickle'%(index), 'wb') as handle:
        pickle.dump(history.history, handle)