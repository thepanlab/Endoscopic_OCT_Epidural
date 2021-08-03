import tensorflow as tf
from tensorflow import keras
from time import perf_counter
import pickle
import numpy as np

my_gpu = 1

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[my_gpu], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# 1. load test images (S7)
# 2. load model
# 3. select a subset of ~1000
# 4. start the timer
# 5. m.predict(X_test_temp, y_test_temp)
# 6. end the timer
# 7. divide the time by number of images in X_test_temp
path2cv='/home/jreynolds/21summer/epidural/ct_cv_models/'
model_dir='A/'
version='S7/'
this_import_path = path2cv+model_dir+version

infile3 = 'S7_X_test_InceptionV3'
#import X_test
f = open(this_import_path+infile3, 'rb')
X_test = pickle.load(f)
f.close()
import_path = '/home/jreynolds/21summer/epidural/ct_cv_models/A/S7/'
f=open(import_path+'S7_X_test_InceptionV3', 'rb')
X_test = pickle.load(f)
f.close()

print("X_test.shape = ", X_test.shape)
t_start_load = perf_counter()
m = keras.models.load_model("/home/jreynolds/21summer/epidural/ct_cv_models/A/S7/model_S7_Xception.h5")
t_end_load = perf_counter()

t_start_pred = perf_counter()
prediction_temp = m.predict(X_test[:1000][ ..., np.newaxis])

t_end_pred = perf_counter()
print("Prediction = ", prediction_temp)
time_to_load = t_end_load-t_start_load
time_to_predict = t_end_pred - t_start_pred
total_time = time_to_load + time_to_predict
print("Time loading model:           ", time_to_load, " seconds")
print("Prediction time (1000 images):", time_to_predict, " seconds")
print("Prediction time per image:    ", time_to_predict/1000, " seconds")
print("Total time:                   ", total_time, " seconds")
