import sys
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import sem
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import os
import pandas as pd
import time
import pickle

''' 
Callbacks 
- TimeHistory, 
- PrintValTrainRatioCallback, 
- LossAndErrorPrintingCallback,
- EarlyStopping (from keras)
'''
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"])
        )
''' 
Architectures
- ResNet50,
- InceptionResNet50
- Xception
'''
# ResNet50 model
def create_ResNet50_model(input_shape):
    #ResNet50
    base_model_empty = keras.applications.resnet50.ResNet50(
        include_top=False, 
        weights=None, 
        input_tensor=None, 
        input_shape=input_shape, 
        pooling=None
    )
    avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
    output = keras.layers.Dense(1, activation="linear")(avg)
    model_resnet50 = keras.models.Model(inputs=base_model_empty.input, outputs=output)
    return model_resnet50

def create_InceptionV3_model(input_shape, top='max'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    Layer = {
        'flatten': keras.layers.Flatten(),
        'avg': keras.layers.GlobalAveragePooling2D(),
        'max': keras.layers.GlobalMaxPooling2D()
    }[top]
    base = keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    x = Layer(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

def create_Xception_model(input_shape, top='max'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    Layer = {
        'flatten': keras.layers.Flatten(),
        'avg': keras.layers.GlobalAveragePooling2D(),
        'max': keras.layers.GlobalMaxPooling2D()
    }[top]
    base = keras.applications.Xception(input_shape=input_shape, include_top=False, weights=None)
    x = Layer(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

''' MAIN FUNCTION '''
if __name__ == '__main__':
    my_gpu = 0
    ########## Edit paths ########## 
    # edit paths as needed
    path2procdata =      "/home/jreynolds/21summer/epidural/processed_data/"
    export_path =        "/home/jreynolds/21summer/epidural/ct_cv_models/"
    this_model =         "A/"
    this_model_version = "2/" # EDIT ME !!!! 

    this_export_path = export_path+this_model+this_model_version
    # for EXCLUSION of zero-distance data == 24000 total images
    imp_images =         "export_nz_images_1D_20210607.npy"
    imp_imagefilenames = "export_nz_names_20210607.npy"
    imp_distances =      "export_nz_distances_20210607.npy"
    imp_eid =            "export_nz_eid_20210607.npy"
    # for INCLUSION of zero-distance data == 28800 total images
    #imp_images =         "export_wz_images_1D.npy"
    #imp_imagefilenames = "export_wz_names.npy"
    #imp_distances =      "export_wz_distances.npy"
    #imp_eid =            "export_wz_eid.npy"

    select_test_eids = [1, 2, 3, 4, 5, 6, 7, 8] # the subjects we choose for testing
    select_val_eids = [1, 2, 3, 4, 5, 6, 7, 8] # all subjects for cross-validation
    
    n_epochs = 20
    batch_size = 32
    my_metrics = ['mape', 'mae', 'mse']

    architecture_dict = {0: 'ResNet50', 1: 'InceptionV3', 2: 'Xception'}
    ##########################################

    image_shape = (681, 241, 1)

    # input images
    with open(path2procdata+imp_images, 'rb') as f:
        images_1D_list = np.load(f)
    # input image names
    with open(path2procdata+imp_imagefilenames, 'rb') as f:
        image_names_list = np.load(f)
    # input image distances (ground truth)
    with open(path2procdata+imp_distances, 'rb') as f:
        image_dist_list = np.load(f)
    # input EID
    with open(path2procdata+imp_eid, 'rb') as f:
        image_eid_list = np.load(f)
    # get list of unique EIDs
    eid_unique = np.unique(image_eid_list)

    # satisfying tf needs
    images_1D = images_1D_list[..., np.newaxis]

    # zip imported arrays
    data_mat = list(zip(image_names_list, image_eid_list, image_dist_list, images_1D))

    my_shape_count = 0
    for i in images_1D:
        if i.shape == image_shape:
            my_shape_count+=1

    print("----- About -----")
    print("n_epochs:       ", n_epochs)
    print("batch_size:     ", batch_size)
    print("Num images:     ", my_shape_count)
    print("Image shape:    ", image_shape)
    print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

    print("-----------------")
    total_fold_time=0
    global_count = 1
    # Start Cross-Testing
    for testeid in select_test_eids:
        start_time = time.time()
        # Get X_test and y_test
        test_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] == testeid]
        test_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] == testeid]
        test_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == testeid]
        # Get training images without validation set for best model in current fold
        all_train_imgs = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] in select_val_eids]
        all_train_dists = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] in select_val_eids]
        all_train_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] in select_val_eids]

        all_X_train = np.array(all_train_imgs)
        all_y_train = np.array(all_train_dists)

        unique_all_train_eid = np.unique(all_train_check)
        
        arch_mean_mape_dict = {}
        arch_sem_dict = {}
        arch_mean_mae_and_sem_dict = {}
        arch_mean_mape_and_sem_dict = {}
        # Loop through all the architectures for the current test-fold and perform CV with that arch. 
        print("\n--- start S%d test-fold ---" %testeid)
        for arch in architecture_dict:
            print("--- enter "+str(architecture_dict[arch])+"...")
            cv_results_list = []
            cv_mape_list = []
            cv_mae_list=[]
            cv_sem_list=[]
            cv_time_list = []

            # Cross-validation with current configuration in he current testing fold. 
            for valeid in select_val_eids:
                if valeid == testeid:
                    print("##### skipping using valeid(%d)==testeid(%d) #####" %(valeid, testeid))
                    continue

                # validation data
                val_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] == valeid]
                val_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] == valeid]
                val_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == valeid]
                # training data
                train_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] != valeid and data_mat[i][1] in select_val_eids]
                train_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] != valeid and data_mat[i][1] in select_val_eids]
                train_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testeid and data_mat[i][1] != valeid and data_mat[i][1] in select_val_eids]

                keras.backend.clear_session()
                np.random.seed(0)
                tf.random.set_seed(0)

                X_train = np.array(train_img_list)
                y_train = np.array(train_dist_list)
                X_val = np.array(val_img_list)
                y_val = np.array(val_dist_list)
                X_test = np.array(test_img_list)
                y_test = np.array(test_dist_list)

                # Confirming integrity of splits
                if len(X_train) != len(y_train):
                    print("ERROR - length mismatch len(X_train)=", len(X_train), ", len(y_train)=", len(y_train))
                    exit()
                if len(X_val) != len(y_val):
                    print("ERROR - length mismatch len(X_val)=", len(X_val), ", len(y_val)=", len(y_val))
                if len(X_test) != len(y_test):
                    print("ERROR - length mismatch len(X_test)=", len(X_test), ", len(y_test)=", len(y_test))
                    exit()

                unique_train_eid = np.unique(train_check)
                unique_val_eid = np.unique(val_check)
                unique_test_eid = np.unique(test_check)

                for i in range(len(train_check)):
                    if train_check[i] == testeid or train_check[i] == valeid:
                        print("ERROR - train set contamination, train_check[", i, "]=", train_check[i], " belongs elsewhere.")
                        exit()
                for i in range(len(val_check)):
                    if val_check[i] != valeid:
                        print("ERROR - validation set contamination, val_check[", i, "]=", val_check[i], " belongs elsewhere.")
                        exit()
                for i in range(len(test_check)):
                    if test_check[i] != testeid:
                        print("ERROR - validation set contamination, test_check[", i, "]=", test_check[i], " belongs elsewhere.")
                        exit()

                ### Get the appropriate model architecture  
                if arch == 0:
                    # ResNet50
                    print("* ResNet50 - S%d - V%d - #%d *" %(testeid, valeid, global_count))
                    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                    time_cb = TimeHistory()
                    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
                    model = create_ResNet50_model(image_shape)
                elif arch == 1:
                    # InceptionV3
                    print("* InceptionV3 - S%d - V%d - #%d *" %(testeid, valeid, global_count))
                    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                    time_cb = TimeHistory()
                    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
                    model = create_InceptionV3_model(image_shape)
                elif arch == 2:
                    # Xception
                    print("* Xception - S%d - V%d - #%d *" %(testeid, valeid, global_count))
                    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                    time_cb = TimeHistory()
                    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
                    model = create_Xception_model(image_shape)

                print("\t", len(X_train), "train images from subjects", unique_train_eid)
                print("\t", len(X_val), "val images from subject", unique_val_eid)
                print("\t", len(X_test), "test images from subject", unique_test_eid)

                # Compile the model
                model.compile(loss=keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=my_metrics)

                # Fit the model
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=batch_size, callbacks=[time_cb, early_stopping_cb])

                ########### Evaluate ###########
                times = time_cb.times
                print("--- Validate - S%d - V%d - %s ---:" %(testeid, valeid, architecture_dict[arch]))
                # store the validation error (mape)
                my_model_val = model.evaluate(X_val, y_val)
                # append validation score 
                cv_mape_list.append(my_model_val[1]) # mapes
                cv_mae_list.append(my_model_val[2]) # maes
                cv_time_list.append(times) 
                # delete the model for the next round
                del model
                global_count += 1
                #--------- END FOR valeid in select_val_eid_list ---------
            my_mean_mape = np.mean(cv_mape_list)
            my_mean_mae = np.mean(cv_mae_list)
            my_sem = sem(cv_mape_list)
            my_sem2 = sem(cv_mae_list)
            my_duo = (my_mean_mape, my_sem)
            my_duo2 = (my_mean_mae, my_sem2)
            
            arch_mean_mape_dict[arch] = my_mean_mape 
            arch_sem_dict[arch] = my_sem 
            
            arch_mean_mape_and_sem_dict[arch] = my_duo
            arch_mean_mae_and_sem_dict[arch] = my_duo2

            print("\n--- done with S%d %s ---" %(testeid, str(architecture_dict[arch])))
            print("arch_mean_MAPE_and_sem_dict:")
            print(arch_mean_mape_and_sem_dict)
            print("arch_mean_MAE_and_sem_dict:")
            print(arch_mean_mae_and_sem_dict)
            #--------- END FOR arch in architecture_dict ---------
        print("\n--- done with cross-validation in S%d ---" %testeid)
        outfile0 = open(this_export_path+"S"+str(testeid)+"_CV_MeanMAPEandSem_allArchs", "wb")
        pickle.dump(arch_mean_mape_and_sem_dict, outfile0)
        outfile0.close()

        outfile025 = open(this_export_path+"S"+str(testeid)+"_CV_MeanMAEandSem_allArchs", "wb")
        pickle.dump(arch_mean_mae_and_sem_dict, outfile025)
        outfile025.close()
    
        ####### NEW MODEL ######
        # Using the config (i.e. resnet50, inceptionV3, xception) with the
        # lowest average CV score in this testing fold. 
        best_arch_mean = 999999
        best_arch=''
        for i in arch_mean_mape_dict:
            if arch_mean_mape_dict[i] < best_arch_mean:
                best_arch_mean = arch_mean_mape_dict[i]
                best_arch = architecture_dict[i]
        print("--- finding best S%d configuration ---" %(testeid))
        print("arch_mean_dict:")
        print(arch_mean_mape_dict)
        print("WINNER -- best_arch=", best_arch, ", best_arch_mean=", best_arch_mean)
        print("--- training new %s model with all %d train images for S%d fold ---" %(str(best_arch), int(len(all_X_train)), testeid)) 
        # Train a new model based off the best of the models in the current testing fold. 
        if best_arch == architecture_dict[0]:
            print("ResNet50")
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_ResNet50_model(image_shape)
        elif best_arch == architecture_dict[1]:
            print("InceptionV3")
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_InceptionV3_model(image_shape)
        elif best_arch == architecture_dict[2]:
            print("Xception")
            early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_Xception_model(image_shape)
        
        print("* testeid=", testeid, " *")
        print("\t", len(all_X_train), "train images from subjects", unique_all_train_eid)
        print("\t", len(X_test), "test images from subject", unique_test_eid)
        
        # Compile the new model
        model.compile(loss=keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=my_metrics)

        # Fit the new model without validation data
        history = model.fit(all_X_train, all_y_train, epochs=n_epochs, batch_size=batch_size, callbacks=[time_cb, early_stopping_cb])
        times = time_cb.times
        print(f'total time for %d epochs is %.3f secs or %.3f mins' % (n_epochs, sum(times), sum(times)/60.0))
        print(f'average time per epoch is %.3f secs or %.3f mins' % (np.mean(times), np.mean(times)/60.0))

        print("--- test new model on unseen S%s ---" %str(testeid))
        test_eval = model.evaluate(X_test, y_test)
        print("\n")
        y_preds = model.predict(X_test)
        
        model.save(this_export_path+'model_S%s_%s.h5' %(str(testeid), str(architecture_dict[arch])))  # creates a HDF5 file 'my_model.h5'
               
        my_history = history.history
        hist_df = pd.DataFrame(history.history)
        print("\nTraining history of new model")
        print(hist_df)
        print("\nExporting results for fold", testeid)
        
        print("\ttraining history")
        outfile1 = open(this_export_path+"S"+str(testeid)+"_trainhist_"+str(best_arch), "wb")
        pickle.dump(my_history, outfile1)
        outfile1.close()
        
        print("\ty_preds")
        outfile2 = open(this_export_path+"S"+str(testeid)+"_y_preds_"+str(best_arch), "wb")
        pickle.dump(y_preds, outfile2)
        outfile2.close()

        print("\ty_test")
        outfile3 = open(this_export_path+"S"+str(testeid)+"_y_test_"+str(best_arch), "wb")
        pickle.dump(y_test, outfile3)
        outfile3.close()
        
        print("\ttimes")
        outfile4 = open(this_export_path+"S"+str(testeid)+"_times_"+str(best_arch), "wb")
        pickle.dump(times, outfile4)
        outfile4.close()
        
        print("\tX_test")
        outfile5 = open(this_export_path+"S"+str(testeid)+"_X_test_"+str(best_arch), "wb")
        pickle.dump(X_test, outfile5)
        outfile5.close()
        
        print("\ttest evaluation")
        outfile6 = open(this_export_path+"S"+str(testeid)+"_testEval_"+str(best_arch), "wb")
        pickle.dump(test_eval, outfile6)
        outfile6.close()
        
        total_fold_time = total_fold_time + (time.time() - start_time)
        print("#-#-#-#-#-#-# Done with S%s" %str(testeid))
        print(f'#-#-#-#-#-#-# time: %.2f seconds, %.2f mins, %.2f hrs' %(total_fold_time, total_fold_time/60.0, total_fold_time/3600.0))
        print("#-#-#-#-#-#-# best_arch:  ", best_arch)
        print("#-#-#-#-#-#-# test_score: ", test_eval[1], ", ", test_eval[2])
        print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
        # END FOR testeid in select_test_eids ------------
