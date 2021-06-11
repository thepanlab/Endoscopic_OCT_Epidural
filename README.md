# README

Code for the paper to be published in a future paper. The following pieces of python code and jupyter notebooks were used for the paper. The following architectures were used: 
* Resnet 50
* InceptionV3
* Xception

# Dataset
The dataset used can be found in [[1]](#1)

# Prerequisites

The language used is Python. We used Tensorflow 2.3.

# Structure:
* `0a-Read_images.ipynb` <br>
    It process the images for tissues: fat, ligament, flavum ,and spinal cord from JPEG to numpy binary files

* `0b-Read_new_class.ipynb` <br>
    It process the images for epidural space (empty) from JPEG to numpy binary files

* `0c-Join_datasets.ipynb` <br>
    It concatenates the numpy binary files from 0a and 0b.

* `Inception_5cat/`:  Code for classification for 5 categories
    * `inception_5cat_batch/`
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging
    * `inception_5cat_python/`
        * `inceptionV3_arg.py`

* `ResNet50_5cat/`: Code for classification for 5 categories
    * `resnet50_5cat_batch/`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_5cat_python/`
        * `archResNet50_arg.py`
    
* `Xception_5cat/`:  Code for classification for 5 categories
    * `xception_5cat_batch/`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit used for debugging
    * `xception_5cat_python/`
        * `archXception_arg.py`

* `InceptionV3_binaries/`:  Code for binary classification 
    * `inceptionV3_bin1_batch/`:
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging
    * `inceptionV3_bin1_python/`
        * `archInceptionV3_arg.py`
    * `inceptionV3_bin2_batch/`
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging
    * `inceptionV3_bin2_python/`
        * `archInceptionV3_arg.py`
    * `inceptionV3_bin3b_batch/`
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging
    * `inceptionV3_bin3b_python/`
        * `archInceptionV3_arg.py`
    * `inceptionV3_bin4_batch/`
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging
    * `inceptionV3_bin4_python/`
        * `archInceptionV3_arg.py`

* `ResNet50_binaries/`:  Code for binary classification 
    * `resnet50_bin1_batch/`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin1_python/`
        * `archResNet50_arg.py`
    * `resnet50_bin2_batch/`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin2_python/`
        * `archResNet50_arg.py`
    * `resnet50_bin3b_batch/`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin3b_python/`
        * `archResNet50_arg.py`
    * `resnet50_bin4_batch/`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin4_python/`
        * `archResNet50_arg.py`

* `Xception_binaries/`:  Code for binary classification 
    * `xception_bin1_batch/`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit used for debugging
    * `xception_bin1_python/`
        * `archXception_arg.py`
    * `xception_bin2_batch/`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit used for debugging
    * `xception_bin2_python/`
        * `archXception_arg.py`
    * `xception_bin3b_batch/`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit used for debugging 
    * `xception_bin3b_python/`
        * `archXception_arg.py`
    * `xception_bin4_batch/`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit  used for debugging
    * `xception_bin4_python/`
        * `archXception_arg.py`

* `ResNet50_binaries_test/`:  Code for binary classification for cross-testing 
    * `resnet50_bin1_test_batch/`
        * `resnet50_arg_test_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin1_test_python/`
        * `archResNet50_arg_test.py`
    * `resnet50_bin2_test_batch/`
        * `resnet50_arg_test_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin2_test_python/`
        * `archResNet50_arg_test.py`
    * `resnet50_bin3b_test_batch/`
        * `resnet50_arg_test_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin3b_test_python/`
        * `archResNet50_arg_test.py`
    * `resnet50_bin4_test_batch/`
        * `resnet50_arg_test_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    * `resnet50_bin4_test_python/`
        * `archResNet50_arg_test.py`

For Binary models:
```
bin1: Fat vs Ligament
bin2: Ligament vs Flavum
bin3: Flavum vs Epidural space (aka empty) 
bin4: Epidural space vs Spinal cord
```

For Python files, for cross-validation they should be run like this:

```sh
archResNet50_arg.py test_subject val_subject
```
e.g
```sh
archResNet50_arg.py 1 2
```
For cross-testing, they should be run like this:

```sh
archResNet50_arg.py test_subject n_epochs
```
e.g
```sh
archResNet50_arg.py 1 7
```

# Video

https://user-images.githubusercontent.com/12533066/121708900-a1504280-ca9d-11eb-8d18-dd60dfb22ec1.mp4

*Slower version of video*

https://user-images.githubusercontent.com/12533066/121708911-a31a0600-ca9d-11eb-89f2-452bd1f26ace.mp4

# Paper
To be published

# References
<a id = "1">[1]</a>
Chen Wang, Qinggong Tang, & Nu Bao Tran Ton. (2021). Endoscopic OCT for epidural anesthesia [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4891265

# Contact

Paul Calle - pcallec@ou.edu <br>
Project link: https://github.com/thepanlab/OCT-Epidural
