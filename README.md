# README

Code for the paper to be published in a future paper. The following pieces of python code and jupyter notebooks were used for the paper. The following architectures were used: 
* Resnet 50
* InceptionV3
* Xception

# Prerequisites

The language used is Python. We used Tensorflow 2.3.

# Structure:
* `0a-Read_images.ipynb` <br>
    It process the images for tissues: fat, ligament, flavum ,and spinal cord from JPEG to numpy binary files

* `0b-Read_new_class.ipynb` <br>
    It process the images for epidural space (empty) from JPEG to numpy binary files

* `0c-Join_datasets.ipynb` <br>
    It concatenates the numpy binary files from 0a and 0b.

* `Inception_5cat`:  Code for classification of for categories
    * `nception_5cat_python`
        * `inceptionV3_arg.py`
    * `inception_5cat_batch`
        * `inceptionV3_arg_simult.batch`: batch file for Summit
        * `inceptionV3_arg_test.batch`: batch file for Summit used for debugging

* `ResNet50_5cat`: Code for classification for 5 categories
    * `resnet50_5cat_python`
        * `archResNet50_arg.py`
    * `resnet50_5cat_batch`
        * `resnet50_arg_simult.batch`: batch file for Summit
        * `resnet50_arg_test.batch`: batch file for Summit used for debugging
    
* `Xception_5cat`:  Code for classification for 5 categories
    * `xception_5cat_python`
        * `archXception_arg.py`
    * `xception_5cat_batch`
        * `xception_arg_simult.batch`: batch file for Summit
        * `xception_arg_test.batch`: batch file for Summit used for debugging

# Paper
To be published

# References

# Contact

Paul Calle - pcallec@ou.edu <br>
Project link: https://github.com/thepanlab/OCT-Epidural
