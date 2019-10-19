This is a TensorFlow implementation of the BO-Aug as described in our paper:
Zhang, C., Cui, J., Yang, B.: Learning optimal data augmentation policies via bayesian optimization for image classification tasks (2019)
The paper arxiv link is: https://arxiv.org/abs/1905.02610?context=stat.ML

/dataset:
This directory contains the datasets used in the experiment, including CIFAR and SVHN. It is worth noting that the data files given here are the data files required when searching the optimal DA policies in this experiment. Please download the complete data files yourself when performing the final test.

/CIFAR: (/SVHN is similar)
This directory contains all the code associated with the CIFAR data set. Specifically, it contains three subdirectories, "/select", "/reduced", and "all". The "/select" directory contains all the code needed to search the optimal DA policies. The "/reduced" directory contains all the code needed to test on the reduced dataset. The "/all" directory contains all the code needed to test on the complete dataset. 

Search optimal DA policies:
If you want to search for the optimal policies, please run the "BO.py". Note, however, that you need to import different header files for different data sets.

Test on the reduced dataset: (Take CIFAR dataset as an example, SVHN dataset is similar)
If you want to test the performance of the selected policies on the reduced dataset, then you need to switch to the "/reduced" directory and then run the "train_cifar.py". It should be noted that you need to assign the selected policies to the variable "policies" in the main function of the "train_cifar.py" file. The policies in the variable "policies" is now the optimal policies selected in the paper.

Test on the complete dataset: (Take CIFAR dataset as an example, SVHN dataset is similar)
If you want to test the performance of the selected policies on the complete dataset, then you need to switch to the "/all" directory and then run the "train_cifar.py". It should be noted that you need to assign the selected policies to the variable "policies" in the main function of the "train_cifar.py" file. The policies in the variable "policies" is now the optimal policies selected in the paper.
