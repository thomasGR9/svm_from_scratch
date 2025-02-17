# Support Vector Machine (SVM) Implementation

This repository contains my custom implementation of a **Support Vector Machine (SVM)**, supporting both **Hard** and **Soft** margins. The model utilizes the **Kernel Trick** with support for **Polynomial** and **Gaussian RBF kernels**, allowing it to handle linearly inseparable data by transforming it into a higher-dimensional space. The implementation only has the OneVSOne method of handling multiclass problems (as the more memory efficient in SVM's) and supports saving the binary SVM's in the disk via the ‘MemoryProblem’ hyperparameter, which if set to True:

1)Each binary SVM, after its own fit, is stored in a .joblib file on disk instead of in a dictionary within the class.

2)The test set is split into 5 batches, and predictions (from all binary SVMs) are made on each batch separately and finally merged into the final numpy array. For the predictions, each binary SVM is loaded from disk to make its prediction.

## Repository Structure

- **`SVM.ipynb`**  
    A Jupyter notebook that builds and tests the functions used in the SVM class. It also contains visualization functions for plotting **2D SVMs** and calculating the **distance to the decision boundary**. Note that these visualization functions are currently not integrated into the final SVM class but are included as supplementary tools for analysis and debugging.

- **`SVM_class_2.py`**  
    The final **SVM class** implementation, which includes all dependent functions and necessary libraries. This file encapsulates the core logic and functionality required for training and evaluating SVM models using the **Hard or Soft margin** and selected **kernel**.
