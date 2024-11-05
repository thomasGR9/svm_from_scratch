# Support Vector Machine (SVM) Implementation

This repository contains my custom implementation of a **Support Vector Machine (SVM)**, supporting both **Hard** and **Soft** margins. The model utilizes the **Kernel Trick** with support for **Polynomial** and **Gaussian RBF kernels**, allowing it to handle linearly inseparable data by transforming it into a higher-dimensional space.

## Repository Structure

- **`SVM.ipynb`**  
    A Jupyter notebook that builds and tests the functions used in the SVM class. It also contains visualization functions for plotting **2D SVMs** and calculating the **distance to the decision boundary**. Note that these visualization functions are currently not integrated into the final SVM class but are included as supplementary tools for analysis and debugging.

- **`SVM_class.py`**  
    The final **SVM class** implementation, which includes all dependent functions and necessary libraries. This file encapsulates the core logic and functionality required for training and evaluating SVM models using the **Hard or Soft margin** and selected **kernel**.
