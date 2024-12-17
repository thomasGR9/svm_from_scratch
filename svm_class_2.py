import numpy as np
import cvxopt #The library to solve the qp problem
from itertools import combinations #Used for the ovo strategy
from collections import Counter #Used for calculating the most frequent predicted class among the binary classifiers
from cvxopt import solvers, matrix, spmatrix
import joblib #Used for storing binary svms in disk if you have memory problems
import os #For deleting the joblib files that contain the binary svms
import scipy.sparse as sp

def calculate_matrixes_hard_margin(x, y, n):
    if len(np.unique(y))!=2:  #Accept only binary problems
        return
    q_matrix = np.array([-1 for i in range(n)]).reshape(n,1).astype(float) #The astype(float) seems to resolve an error with the solver
    G_matrix = np.multiply(np.identity(n), -1).astype(float)
    h_matrix = np.array([0 for i in range(n)]).astype(float)
    y_values = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).reshape((1,n)) #make the two classes -1 and 1 for the needs of the problem
    A_matrix = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).astype(float)
    b_matrix = 0.0
    dot_x_matrix = np.dot(x, np.transpose(x))
    dot_y_matrix = np.dot(np.transpose(y_values), y_values)
    p_matrix = np.multiply(dot_x_matrix, dot_y_matrix).astype(float) #The H matrix
    return p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values

def cvxopt_solve_qp(P, q, G, h, n, A=None, b=None):
    cvxopt.solvers.options['show_progress'] = False
    args = [cvxopt.matrix(P, (n, n), 'd'), cvxopt.matrix(q, (n,1), 'd')] #Specifying dimensions and 'd' type of matrix seems to solve some problems
    args.extend([cvxopt.matrix(G, (n, n), 'd'), cvxopt.matrix(h, (n,1), 'd')])
    if A is not None:
        args.extend([cvxopt.matrix(A, (1, n), 'd'), cvxopt.matrix(b, (1, 1), 'd')])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']: #Accept only converged solutions
        return None
    print("Optimal solution found")
    return np.array(sol['x']).reshape((P.shape[1],))  #return the matrix with the values of ai's

def calculate_w_b(a,x,y):
    w = np.dot((y*a).T, x)
    S = (a > 1e-5).flatten() #criterion to set the support vectors.If the alpha value is greater than 1e-5.
    b = np.mean(y[S] - np.dot(x[S], w.T)) #Mean value of b for all support vectors
    return w, b

def calculate_matrixes_soft_margin(x, y, C, n):
    if len(np.unique(y))!=2:
        raise ValueError("Not binary.")
    unique_y = np.unique(y)
    class_0 = unique_y[0]
    q_matrix = np.full((n, 1), -1.0, dtype=float)
    negative_identity = -sp.identity(n, format='csr')
    positive_identity = sp.identity(n, format='csr')
    G_matrix = sp.vstack([negative_identity, positive_identity])  #(2n,n) matrix, is like having two identity matrixes ,one with -1 and one with 1, with the first on top of the second
    h_matrix = np.concatenate([np.zeros(n), np.full(n, C)], axis=0).astype(float) #(2n, 1) matrix with the value 0 for [0,n] and C for [n,2n]
    y_values = np.where(y == class_0, -1, 1).reshape(1, n)
    A_matrix = np.where(y == class_0, -1, 1).astype(float)
    b_matrix = 0.0
    dot_x_matrix = np.dot(x, x.T)
    dot_y_matrix = np.dot(y_values.T, y_values)
    p_matrix = np.multiply(dot_x_matrix, dot_y_matrix).astype(float)
    return p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP

def cvxopt_solve_qp_soft_margin(P, q, G, h, n, A=None, b=None):
    cvxopt.solvers.options['show_progress'] = False
    args = [cvxopt.matrix(P, (n, n), 'd'), cvxopt.matrix(q, (n,1), 'd')]
    args.extend([scipy_sparse_to_spmatrix(G), cvxopt.matrix(h, (2*n,1), 'd')]) #change the dimensions of G and h matrixes
    if A is not None:
        args.extend([cvxopt.matrix(A, (1, n), 'd'), cvxopt.matrix(b, (1, 1), 'd')])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']: 
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def calculate_w_b_soft_margin(a,x,y):
    S = (a > 1e-5).flatten()
    w = np.dot((y[S]*a[S]).T, x[S])
    b = np.mean(y[S] - np.dot(x[S], w.T))
    return w, b



def rbf_matrix_fit(x, gamma):
    #This will split the matrix (of shape (x,x)) on 9 blocks and calculate the rbf kernel for the 6 lower half blocks with iterations.
    # On each iteration it will copy the transposed block on the mirrored one using the fact that the matrix is symmetric
    N = x.shape[0]
    if N <= 2000:
        block_size = max(1, N // 3) #ensure that is one when x<3. The 9 blocks i found ran faster on the iris dataset, so i picked it
    else:
        block_size = max(1, N // 6) #If N is large use smaller block size to reduce peak memory consumption
    result = np.zeros((N, N), dtype=np.float32)
    x = x.astype(np.float32)  # Convert inputs to float32
    gamma = np.float32(gamma)
    for i in range(0, N, block_size):
        for j in range(i, N, block_size):
            # Compute block indices
            x_block_i = x[i:i+block_size]
            x_block_j = x[j:j+block_size]
            # Compute pairwise differences for the current block
            diff = x_block_i[:, np.newaxis, :] - x_block_j[np.newaxis, :, :] #Vectorized calculations of differences on this block
            squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)
            exp_block = np.exp(-gamma * squared_diff, dtype=np.float32)
            # Assign to the result matrix
            result[i:i+block_size, j:j+block_size] = exp_block
            if i != j:  # Mirror for symmetric positions
                result[j:j+block_size, i:i+block_size] = exp_block.T
    return result

def kernelized_dot_product_fit(x1, x2, kernel, d, gamma):
    if kernel == "Linear":
        dot_matrix = np.dot(x1, x2.T)
        return dot_matrix
    elif kernel == "Polynomial":
        dot_matrix = (np.dot(x1, x2.T) + 1)**d #Polynomial kernel of power d
        return dot_matrix
    elif kernel == "RBF":
        dot_matrix = rbf_matrix_fit(gamma=gamma, x=x1)
        return dot_matrix.astype(np.float64) #Convert to float64 again to be compatible with cvxopt
    
def calculate_matrixes_soft_margin_kernel(x, y, C, kernel, d, gamma, n):
    if len(np.unique(y))!=2:
        raise ValueError("Not binary.")
    class_0 = np.unique(y)[0]
    q_matrix = np.full((n, 1), -1.0).astype(float)
    negative_identity = -sp.identity(n, format='csr')
    positive_identity = sp.identity(n, format='csr')
    G_matrix = sp.vstack([negative_identity, positive_identity]) 
    h_matrix = np.array([0 if i<n else C for i in range(2*n)]).astype(float)
    y_values = np.where(y == class_0, -1, 1).reshape((1,n))
    A_matrix = np.where(y == class_0, -1, 1).astype(float)
    b_matrix = 0.0
    dot_x_matrix = kernelized_dot_product_fit(x1=x, x2=x, kernel=kernel, d=d, gamma=gamma) #The only matrix that changes.Instead of simple dot product of each point with another, will be the kernel function of each.
    dot_y_matrix = np.dot(y_values.T, y_values)
    p_matrix = np.multiply(dot_x_matrix, dot_y_matrix).astype(float)
    return p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values

def calculate_b_kernelized(x, kernel, d, gamma, aiyi, y):
    #will use the kernelized_dot_product to calculate b with the kernel trick
    kernelized_xixj = kernelized_dot_product_fit(x1=x, x2=x, kernel=kernel, d=d, gamma=gamma)
    ajyjxjxi = kernelized_xixj * aiyi[np.newaxis, :]
    sum_of_all = sum(ajyjxjxi)
    bi=y - sum_of_all
    b = np.mean(bi) 
    return b
    
def make_preds_linear_forclass(x, w, b, neg_class, pos_class): 
    y_preds = np.where(np.dot(x, w.T) + b < 0, neg_class, pos_class) 
    return y_preds


def rbf_pred_func(x_supp_vect, x_test, gamma):
    M, N = x_supp_vect.shape
    m = x_test.shape[0]

    x_supp_vect = x_supp_vect.astype(np.float32)  # Convert inputs to float32
    x_test = x_test.astype(np.float32)
    gamma = np.float32(gamma)

    final_result = np.zeros((M, m), dtype=np.float32) 

    chunk_size = max(1, m // 3)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        x_test_chunk = x_test[start:end, :].astype(np.float32) 

        
        diff = x_supp_vect[:, np.newaxis, :] - x_test_chunk[np.newaxis, :, :]  # Shape (M, chunksize, N)
        squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)  # Sum along the N dimension to get (M, chunksize)
        final_result[:, start:end] = np.exp(-gamma*squared_diff, dtype=np.float32)

    return final_result.astype(np.float64) #Convert back to float64 to avoid mixed presicion calculations
    
    

def make_preds_soft_margin_kernel_forclass(x_support_vect, x_pred, kernel, d, gamma, b, aiyi, neg_class, pos_class): 
    if kernel == "RBF":
        kernelized_xixtesti = rbf_pred_func(x_supp_vect=x_support_vect, x_test=x_pred, gamma=gamma)
    else:
        kernelized_xixtesti = kernelized_dot_product_fit(x1=x_support_vect, x2=x_pred, kernel=kernel, d=d, gamma=gamma) # Shape: (Number of supp vect X number of test samples)
    sum_for_every_sup_vector = np.sum(aiyi[:, np.newaxis]*kernelized_xixtesti, axis=0, keepdims=True) #Shape: (1 X number of test samples)
    decisions = sum_for_every_sup_vector+b #Predictions .Shape: (1 X number of test samples) 
    predictions = np.where(decisions.flatten() < 0, neg_class, pos_class)
    return predictions




class svm:
    def __init__(self, margin_type="Soft", C=None, kernel=None, gamma=None, d=None):
        
        self.margin_type = margin_type
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.d = d

        #Atributes learned after fit
        self.x_support_vectors_ = None
        self.aiyi_ = None
        self.w_ = None
        self.b_ = None
        self.neg_class_ = None
        self.pos_class_ = None
        
        

    def fit(self, x, y):
        if len(np.unique(y)) != 2:
            raise ValueError("Problem not binary")
        n = x.shape[0]
        self.neg_class_ = np.unique(y)[0] 
        self.pos_class_ = np.unique(y)[1]
        if self.margin_type == 'Hard' and self.kernel == 'Linear':
             p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, self.y_values_ = calculate_matrixes_hard_margin(x=x, y=y, n=n)
             self.alphas_matrix_ = cvxopt_solve_qp(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, n=n, A=A_matrix, b=b_matrix)
             self.w_, self.b_ = calculate_w_b(a=self.alphas_matrix_,x=x,y=self.y_values_.reshape(n))
        elif (self.margin_type == 'Soft') and (self.kernel == 'Linear'):
            p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values_ = calculate_matrixes_soft_margin(x=x, y=y, C=self.C, n=n)
            alphas_matrix_ = cvxopt_solve_qp_soft_margin(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, n=n, A=A_matrix, b=b_matrix)
            self.w_, self.b_ = calculate_w_b_soft_margin(a=alphas_matrix_,x=x,y=y_values_.reshape(n))
        elif (self.margin_type == 'Soft') and (self.kernel in ['RBF', 'Polynomial']):
            p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values_ = calculate_matrixes_soft_margin_kernel(x=x, y=y, C=self.C, kernel=self.kernel, d=self.d, gamma=self.gamma, n=n)
            alphas_matrix_ = cvxopt_solve_qp_soft_margin(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, n=n, A=A_matrix, b=b_matrix)
            S = (alphas_matrix_ > 1e-5).flatten()
            self.x_support_vectors_ = x[S]
            yi = y_values_.flatten()[S]
            self.aiyi_ = alphas_matrix_[S]*yi
            self.b_ = calculate_b_kernelized(x=self.x_support_vectors_, kernel=self.kernel, d=self.d, gamma=self.gamma, aiyi=self.aiyi_, y=yi)
        else:
            raise ValueError("Invalid configuration for margin type or kernel.")
        

    def predict(self, x_test):
        if self.margin_type == 'Hard' and self.kernel == 'Linear':
            y_preds = make_preds_linear_forclass(x=x_test, w=self.w_, b=self.b_, neg_class=self.neg_class_, pos_class=self.pos_class_)
            return y_preds
        elif (self.margin_type == 'Soft') and (self.kernel == 'Linear'):
            y_preds = make_preds_linear_forclass(x=x_test, w=self.w_, b=self.b_, neg_class=self.neg_class_, pos_class=self.pos_class_)
            return y_preds
        elif (self.margin_type == 'Soft') and (self.kernel in ['RBF', 'Polynomial']):
            y_preds = make_preds_soft_margin_kernel_forclass(x_support_vect=self.x_support_vectors_, x_pred=x_test, kernel=self.kernel, d=self.d, gamma=self.gamma, b=self.b_,aiyi=self.aiyi_ , neg_class=self.neg_class_, pos_class=self.pos_class_)
            return y_preds
        else:
            raise ValueError("Invalid configuration for margin type or kernel.")

    def get_params(self, deep=True):
        # To be able to use the clone() function on the k-fold validation 
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'd': self.d,
        }
        
        
class OvOSVM:
    def __init__(self, margin_type="Soft", C=None, kernel=None, gamma=None, d=None, MemoryProblem=False):
        
        if margin_type not in ["Hard", "Soft"]:
            raise ValueError("margin_type must be a str with values ('Hard' or 'Soft')")
        if C is not None and C <= 0:
            raise ValueError("C must be positive.")
        if gamma is not None and gamma<=0:
            raise ValueError("gamma must be positive.")
        if (kernel is not None) and (kernel not in ["Linear", "Polynomial", "RBF"]):
            raise ValueError("kernel must be a str with values ('Linear', 'Polynomial' or 'RBF')")
        if d is not None and d<2:
            raise ValueError("d must be d>=2.")
        
        
        if margin_type == "Soft" and (C==None):
            raise ValueError("Set C")
        if (kernel == "RBF") and (gamma==None):
            raise ValueError("Set gamma")
        if (kernel == "Polynomial") and (d is None or gamma is None):
            raise ValueError("Set d and gamma")
        
        if margin_type == 'Hard' and kernel != 'Linear':
            raise ValueError("margin_type : 'Hard' only supported with kernel: 'Linear'")
        
        self.margin_type = margin_type
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.d = d
        self.MemoryProblem = MemoryProblem
        
        #Learned after fit
        self.classifiers = {}  # Dictionary to store the SVMs for each pair of classes 
        self.pairs_ = None
        self.number_of_classes = None
        self.class_to_index = None
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.number_of_classes = len(self.classes_)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.pairs_ = list(combinations(self.classes_, 2))
        number_of_binary_svms = len(self.pairs_)
        i=1
        for (class1, class2) in self.pairs_:
            # For one pair
            idx = np.where((y == class1) | (y == class2))
            X_pair = X[idx]
            y_pair = y[idx]
            
            # Train a binary SVM for this pair
            clf = svm(margin_type=self.margin_type, C=self.C, kernel=self.kernel, gamma=self.gamma, d=self.d)
            clf.fit(X_pair, y_pair)
            if self.MemoryProblem:
                model_filename = f"svm_{class1}_vs_{class2}.joblib"
                joblib.dump(clf, model_filename)  # Save to disk
                self.classifiers[(class1, class2)] = model_filename  # Store the filename
            else:
                self.classifiers[(class1, class2)] = clf  # Store the svm object
            print(f"Fitted binary SVM {i}/{number_of_binary_svms}")
            i=i+1
    def predict(self, X_test):
        num_samples = X_test.shape[0]
        votes = np.zeros((num_samples, self.number_of_classes), dtype=int)
        
        if self.MemoryProblem:
            # Process test data in batches
            batch_size = max(1, num_samples // 5)
            j=1
            for i in range(0, num_samples, batch_size):
                end = min(i+batch_size, num_samples)
                batch = X_test[i:end]
                print(f"Making predictions for test batch {j} out of 5")
                j=j+1
                for pair in self.pairs_:
                    # Load the model for this pair
                    clf = joblib.load(self.classifiers[pair])
                    preds = clf.predict(batch)
                    votes[i:end, self.class_to_index[pair[0]]] += (preds == pair[0])
                    votes[i:end, self.class_to_index[pair[1]]] += (preds == pair[1])     
        else:            
            for pair in self.pairs_:
                preds = self.classifiers[pair].predict(X_test) 
                votes[:, self.class_to_index[pair[0]]] += (preds == pair[0])
                votes[:, self.class_to_index[pair[1]]] += (preds == pair[1])
            
        predicted_classes_indexes = np.argmax(votes, axis=1)
        predicted_classes = np.array(self.classes_)[predicted_classes_indexes]
        return np.array(predicted_classes)

    def get_params(self, deep=True):
        # To be able to use the clone() function on the k-fold validation 
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'd': self.d,
            'MemoryProblem': self.MemoryProblem
        }  
    
    def cleanup_binary_svms(self):
        #Deletes all model files created during training when MemoryProblem=True.
        if not self.MemoryProblem:
            print("No files to clean up because MemoryProblem is set to False.")
            return
        
        print("Cleaning up model files...")
        for pair, filename in self.classifiers.items():
            if isinstance(filename, str):  
                try:
                    os.remove(filename)
                    print(f"Deleted file: {filename}")
                except FileNotFoundError:
                    print(f"File not found, skipping: {filename}")
                except Exception as e:
                    print(f"Error deleting file {filename}: {e}")
        self.classifiers = {}  
        print("Cleanup completed.")
            
            
        