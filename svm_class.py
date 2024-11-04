def calculate_matrixes_hard_margin(x, y):
    n = x.shape[0]
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

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    n = x.shape[0]
    P = .5 * (P + P.T)  # make sure P is symmetric (solver will not check)
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
    return w, b, S

def calculate_matrixes_soft_margin(x, y, C):
    n = x.shape[0]
    if len(np.unique(y))!=2:
        return
    q_matrix = np.array([-1 for i in range(n)]).reshape(n,1).astype(float)
    G_matrix = np.vstack((np.multiply(np.identity(n), -1), np.identity(n))).astype(float) #(2n,n) matrix, is like having two identity matrixes ,one with -1 and one with 1, with the first on top of the second
    h_matrix = np.array([0 if i<n else C for i in range(2*n)]).astype(float) #(2n, 1) matrix with the value 0 for [0,n] and C for [n,2n]
    y_values = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).reshape((1,n))
    A_matrix = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).astype(float)
    b_matrix = 0.0
    dot_x_matrix = np.dot(x, np.transpose(x))
    dot_y_matrix = np.dot(np.transpose(y_values), y_values)
    p_matrix = np.multiply(dot_x_matrix, dot_y_matrix).astype(float)
    return p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values

def cvxopt_solve_qp_soft_margin(P, q, G, h, A=None, b=None):
    n = x.shape[0]
    P = .5 * (P + P.T)  # make sure P is symmetric
    cvxopt.solvers.options['show_progress'] = False
    args = [cvxopt.matrix(P, (n, n), 'd'), cvxopt.matrix(q, (n,1), 'd')]
    args.extend([cvxopt.matrix(G, (2*n, n), 'd'), cvxopt.matrix(h, (2*n,1), 'd')]) #change the dimensions of G and h matrixes
    if A is not None:
        args.extend([cvxopt.matrix(A, (1, n), 'd'), cvxopt.matrix(b, (1, 1), 'd')])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']: 
        return None
    print("Optimal solution found")
    return np.array(sol['x']).reshape((P.shape[1],))

def calculate_w_b_soft_margin(a,x,y):
    w = np.dot((y*a).T, x)
    S = (a > 1e-5).flatten()
    b = np.mean(y[S] - np.dot(x[S], w.T))
    ek_support_vectors = ((np.dot(x[S], w.T) + b) * y[S] - 1)* -1 #The distance of support vectors from their border. The positive values are in the wrong side of their classification border and the negatives on the right
    return w, b, S, ek_support_vectors



def rbf_kernel(gamma, x_one, x_two):
    r = x_one-x_two
    rbf_calc = np.exp(-gamma*(np.dot(r.T, r)))
    return rbf_calc

def rbf_matrix(gamma, x1, x2):
    final_matrix = []
    try:
        x2.shape[1]  #This is for the case that we want to use the rbf kernel to predict only one instance (on inference!). 
    except IndexError:
        matrix_row = []
        for i in range(x1.shape[0]):
            matrix_row.append(rbf_kernel(gamma, x_one=x1[i], x_two=x2))
        final_matrix.append(matrix_row)
        final_matrix = np.array(final_matrix)
        return final_matrix        
        
    for i in range(x1.shape[0]):  
        matrix_row = []
        for j in range(x2.shape[0]):
            matrix_row.append(rbf_kernel(gamma, x_one=x1[i], x_two=x2[j]))
        final_matrix.append(matrix_row)
    final_matrix = np.array(final_matrix) #Will make a (n, n) matrix where each element uses the rbf kernel
    return final_matrix

def kernelized_dot_product(x1, x2, kernel, d, gamma):
    if str(kernel) == "Linear":
        dot_matrix = np.dot(x1, np.transpose(x2))
        return dot_matrix
    elif str(kernel) == "Polynomial":
        dot_matrix = (np.dot(x1, np.transpose(x2)) + 1)**d #Polynomial kernel of power d
        return dot_matrix
    elif str(kernel) == "RBF":
        dot_matrix = rbf_matrix(gamma=gamma, x1=x1, x2=x2)
        return dot_matrix
    
def calculate_matrixes_soft_margin_kernel(x, y, C, kernel, d, gamma):
    n = x.shape[0]
    if len(np.unique(y))!=2:
        return
    q_matrix = np.array([-1 for i in range(n)]).reshape(n,1).astype(float)
    G_matrix = np.vstack((np.multiply(np.identity(n), -1), np.identity(n))).astype(float)
    h_matrix = np.array([0 if i<n else C for i in range(2*n)]).astype(float)
    y_values = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).reshape((1,n))
    A_matrix = np.array([-1 if value==np.unique(y)[0] else 1 for value in y]).astype(float)
    b_matrix = 0.0
    dot_x_matrix = kernelized_dot_product(x1=x, x2=x, kernel=kernel, d=d, gamma=gamma) #The only matrix that changes.Instead of simple dot product of each point with another, will be the kernel function of each.
    dot_y_matrix = np.dot(np.transpose(y_values), y_values)
    p_matrix = np.multiply(dot_x_matrix, dot_y_matrix).astype(float)
    return p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, y_values

def calculate_b_kernelized(x, a, kernel, d, gamma,y):
    S = (a > 1e-5).flatten()
    x_support_vectors = x[S]
    ajyj = y[S]*a[S]
    kernelized_xixj = kernelized_dot_product(x1=x_support_vectors, x2=x_support_vectors, kernel=kernel, d=d, gamma=gamma)
    ajyjxjxi = kernelized_xixj * ajyj[np.newaxis, :]
    sum_of_all = sum(ajyjxjxi)
    bi=y[S] - sum_of_all
    b = np.mean(bi) #will use the kernelized_dot_product to calculate b with the kernel trick
    return b
    
def make_preds_linear_forclass(x, w, b, neg_class, pos_class): 
    y_preds = [neg_class if value<0 else pos_class for value in np.dot(x, w.T)+b] 
    return y_preds


def make_preds_soft_margin_kernel_forclass(x_all, x_pred, a, kernel, d, gamma, b, y_val, neg_class, pos_class, S):
    predictions = []
    try: #again used only when we want to predict only one instance (in inference)
        x_pred.shape[1]
    except IndexError:
        aiyi = y_val[S]*a[S]
        kernelized_xixtesti = kernelized_dot_product(x1=x_all[S], x2=x_pred, kernel=kernel, d=d, gamma=gamma)
        sum_for_every_sup_vector = np.sum(aiyi*kernelized_xixtesti)
        decision = sum_for_every_sup_vector+b
        if decision<0:
            prediction = neg_class
        elif decision>=0:
            prediction = pos_class
        return prediction
        
    for i in range(x_pred.shape[0]):  #predictions for a list of test instances, again using the kernel trick.Note that we dont compute w!
        aiyi = y_val[S]*a[S]
        kernelized_xixtesti = kernelized_dot_product(x1=x_all[S], x2=x_pred[i], kernel=kernel, d=d, gamma=gamma)
        sum_for_every_sup_vector = np.sum(aiyi*kernelized_xixtesti)
        decision = sum_for_every_sup_vector+b
        if decision<0:
            prediction = neg_class
        elif decision>=0:
            prediction = pos_class
        predictions.append(prediction)
    return predictions


class svm:
    def __init__(self, margin_type="Soft", C=None, kernel=None, gamma=None, d=None):
        if margin_type not in ["Hard", "Soft"]:
            raise ValueError("margin_type must be a str with values ('Hard' or 'Soft')")
        if C is not None and C <= 0:
            raise ValueError("C must be positive.")
        if gamma is not None and gamma<=0:
            raise ValueError("gamma must be positive.")
        if (kernel is not None) and (str(kernel) not in ["Linear", "Polynomial", "RBF"]):
            raise ValueError("kernel must be a str with values ('Linear', 'Polynomial' or 'RBF')")
        if d is not None and d<2:
            raise ValueError("d must be d>=2.")
        
        

        if str(margin_type) == 'Hard' and str(kernel) != 'Linear':
            raise ValueError("margin_type : 'Hard' only supported with kernel: 'Linear'")
        
        
        

        self.margin_type = str(margin_type)
        self.C = C
        self.kernel = str(kernel)
        self.gamma = gamma
        self.d = d

        #Atributes learned after fit
        self.n_ = None
        self.x_all_ = None
        self.y_values_ = None
        self.support_vectors_ = None
        self.alphas_matrix_ = None
        self.w_ = None
        self.b_ = None
        self.S_ = None
        self.ek_support_vectors_ = None
        self.neg_class_ = None
        self.pos_class_ = None
        
        

    def fit(self, x, y):
        self.n_ = x.shape[0]
        self.x_all_ = x
        self.neg_class_ = np.unique(y)[0] 
        self.pos_class_ = np.unique(y)[1]
        if self.margin_type == 'Hard' and self.kernel == 'Linear':
             p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, self.y_values_ = calculate_matrixes_hard_margin(x=x, y=y)
             self.alphas_matrix_ = cvxopt_solve_qp(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, A=A_matrix, b=b_matrix)
             self.w_, self.b_, self.S_ = calculate_w_b(a=self.alphas_matrix_,x=x,y=self.y_values_.reshape(self.n_))
        elif (self.margin_type == 'Soft') and (self.kernel == 'Linear'):
            p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, self.y_values_ = calculate_matrixes_soft_margin(x=x, y=y, C=self.C)
            self.alphas_matrix_ = cvxopt_solve_qp_soft_margin(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, A=A_matrix, b=b_matrix)
            self.w_, self.b_, self.S_, self.ek_support_vectors_ = calculate_w_b_soft_margin(a=self.alphas_matrix_,x=x,y=self.y_values_.reshape(self.n_))
        elif (self.margin_type == 'Soft') and (self.kernel in ['RBF', 'Polynomial']):
            p_matrix, q_matrix, G_matrix, h_matrix, A_matrix, b_matrix, self.y_values_ = calculate_matrixes_soft_margin_kernel(x=x, y=y, C=self.C, kernel=self.kernel, d=self.d, gamma=self.gamma)
            self.alphas_matrix_ = cvxopt_solve_qp_soft_margin(P=p_matrix, q=q_matrix, G=G_matrix, h=h_matrix, A=A_matrix, b=b_matrix)
            self.b_ = calculate_b_kernelized(x=x, a=self.alphas_matrix_, kernel=self.kernel, d=self.d, gamma=self.gamma, y=self.y_values_.reshape(self.n_))
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
            y_preds = make_preds_soft_margin_kernel_forclass(x_all=self.x_all_, x_pred=x_test, a=self.alphas_matrix_, kernel=self.kernel, d=self.d, gamma=self.gamma, b=self.b_, y_val=self.y_values_, neg_class=self.neg_class_, pos_class=self.pos_class_, S=self.S_)
            return y_preds
        else:
            raise ValueError("Invalid configuration for margin type or kernel.")


    
