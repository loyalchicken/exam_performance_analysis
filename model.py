import numpy as np

def shuffle(train_data, train_labels, random_seed):
    np.random.seed(random_seed)
    permutation = np.random.permutation(train_data.shape[0])
    reconstructed_train_data = np.ndarray(shape=train_data.shape)
    reconstructed_train_labels = np.ndarray(shape=train_labels.shape)
    counter = 0
    test_bool = True
    for i in permutation:
        reconstructed_train_data[counter] = train_data.iloc[i]
        reconstructed_train_labels[counter] = train_labels.iloc[i]
        if np.array_equal(reconstructed_train_data[counter], train_data.iloc[i])==False:
            test_bool=False
        counter+=1    
    return reconstructed_train_data, reconstructed_train_labels

def split(data, train_percent, val_percent):
    n=data.shape[0]
    train=data[:int(n*train_percent)]
    val=data[int(n*train_percent):int(n*(train_percent+val_percent))]
    test=data[int(n*(train_percent+val_percent)):]
    return train, val, test

def construct_polynomial_kernel_matrix(m1, m2T, power):
  K = [[0 for i in range(m2T.shape[1])] for j in range(m1.shape[0])]
  for i in range(m1.shape[0]):
      for j in range(m2T.shape[1]):
          K[i][j] = np.power(np.dot(np.array(m1)[i,:], np.array(m2T)[:,j]) + 1, power)        
  return np.array(K)

def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''        
    num_mixups = 1
    data = np.array([])
    data_labels = np.array([])
    
    for i in range(num_mixups):
        #lam = np.random.beta(alpha, alpha)
        lam = 0.5
        index = np.random.permutation(x.shape[0])
        
        mixed_x = lam * np.array(x) + (1 - lam) * np.array(x)[index,:]
        y_a, y_b = y, y[index]
        mixed_x_labels = lam*y_a + (1 - lam) * y_b
        
        if len(data) == 0:
            data = mixed_x
            
        else:
            data = np.vstack((data, mixed_x))
        data_labels = np.append(data_labels, mixed_x_labels)
        
    return data, data_labels

  
#LINEAR REGRESSION
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge

def regression(classifier, regularizer, kernel_degree, mixup, train, validation, test, train_labels, validation_labels, test_labels):
    X, V, T = train.copy(), validation.copy(), test.copy()
    X_labels, V_labels, T_labels = train_labels.copy(), validation_labels.copy(), test_labels.copy()
    
    ########################## PREPROCESS DATA: CENTER AND ADD BIAS ############################
    
    #square View Per HW
    #if 2 in X.columns:
        #X[2] = X[2]**2
    #if 1 in X.columns:
        #X[1] = X[1]**2

    #standardize
    X = (X-np.mean(X,axis=0)) / np.std(X, axis=0)
    V = (V-np.mean(V,axis=0)) / np.std(V, axis=0) 
    T = (T-np.mean(T,axis=0)) / np.std(T, axis=0)
    
    #normalize 
    #X=X/np.linalg.norm(X, axis=0)
    #V=V/np.linalg.norm(V, axis=0)
    #T=T/np.linalg.norm(T, axis=0)
    
    #mixup
    if mixup == True:
        X, X_labels = mixup_data(X, X_labels, 2)
    
    #add bias
    X = np.hstack((X, np.ones(X.shape[0]).reshape(X.shape[0], 1)))
    V = np.hstack((V, np.ones(V.shape[0]).reshape(V.shape[0], 1)))
    T = np.hstack((T, np.ones(T.shape[0]).reshape(T.shape[0], 1)))

    #standardizing labels
    #X_labels=(X_labels-np.mean(X_labels,axis=0)) 
    #V_labels=(V_labels-np.mean(V_labels,axis=0)) 
    #T_labels=(T_labels-np.mean(T_labels,axis=0)) 

    #preprocess another copy of train 
    train_for_mixup = train.copy()
    train_for_mixup = (train_for_mixup-np.mean(train_for_mixup,axis=0)) / np.std(train_for_mixup, axis=0)
    train_for_mixup = np.hstack((train_for_mixup, np.ones(train_for_mixup.shape[0]).reshape(train_for_mixup.shape[0], 1)))

    ####################### LEARN PREDICTIVE FUNCTION FROM TRAINING DATA #######################
    
    if classifier=='ridge':
        w_s = np.linalg.inv(X.T@X+np.identity(X.shape[1])*regularizer) @ X.T @ X_labels

    elif classifier=='ridge_sklearn':
        clf = linear_model.Ridge(alpha=regularizer, fit_intercept=False, normalize=False)
        clf.fit(X, X_labels)
        w_s = clf.coef_

    elif classifier=='ridge_kernel':
        #build kernel matrix
        K = np.array(construct_polynomial_kernel_matrix(X, X.T, kernel_degree))
    
        #solve (XXt + lambda*I)a = y
        a_s = np.linalg.inv(K + np.identity(K.shape[0])*regularizer) @ X_labels

    elif classifier=='ridge_kernel_sklearn':
        clf = KernelRidge(alpha=regularizer, kernel='polynomial', degree=kernel_degree)
        clf.fit(X, X_labels)
        
    elif classifier=='lasso_sklearn':
        clf = linear_model.Lasso(alpha=regularizer, fit_intercept=False, normalize=False)
        clf.fit(X, X_labels)
        w_s = clf.coef_
    
    elif classifier=='lstsq':
        w_s=np.dot(np.linalg.inv(X.T@X) @ X.T, X_labels) 

    elif classifier=='lstsq_sklearn':
        clf = linear_model.LinearRegression(fit_intercept=False, normalize=False)
        clf.fit(X, X_labels)
        w_s = clf.coef_

    #################### COMPUTE LEARNED MODEL PREDICTIONS ON VAL AND TEST DATA #####################
    
    if classifier == "ridge_kernel":
        #construct kernel matrices for train, val, and test
        K_train = K
        K_val = construct_polynomial_kernel_matrix(V, X.T, kernel_degree)     
        K_test = construct_polynomial_kernel_matrix(T, X.T, kernel_degree)        
        
        validation_predictions = K_val @ a_s
        test_predictions = K_test @ a_s
        train_predictions = K_train @ a_s

    elif classifier == 'ridge_kernel_sklearn':
        validation_predictions=clf.predict(V)
        test_predictions=clf.predict(T)
        train_predictions=clf.predict(train_for_mixup)
    
    else:          
        validation_predictions=np.dot(V, w_s)
        test_predictions=np.dot(T, w_s)
        train_predictions=np.dot(train_for_mixup, w_s)

    ############### COMPUTE ABSOLUTE LOSS OF PREDICTIONS RELATIVE TO GROUND TRUTH  ################
    
    validation_loss=np.average(np.abs(validation_predictions-V_labels))
    test_loss=np.average(np.abs(test_predictions-T_labels))
    train_loss=np.average(np.abs(train_predictions-train_labels.copy()))
    
    return train_loss, validation_loss, test_loss

def hyperparam_search(model, regularization_params, poly_kernel_params, mixups, train_data, val_data, test_data, train_labels, val_labels, test_labels):
    lowest_test_loss = np.inf
    lowest_train_loss = np.inf
    lowest_val_loss = np.inf
    best_params = (None, None, None)
    
    #LEAST SQUARES
    if len(regularization_params) == 0:
        for mixup in mixups:
            train_loss, val_loss, test_loss = regression(model, None, None, mixup, train_data, val_data, test_data, train_labels, val_labels, test_labels)
            return model, train_loss, val_loss, test_loss, (None, None, mixup)

    for mixup in mixups:
        for p in regularization_params:
            
            #RIDGE KERNEL
            if len(poly_kernel_params) > 0:
                for d in poly_kernel_params:
                    train_loss, val_loss, test_loss = regression(model, p, d, mixup, train_data, val_data, test_data, train_labels, val_labels, test_labels)
                    if val_loss < lowest_val_loss:
                        lowest_test_loss = test_loss
                        lowest_train_loss = train_loss
                        lowest_val_loss = val_loss
                        best_params = (np.round(p, 2), d, mixup)
            
            #EVERYTHING ELSE
            else:
                train_loss, val_loss, test_loss = regression(model, p, None, mixup, train_data, val_data, test_data, train_labels, val_labels, test_labels)
                if val_loss < lowest_val_loss:
                    lowest_test_loss = test_loss
                    lowest_train_loss = train_loss
                    lowest_val_loss = val_loss
                    best_params = (np.round(p, 2), None, mixup)
                
    return model, lowest_train_loss, lowest_val_loss, lowest_test_loss, best_params