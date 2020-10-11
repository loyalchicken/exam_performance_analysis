import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def PCA(training_std):
  
    ###################### EXTRACTING PRINCIPAL COMPONENTS #######################
    
    #Method 1: Use SVD
    u, s, vt = np.linalg.svd(training_std, full_matrices=False) 
    u.shape, s.shape, vt.shape #columns of v are right singular vectors of X (also the eigenvectors of XtX=COV)

    #Method 2: Maximize Rayleigh's quotient / find direction that maximizes variance of projected data
    #X=UDVt, XtX=VD^2Vt -- V are eigenvectors of XtX, sigma^2 are the eigenvalues of XtX
    cov=np.dot(np.transpose(training_std), training_std)
    evalues, evectors = np.linalg.eigh(cov)  
    evalues=np.flip(evalues) 
    evectors=np.flip(evectors, axis=1)

    ################################### PLOTS #####################################
    
    #1. PROJECTION ONTO THE FIRST 2 PRINCIPAL COMPONENTS
    projections=np.dot(training_std, evectors)
    p1=sns.scatterplot(x=projections[:, 0], y=projections[:, 1])
    p1.set_title("Training points projected onto the first two principal components", 
                 fontsize=16)
    p1.set_xlabel("PC1",fontsize=14)
    p1.set_ylabel("PC2", fontsize=14)
    plt.show()

    #2. SCREE PLOT
    p1=sns.lineplot(np.arange(len(s))+1, np.round(s**2 / sum(s** 2), 3)) #only one principal component needed
    p1.set_title("Scree Plot: Magnitude of the squared singular values aka the variance of the columns of X captured by its corresponding principal component ", 
                 fontsize=16)
    p1.set_ylabel("Percentage of Variance Captured",fontsize=14)
    p1.set_xlabel("Principal Component", fontsize=14)
    plt.show()
    
    
    #3. PROJECTION ONTO THE FIRST PRINCIPAL COMPONENT
    projections_1_pc=np.dot(training_std, evectors[:,0].reshape((6,1)))
    p2=sns.distplot(projections_1_pc, bins=100, kde=False)
    p2.set_title("Histogram of the 1D coordinates of the training data projected on the first principal direction", 
                 fontsize=16)
    p2.set_xlabel("Coordinate Value",fontsize=14)
    p2.set_ylabel("Number", fontsize=14)
    plt.show()
    
    #4. PRINT SQUARED SINGULAR VALUES
    print("variances of features explained by singular values")
    print(np.round(s**2 / sum(s** 2), 2))

    print("cumulative variances of features explained by singular values")
    print(np.round(np.cumsum(s**2) / sum(s** 2), 2))
    
    print("sum of the squared singular values of X (normalized) == sum of the variances of the features!")
    print(sum(s**2)/training_std.shape[0], sum(np.var(training_std, axis=0)))