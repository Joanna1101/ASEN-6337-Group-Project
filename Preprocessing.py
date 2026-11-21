## IMPORTS ######################################################################
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy 

## PLOTTING? #####################################################################
#################################################################################


## PCA ##########################################################################
#################################################################################
def PCA(image, numPCs):
    """
    Performs PC transform
    Inputs:  image..........(np.ndarray) one img
             num...........(int) max number of principal components to return
    Outputs: proj_data.....(np.ndarray) 
             proj_data1....(np.ndarray)
             pcs...........(np.array) (200, num) principal components
             var...........(list) variance for each component
             var_ratios....(list) percentage variance explained for each component
    """
    H, W, B = image.shape
    X = image.reshape(-1, B)                 
    X = X - X.mean(axis = 0)                  # average each band
    C = np.cov(X, rowvar = False)             # cols are variables
    var, pcs = scipy.linalg.eigh(C)           # C is Hermitian! Yay!
    
    # sort and take largest
    idx = np.argsort(var)[::-1]               # largest eigenvalues first (argsort puts smallest first)
    var_ratios = var/var.sum()
    var_ratios = var_ratios[idx][:numPCs]
    var = var[idx][:numPCs]
    pcs = pcs[:, idx][:, :numPCs]
    
    proj_data = X@pcs                         # (145*145, num) row: one pixel, num PC coefficients
    
    # Reshape to image format
    proj_data = proj_data.reshape(H, W, numPCs)
    proj_data1 = proj_data[:, :, 0]
    
    return proj_data, proj_data1, pcs, var, var_ratios
    

## DRIVER #######################################################################
#################################################################################
def testKs(images):
    """
    Performs dimensionality reduction via PCA
    Inputs:  image.....(np.array) just one for now
    Outputs: PCs
    """
    Ks = np.arange(3, 14, 1)
    varExplained = []
    
    # for each K taking average across all imgs of total variance explained
    for K in Ks:
        varExplainedTotals = []
        for img in images:
            proj_data, proj_data1, pcs, var, var_ratios = PCA(img, K)
            varTot = np.cumsum(var_ratios)[-1]
            varExplainedTotals.append(varTot)
        avgtot = np.mean(varExplainedTotals)
        varExplained.append(avgtot)
    
    plt.plot(Ks, varExplained, marker = "o", color = "mediumvioletred")
    plt.axhline(y = 0.95, linestyle = '--', color = "royalblue", label = "95 %")
    plt.axhline(y = 0.99, linestyle = '--', color = "royalblue", label = "99 %")
    # plt.plot(25, varExplained[14], marker = "o", color = "darkblue", label = "PC 23")    
    # plt.plot(5, varExplained[3], marker = "o", color = "royalblue", label = "PC 5") 
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Total Variance Explained")
    plt.legend()
    plt.title(f"Highest: {varExplained[-1]:.5f}, {Ks[-1]} PCs.  Lowest: {varExplained[0]:.5f}, {Ks[0]} PCs.")
    plt.savefig("testNumPCs.png")
    
def preprocess(images, n, K):
    """
    Runs PCA capped at K components on n randomly selected images from images.
    """
    rng = np.random.default_rng(seed = 57)
    tot_len = len(images)
    idxs = rng.choice(tot_len, size = n, replace = False)
    downsampled = images[idxs]
    
    PC1s = []
    
    for img in images:
        proj_data, proj_data1, pcs, var, var_ratios = PCA(img, K)
        varTot = np.cumsum(var_ratios)[-1]
        PC1s.append(proj_data1)

    # plot some of the pc1s        

## MAIN #########################################################################
#################################################################################
if __name__ == "__main__":
    train = np.load("train.npz")
    test = np.load("test.npz")
    Xtrain = train["X"]
    Xtest = test["X"]
    # ytrain = train["y"]
    # ytest = test["y"]
    classes = train["classes"]
    
    rng = np.random.default_rng(seed = 57)
    n = len(Xtrain)
    idxs = rng.choice(n, size = n//2, replace = False)
    Xtrain_downsampled = Xtrain[idxs]
    # ytrain_downsampled = ytrain[idxs]
    
    testKs(Xtrain_downsampled)
    
    K = 7
    n = n//2
    preprocess(Xtrain, n, K)
