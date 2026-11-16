## IMPORTS ######################################################################
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from CNN import *
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
    

## PCA ##########################################################################
#################################################################################
def fitPCA(images, numPCs):
    """
    Finds PCs over all the images
    Inputs:  images..........(np.ndarray) all images
             num...........(int) max number of principal components to return
    Outputs: proj_data.....(np.ndarray) 
             proj_data1....(np.ndarray)
             pcs...........(np.array) (200, num) principal components
             var...........(list) variance for each component
             var_ratios....(list) percentage variance explained for each component
    """
    num, height, width, channels = images.shape
    X = images.reshape(-1, channels)
    mean = X.mean(axis=0)
    X = X-mean
    C = np.cov(X, rowvar = False)
    var, pcs = scipy.linalg.eigh(C)
    
    # sort eigenvalues
    idx = np.argsort(var)[::-1]
    var = var[idx]
    pcs = pcs[:, idx]
    
    # variance explained
    var_ratios = var/np.sum(var)
    var = var[:numPCs]
    pcs = pcs[:, :numPCs]
    
    return mean, pcs, var, var_ratios, np.sum(var_ratios[:numPCs])

def transPCA(image, mean, pcs):
    """
    Projects one image onto PC basis
    Inputs:  image.....(np.ndarray) image to transform
             mean......(int) mean of all images
             pcs.......(np.ndarray) principal components
    Outputs: proj_X....(np.ndarray) projected X
    """
    height, width, channels = image.shape
    X = image.reshape(-1, channels)
    X = X-mean
    proj_X = X @ pcs
    proj_X = proj_X.reshape(height, width, pcs.shape[1])
    return proj_X

def preprocessPCA(train_imgs, val_imgs, test_imgs, numPCs):
    mean, pcs, var, var_ratios, tot_explained = fitPCA(train_imgs, numPCs)
    Xtrain_PCA = []
    Xval_PCA = []
    Xtest_PCA = []
    
    for img in train_imgs:
        proj = transPCA(img, mean, pcs)
        proj = np.moveaxis(proj, -1, 0)
        Xtrain_PCA.append(proj)
        
    Xtrain_PCA = np.stack(Xtrain_PCA)
    
    for img in val_imgs:
            proj = transPCA(img, mean, pcs)
            proj = np.moveaxis(proj, -1, 0)
            Xval_PCA.append(proj)
    
    Xval_PCA = np.stack(Xval_PCA)
    
    for img in test_imgs:
        proj = transPCA(img, mean, pcs)
        proj = np.moveaxis(proj, -1, 0)
        Xtest_PCA.append(proj)

    Xtest_PCA = np.stack(Xtest_PCA)
    return Xtrain_PCA, Xval_PCA, Xtest_PCA, tot_explained

        
## TESTER #######################################################################
#################################################################################
def run_tests(Xtrain, ytrain, Xval, yval, Xtest, ytest, classes):
    """
    Tests Ks for PCA
    Inputs:  train/test/val data
             classes........(np.ndarray) class labels
    """
    Ks = np.arange(2, 10, 1)
    nclasses = len(classes)
    varExplained = []
    f1s = []
    explains99 = 0
    amount = 0
    
    # How Many PCs??
    for K in Ks:
        print(f"Testing K = {K}")
        train, val, test, tot_explained = preprocessPCA(Xtrain, Xval, Xtest, K)
        model, hist = train_CNN(train, ytrain, val, yval, nclasses)
        trainPlot(hist, f"PCA_{K}")
        fw, cm = test_CNN(model, test, ytest, classes, f"PCA_{K}")
        f1s.append(fw)
        varExplained.append(tot_explained)
     
        if tot_explained >= 0.99 and amount == 0:
            explains99 = K
            amount = tot_explained
       
    plt.plot(Ks, varExplained, marker = "o", color = "royalblue")
    plt.axhline(y = 0.95, linestyle = '--', color = "violet", label = "95 %")
    plt.axhline(y = 0.99, linestyle = '--', color = "violet", label = "99 %")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Total Variance Explained")
    plt.legend()
    plt.title(f"{explains99} Components explain {amount:.5f} Variance.")
    plt.savefig("testNumPCs.png")
    plt.close()
    
    plt.plot(Ks, f1s, marker = 'o', color = "royalblue")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("F1 Score")
    plt.title("CNN Performance")
    plt.savefig("CNN_numpcs.png")
    plt.close()

def get_data():
    """
    creates train test split of 70/15/15
    """
    print("Getting Data and Downsampling")
    train = np.load("train.npz")
    Xtrain = train["X"]
    ytrain = train["y"]
    classes = train["classes"]
    
    Xtrain, Xtemp, ytrain, ytemp = train_test_split(Xtrain, ytrain, test_size = 0.30, stratify = ytrain, random_state = 42)
    Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, test_size = 0.50, stratify = ytemp, random_state = 42)
    
    run_tests(Xtrain, ytrain, Xval, yval, Xtest, ytest, classes)
    
    
## MAIN #########################################################################
#################################################################################
if __name__ == "__main__":
    get_data()