## IMPORTS ######################################################################
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from CNN import *
from prettytable import PrettyTable
from scipy.spatial.distance import pdist
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
    
    # rescaling
    mu = Xtrain_PCA.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtrain_PCA.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    Xtrain_PCA = (Xtrain_PCA - mu) / sd
    Xval_PCA = (Xval_PCA - mu) / sd
    Xtest_PCA = (Xtest_PCA - mu) / sd
    
    return Xtrain_PCA, Xval_PCA, Xtest_PCA, tot_explained


        
## KPCA #########################################################################
#################################################################################
def makeKernel(X, Y=None, type = "gauss", gamma=1e-3, deg=3, coef0=1):
    """
    Computes the Kernel matrix
    Inputs:  X.........(np.ndarray) selected pixels
             Y.........(np.ndarray) no idea
             type......(string) type of kernel
             gamma.....(float) no idea
             deg.......(int) max degree of polynomial
             coeff0....(int) no idea
    Outputs: K.........(np.ndarray) Kernel matrix
    """
    if Y is None: Y=X
    
    if type == "gauss":
        Xnorm = np.sum(X**2, axis = 1)[:, None]
        Ynorm = np.sum(Y**2, axis = 1)[None, :]
        dist = Xnorm + Ynorm - 2*(X @ Y.T)
        K = np.exp(-gamma*dist)
        
    elif type == "poly":
        K = (gamma*(X @ Y.T) + coef0)**deg
            
    elif type == "sigmoid":
        K = np.tanh(gamma*(X@Y.T) + coef0)
    
    return K
    
    
def fitKPCA(images, numPCs, kernel="gauss", numSamples=2000, gamma=1e-3, degree=3, coef0=1):
    """
    Performs PC transform with specified Kernel
    Inputs:  images........(np.ndarray) images 
             num...........(int) max number of principal components to return
             Kernel........(np.ndarray) kernel matrix
             params........(*) parameters for different kernels
    Outputs: Xref, alphas, vals, ker, gamma, degree, coef0, kmeanrow, kmeanall
    """ 
    rng = np.random.default_rng(42)
    num, height, width, channels = images.shape
    
    allpixels = images.reshape(-1, channels)
    idx = rng.choice(len(allpixels), size = min(numSamples, len(allpixels)), replace = False)
    somepixels = allpixels[idx]
    
    if kernel == "gauss":
        d = pdist(somepixels, metric = "sqeuclidean")
        gamma = 1/np.median(d)
        print(f"Gauss Gamma: {gamma}")
    else:
        gamma = 1.0/13
    
    K = makeKernel(somepixels, type = kernel, gamma = gamma, deg = degree, coef0 = coef0)
    N = K.shape[0]
    
    # WTF IS THIS STUFF DOING??
    oneN = np.ones ((N, N))/N
    Kc = (K - oneN@K - K@oneN + oneN@K@oneN)
    
    # eigenstuff sorting
    print("Eigen Time")
    numeigs = min(100, N-1)
    vals, vecs = scipy.sparse.linalg.eigsh(Kc, k = numeigs, which = "LM")
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # begone numerical error
    mask = vals > 1e-8
    vals = vals[mask]
    vecs = vecs[:, mask]
    
    var_ratio = vals/np.trace(Kc)
    tot_explained = np.sum(var_ratio[:numPCs])
    
    # normieees
    vals = vals[:numPCs]
    vecs = vecs[:, :numPCs]
    alphas = vecs/np.sqrt(vals)
    
    return somepixels, alphas, degree, kernel, gamma, coef0, np.mean(K, axis=0), np.mean(K), tot_explained
        
    
# def transKPCA(image, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall):
#     """
#     Projects one image according to KPCA
    
#     Inputs: image and buncha kpca parameters
#     Outputs: projected image
#     """
#     height, width, channels = image.shape
#     X = image.reshape(-1, channels)    
    
#     ker = makeKernel(X, Y= pixels, type = kernel, gamma = gamma, deg =degree, coef0=coef0)
#     row_mean = np.mean(ker, axis = 1, keepdims = True)
#     ker_centered = ker - kermeanrow[None, :] - row_mean + kermeanall
    
#     proj = ker_centered @ alphas
#     proj = proj.reshape(height, width, alphas.shape[1])
#     return proj

def transKPCA_gpu(imgs, pixels, alphas, kernel, gamma, degree, coef0,
                  kermeanrow, kermeanall, device, batch=64):
    P = torch.tensor(pixels, dtype=torch.float32, device=device)
    A = torch.tensor(alphas, dtype=torch.float32, device=device)
    km = torch.tensor(kermeanrow, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.tensor(imgs[i:i+batch], dtype=torch.float32, device=device)
        n, h, w, c = x.shape
        x = x.reshape(-1, c)
        if kernel == "gauss":
            ker = torch.exp(-gamma * torch.cdist(x, P) ** 2)
        elif kernel == "poly":
            ker = (gamma * (x @ P.T) + coef0) ** degree
        else:
            ker = torch.tanh(gamma * (x @ P.T) + coef0)
        ker = ker - km[None, :] - ker.mean(dim=1, keepdim=True) + kermeanall
        proj = (ker @ A).reshape(n, h, w, -1).permute(0, 3, 1, 2)
        out.append(proj.cpu().numpy())
    return np.concatenate(out)


def preprocessKPCA(trainimgs, valimgs, testimgs, numPCs, kernel):
    """
    Runs KPCA capped at numPCs components.
    Inputs:  images and params for KPCA
    Outputs: projected images
    """
    pixels, alphas, degree, kernel, gamma, coef0, kermeanrow, kermeanall, tot_explained = fitKPCA(trainimgs, numPCs, kernel)
    
    Xtrain_KPCA = []
    Xval_KPCA = []
    Xtest_KPCA = []
    
    print("Projecting all the data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtrain_KPCA = transKPCA_gpu(trainimgs, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall, device)
    Xval_KPCA = transKPCA_gpu(valimgs, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall, device)
    Xtest_KPCA = transKPCA_gpu(testimgs, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall, device)
    # for img in trainimgs:
    #     proj = transKPCA(img, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall)
    #     Xtrain_KPCA.append(np.moveaxis(proj, -1, 0))
        
    # for img in valimgs:
    #     proj = transKPCA(img, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall)
    #     Xval_KPCA.append(np.moveaxis(proj, -1, 0))
        
    # for img in testimgs:
    #         proj = transKPCA(img, pixels, alphas, kernel, gamma, degree, coef0, kermeanrow, kermeanall)
    #         Xtest_KPCA.append(np.moveaxis(proj, -1, 0))
        
    # Xtrain_KPCA = np.stack(Xtrain_KPCA)
    # Xval_KPCA = np.stack(Xval_KPCA)
    # Xtest_KPCA = np.stack(Xtest_KPCA)
    
    # rescaling
    mu = Xtrain_KPCA.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtrain_KPCA.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    Xtrain_KPCA = (Xtrain_KPCA - mu) / sd
    Xval_KPCA = (Xval_KPCA - mu) / sd
    Xtest_KPCA = (Xtest_KPCA - mu) / sd
    
    return Xtrain_KPCA, Xval_KPCA, Xtest_KPCA, tot_explained
        
        
        
## TESTER #######################################################################
#################################################################################
def run_tests(Xtrain, ytrain, Xval, yval, Xtest, ytest, classes):
    """
    Runs all the different preprocessors, plugs em into model
    Inputs:  train/test/val data
             classes........(np.ndarray) class labels
    """
    Kmax = 5
    # Let's standardize the data 
    Xtrain = Xtrain.astype(np.float64)
    Xval = Xval.astype(np.float64)
    Xtest = Xtest.astype(np.float64)
    mean = Xtrain.reshape(-1, Xtrain.shape[-1]).mean(axis=0)
    std = Xtrain.reshape(-1, Xtrain.shape[-1]).std(axis=0)
    
    Xtrain = (Xtrain-mean)/std
    Xval = (Xval-mean)/std
    Xtest = (Xtest-mean)/std
    
    # PCA the data
    print("Running PCA")
    trainPCA, valPCA, testPCA, varPCA = preprocessPCA(Xtrain, Xval, Xtest, Kmax)
    
    # KPCA the data: polynomial
    print("Running Polynomial KPCA")
    trainPKPCA, valPKPCA, testPKPCA, varPKPCA = preprocessKPCA(Xtrain, Xval, Xtest, Kmax, "poly")
    
    # KPCA the data: gaussian
    print("Running Gaussian KPCA")
    trainGKPCA, valGKPCA, testGKPCA, varGKPCA = preprocessKPCA(Xtrain, Xval, Xtest, Kmax, "gauss")

    # KPCA the data: sigmoid
    print("Running Sigmoid KPCA")
    trainSKPCA, valSKPCA, testSKPCA, varSKPCA = preprocessKPCA(Xtrain, Xval, Xtest, Kmax, "sigmoid")
    
    # TRAINING MODEL #################################################################################
    nclasses = len(classes)
    
    print("Training PCA___________________________________________")
    model1, hist1 = train_CNN(trainPCA, ytrain, valPCA, yval, nclasses)
    trainPlot(hist1, "PCA")
    fw1, cm1 = test_CNN(model1, testPCA, ytest, classes, "PCA")
    print(fw1)
    print("_______________________________________________________\n\n")
    
    
    print("Training PKPCA_________________________________________")
    model2, hist2 = train_CNN(trainPKPCA, ytrain, valPKPCA, yval, nclasses)
    trainPlot(hist2, "PKPCA")
    fw2, cm2 = test_CNN(model2, testPKPCA, ytest, classes, "PKPCA")
    print("_______________________________________________________\n\n")

    print("Training GKPCA_________________________________________")
    model3, hist3 = train_CNN(trainGKPCA, ytrain, valGKPCA, yval, nclasses)
    trainPlot(hist3, "GKPCA")
    fw3, cm3 = test_CNN(model3, testGKPCA, ytest, classes, "GKPCA")
    print("_______________________________________________________\n\n")

    print("Training SKPCA_________________________________________")
    model4, hist4 = train_CNN(trainSKPCA, ytrain, valSKPCA, yval, nclasses)
    trainPlot(hist4, "SKPCA")
    fw4, cm4 = test_CNN(model4, testSKPCA, ytest, classes, "SKPCA")
    print("_______________________________________________________\n\n")

    table1 = PrettyTable()
    table1.field_names = ("Model", "Variance Captured", "Test F1")
    table1.add_row(["PCA", f"{varPCA:.5f}", f"{fw1:.5f}"])
    table1.add_row(["PKPCA", f"{varPKPCA:.5f}", f"{fw2:.5f}"])
    table1.add_row(["GKPCA", f"{varGKPCA:.5f}", f"{fw3:.5f}"])
    table1.add_row(["SKPCA", f"{varSKPCA:.5f}", f"{fw4:.5f}"])
    print(table1)
    
def get_data():
    """
    Gets data, downsamples train data for PCA's
    Lol jk creates train test split of 70/15/15
    """
    print("Getting Data and Downsampling")
    train = np.load("train.npz")
    Xtrain = train["X"]
    ytrain = train["y"]
    classes = train["classes"]
    
    # rng = np.random.default_rng(seed = 57)
    # n = len(Xtrain) 
    # idxs = rng.choice(n, size = n//2, replace = False)
    # Xtrain_down = Xtrain[idxs]
    # ytrain_down = ytrain[idxs]
    
    Xtrain, Xtemp, ytrain, ytemp = train_test_split(Xtrain, ytrain, test_size = 0.30, stratify = ytrain, random_state = 42)
    Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, test_size = 0.50, stratify = ytemp, random_state = 42)
    
    run_tests(Xtrain, ytrain, Xval, yval, Xtest, ytest, classes)
    
    
## MAIN #########################################################################
#################################################################################
if __name__ == "__main__":
    get_data()