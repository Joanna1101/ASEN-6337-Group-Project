## IMPORTS ######################################################################
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from CNN import *
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist


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
        train, val, test, tot_explained = preprocessKPCA(Xtrain, Xval, Xtest, K, "gauss")
        model, hist = train_CNN(train, ytrain, val, yval, nclasses)
        trainPlot(hist, f"GPCA_{K}")
        fw, cm = test_CNN(model, test, ytest, classes, f"GPCA_{K}")
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
    plt.savefig("G_testNumPCs.png")
    plt.close()
    
    plt.plot(Ks, f1s, marker = 'o', color = "royalblue")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("F1 Score")
    plt.title("CNN Performance")
    plt.savefig("G_CNN_numpcs.png")
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