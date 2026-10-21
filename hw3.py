## IMPORTS #####################################################################
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

## PLOTTING ####################################################################
################################################################################
def scatterFisher(data, labels):
    """ 
    Inputs:  data......(np array) train data projected to fisher space
             labels....(np array) corresponding labels
    Outputs: scatterplot of data with first two fisher components
    """
    colors = ["limegreen", "deeppink", "blue"]
    labelnames = ["Forest", "AnnualCrop", "Residential"]
    
    for i, label in enumerate(np.unique(labels)):
        just_this_label = labels == label
        plt.scatter(data[just_this_label, 0], data[just_this_label, 1], s = 0.8, label = f"{labelnames[i]}", color = colors[i])
        
    plt.xlabel("Fisher Direction 1")
    plt.ylabel("Fisher Direction 2")
    plt.title("Projected Training Data")
    plt.legend(title = "Class")
    plt.tight_layout()
    plt.savefig(f"Scatter_Fisher.png")
    plt.close()
    
def confusion(ytest, ypred):
    """ 
    Inputs:  ytest..........(np array) fisher LDA predicted labels
             ypred.........(np array) ground truth labels
    Outputs: confusion matrix plot
    """
    classnames = ["Forest", "AnnualCrop", "Residential"]
    confused = confusion_matrix(ytest, ypred, labels = [0,1,2])
    display = ConfusionMatrixDisplay(confusion_matrix=confused, display_labels=classnames)
    display.plot(cmap = "Greens", values_format = "d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("Confusion_Matrix.png")
    plt.close()

## ALGOS #######################################################################
################################################################################
def fisherLDA(data, labels):
    """
    Does fisher LDA
    Inputs:  Data..................(np array) hyperspectral image data
             labels................(np array) corresponding class labels
    Outputs: discrim_directions....(np array) first three Fisher discriminant directions
    """
    # Loading data in 
    data = np.asarray(data, dtype = float)
    labels = np.asarray(labels)
    nsamples, nfeatures = data.shape # should always be 13 features 
    classes = np.unique(labels)
    
    mu = np.mean(data, axis = 0)
        
    # Within-class and between-class scatter matrices
    Sw = np.zeros((nfeatures, nfeatures))
    Sb = np.zeros((nfeatures, nfeatures))
    
    # Forming Sw, Sb for each class
    for c in classes:
        Xc = data[labels ==c]
        muc = np.mean(Xc, axis = 0)
        nc = Xc.shape[0]
        
        Sw += (Xc - muc).T@(Xc - muc)
        
        mudiff_colvec = (muc-mu).reshape(-1,1)
        Sb += nc*(mudiff_colvec@mudiff_colvec.T)
     
    print("Hrmmmmmmm")
    print("Rank of Sw:", np.linalg.matrix_rank(Sw))
    print("Condition number:", np.linalg.cond(Sw))
        
    # Gen. Eigenvalue problem, thank you scipy
    evals, evecs = eigh(Sb, Sw)
    
    # Sort by largest eigenvalue
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    
    # Pull out top three
    discrim_directions = evecs[:, :3]
    
    return discrim_directions   
    
def load_data(trainfile):
    """
    Loads training, testing data
    Selects 500 images from each class and uses a 70/30 split
    """
    train = np.load(trainfile)
    
    X = train["X"]
    y = train["y"]
    classes = train["classes"]
    needed_classes = ["Forest", "AnnualCrop", "Residential"]
    
    seed = 57
    rng = np.random.default_rng(seed)
    selected = []
    selected_labels = []
    
    for newlabel, c in enumerate(needed_classes):
        label = np.where(classes == c)[0][0]
        idxs = np.flatnonzero(y == label)
        
        select = rng.choice(idxs, size = 500, replace = False)
        selected.extend(select)
        selected_labels.extend([newlabel]*500) # assigns label to the 500 selected
    
    newX = X[selected]
    newy = np.array(selected_labels)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(newX, newy, test_size = 0.3, random_state = seed, stratify = newy)
    print(f"Number of Training Images: {Xtrain.shape}")  
    print(f"Number of Testing Images: {Xtest.shape}")  
    
    return Xtrain, Xtest, ytrain, ytest

    
## MAIN ########################################################################
################################################################################
if __name__ == "__main__":
    trainfile = ".//HW3//train.npz"
    train = np.load(trainfile)
    
    Xtrain, Xtest, ytrain, ytest = load_data(trainfile)
    
    # Average each spectral band across image = 13 features per image
    Xtrain_features = Xtrain.mean(axis = (1,2))
    Xtest_features = Xtest.mean(axis = (1,2))

    W = fisherLDA(Xtrain_features, ytrain)
        
    # Projecting into fisher subspace
    proj_train = Xtrain_features@W
    proj_test = Xtest_features@W
    
    # Predicted labels
    classes = np.unique(ytrain)
    centroids = np.array([proj_train[ytrain == c].mean(axis=0) for c in classes])
    dists = np.sum((proj_test[:, None, :] - centroids[None, :, :])**2, axis = 2)
    ypred = classes[np.argmin(dists, axis=1)]

    # Plotting
    scatterFisher(proj_train, ytrain)
    confusion(ytest, ypred)
    