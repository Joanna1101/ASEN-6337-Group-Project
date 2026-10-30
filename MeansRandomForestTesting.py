# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import pathlib as Path
import sklearn.ensemble as skl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

DATA_DIR = Path.Path("GroupProjectData")

# %%
with np.load(DATA_DIR / "train.npz") as data:
    print("Datasets:", data.files)
    X = data["X"]
    Y = data["y"]

    classNames = data["classes"]
    print("Class Names:", classNames)
    classes = np.unique(Y)
    print("Classes:", classes)
    print("Number of classes:", len(classes))

    bands = data["bands"]
    print("Number of bands:", len(bands))
    print("Bands:", bands)

    wavelength_nm = data["wavelength_nm"]
    print("Wavelength (nm):", wavelength_nm)
    print("Number of wavelengths:", len(wavelength_nm))
    print("Wavelengths (nm):", wavelength_nm)

    resolution_m = data["resolution_m"]
    print("Resolution (m):", resolution_m)
    print("Number of resolutions:", len(resolution_m))
    print("Resolutions (m):", resolution_m)


    

# %%
for i, cls in enumerate(classes):
    print(f"Class {i}: {classNames[cls]}")


# %%
# Spatial Averaging Images to 5x13D Feature Vectors
edgeSize = 4

# <chatGPT>
def compress(data, edge_size):
    edge_size = min(edge_size, data.shape[0], data.shape[1])
    return np.array([
        np.mean(data, axis=(0, 1)),
        np.mean(data[0:edge_size, :, :], axis=(0, 1)),   # Top
        np.mean(data[-edge_size:, :, :], axis=(0, 1)),  # Bottom
        np.mean(data[:, 0:edge_size, :], axis=(0, 1)),   # Left
        np.mean(data[:, -edge_size:, :], axis=(0, 1))   # Right
    ])

meansX = np.array([np.ndarray.flatten(compress(entry,edgeSize)) for entry in X])
# </chatGPT>

print("Shape of Means Vector:")
print(meansX.shape)

# %%
# Split into training and test data
trainPercent = 0.80
randomSampleIndices = np.random.choice(len(X), size=math.floor(len(X)*trainPercent), replace=False)
trainX = meansX[randomSampleIndices]
trainY = Y[randomSampleIndices]

testX = np.delete(meansX, randomSampleIndices, 0)
testY = np.delete(Y, randomSampleIndices, 0)

print("Shape of Train/Test Datasets:")
print("trainX", trainX.shape)
print("trainY", trainY.shape)
print("testX", testX.shape)
print("testY", testY.shape)


# %%
trainSeed = 74967
clf = skl.RandomForestClassifier(random_state=trainSeed)
clf.fit(trainX, trainY)

# %%
# How did it go?
print("Total Test Accuracy of Edge Means", clf.score(testX, testY))

# Make confusion matrix
predictY = clf.predict(testX)
cm = confusion_matrix(predictY, testY)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot()
plt.xticks(rotation=90)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix for Edge Means of Size " + str(edgeSize))
plt.savefig("Confusion_Matrix_Test.png")
plt.show()


totalAccuracy = []
# %%
# iterate over many border sizes
for k in range(1, 64):
    meansX = np.array([np.ndarray.flatten(compress(entry, k)) for entry in X])
    trainX = meansX[randomSampleIndices]
    trainY = Y[randomSampleIndices]

    testX = np.delete(meansX, randomSampleIndices, 0)
    testY = np.delete(Y, randomSampleIndices, 0)
    trainSeed = 74967
    clf = skl.RandomForestClassifier(random_state=trainSeed)
    clf.fit(trainX, trainY)

    # %%
    # How did it go?
    print("Total Test Accuracy of Edge Means for Size " + str(k), clf.score(testX, testY))
    totalAccuracy.append(clf.score(testX, testY))

    # Make confusion matrix
    predictY = clf.predict(testX)
    cm = confusion_matrix(predictY, testY)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
    disp.plot()
    plt.xticks(rotation=90)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix for Edge Means of Size " + str(k))
    plt.savefig("Confusion_Matrix_Test_" + str(k) + ".png")
    plt.close()

np.savetxt("totalAccuracy.csv", totalAccuracy, delimiter=",")
plt.plot(totalAccuracy)
plt.title("Accuracy By Edge Size")
plt.xlabel("Edge Size")
plt.ylabel("Accuracy")
plt.show()