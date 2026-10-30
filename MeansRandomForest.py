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
# <chatGPT>
def compress(data):
    return np.array([
        np.mean(data, axis=(0, 1)),
        np.mean(data[0, :, :], axis=0),   # Top
        np.mean(data[-1, :, :], axis=0),  # Bottom
        np.mean(data[:, 0, :], axis=0),   # Left
        np.mean(data[:, -1, :], axis=0)   # Right
    ])

def compress3(data):
    return np.vstack([
        np.mean(data, axis=(0, 1)),
        data[0, :, :],   # Top
        data[-1, :, :],  # Bottom
        data[1:-1, 0, :],   # Left
        data[1:-1, -1, :]   # Right
    ])

meansX = np.array([np.ndarray.flatten(compress(entry)) for entry in X])
# </chatGPT>

secretSecondX = np.mean(X, axis=(1, 2))
meansX3 = np.array([np.ndarray.flatten(compress3(entry)) for entry in X])

print("Shape of Means Vector:")
print(meansX.shape)

# %%
# Split into training and test data
trainPercent = 0.80
randomSampleIndices = np.random.choice(len(X), size=math.floor(len(X)*trainPercent), replace=False)
trainX = meansX[randomSampleIndices]
secretSecondTrainX = secretSecondX[randomSampleIndices]
trainX3 = meansX3[randomSampleIndices]
trainY = Y[randomSampleIndices]

testX = np.delete(meansX, randomSampleIndices, 0)
secretSecondTestX = np.delete(secretSecondX, randomSampleIndices, 0)
testX3 = np.delete(meansX3, randomSampleIndices, 0)
testY = np.delete(Y, randomSampleIndices, 0)

# print("Shape of Train/Test Datasets:")
# print("trainX", trainX.shape)
# print("secretSecondTrainX", secretSecondTrainX.shape)
# print("trainY", trainY.shape)
# print("secretSecondTestX", secretSecondTestX.shape)
# print("testX", testX.shape)
# print("testY", testY.shape)


# %%
trainSeed = 74967
clf = skl.RandomForestClassifier(random_state=trainSeed)
clf.fit(trainX, trainY)

clf2 = skl.RandomForestClassifier(random_state=trainSeed)
clf2.fit(secretSecondTrainX, trainY)

clf3 = skl.RandomForestClassifier(random_state=trainSeed)
clf3.fit(trainX3, trainY)

# %%
# How did it go?
print("Total Test Accuracy of Edge Means", clf.score(testX, testY))
print("Total Test Accuracy of Only Interior Means", clf2.score(secretSecondTestX, testY))
print("Total Test Accuracy of Full Edge", clf3.score(testX3, testY))

# Make confusion matrix
predictY = clf.predict(testX)
cm = confusion_matrix(predictY, testY)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot()
plt.xticks(rotation=90)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix for Edge Means")
plt.savefig("Confusion_Matrix_1.png")
plt.show()

predictY = clf2.predict(secretSecondTestX)
cm = confusion_matrix(predictY, testY)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot()
plt.xticks(rotation=90)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix for Only Interior Means")
plt.savefig("Confusion_Matrix_2.png")
plt.show()

predictY = clf3.predict(testX3)
cm = confusion_matrix(predictY, testY)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot()
plt.xticks(rotation=90)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix for Full Edges")
plt.savefig("Confusion_Matrix_3.png")
plt.show()
