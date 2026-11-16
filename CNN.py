## IMPORTS ######################################################################
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import copy


## CNN ##########################################################################
#################################################################################
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    
def train_CNN(Xtrain, ytrain, Xval, yval, num_classes, epochs = 20, batch_size = 128, lr = 1e-3):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    train = TensorDataset(torch.tensor(Xtrain, dtype = torch.float32), torch.tensor(ytrain, dtype = torch.long))
    val = TensorDataset(torch.tensor(Xval, dtype = torch.float32), torch.tensor(yval, dtype = torch.long))
    trainloader = DataLoader(train, batch_size=batch_size, shuffle = True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle = False)
    
    model = CNN(in_channels=Xtrain.shape[1], num_classes = num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr)
    
    train_losses = []
    val_losses = []
    val_f1s = []
    
    best_f1 = -1
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for Xbatch, ybatch in trainloader:
            Xbatch = Xbatch.to(device)
            ybatch = ybatch.to(device)
            optimizer.zero_grad()
            outputs = model(Xbatch)
            loss = criterion(outputs, ybatch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # validation
        model.eval()
        preds = []
        truths = []
        val_loss = 0

        with torch.no_grad():
            for Xbatch, ybatch in valloader:
                Xbatch = Xbatch.to(device)
                ybatch = ybatch.to(device)
                outputs = model(Xbatch)
                loss = criterion(outputs, ybatch)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs, dim =1)
                preds.extend(pred.cpu().numpy())
                truths.extend(ybatch.cpu().numpy())
            
        # Metrics
        train_loss = (total_loss/len(trainloader))
        val_loss = (val_loss/len(valloader))
        f1 = f1_score(truths, preds, average = "weighted", zero_division=0)
        train_losses.append(train_loss)
        val_losses.append(val_loss)  
        val_f1s.append(f1)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train = {train_loss:.5f}")
        print(f"Val = {val_loss:.5f}")
        print(f"F1 = {f1:.4f}")  
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            print(f"New best F1 = {best_f1:.4f}")
    
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_f1": val_f1s
    }
    
    if best_state is not None: model.load_state_dict(best_state)
    
    return model, history
    
def test_CNN(model, Xtest, ytest, classes, name):
    device = next(model.parameters()).device
    test = TensorDataset(torch.tensor(Xtest, dtype = torch.float32), torch.tensor(ytest, dtype = torch.long))
    test_loader = DataLoader(test, batch_size = 32, shuffle = False)
    
    preds = []
    truths = []
    model.eval()
    
    with torch.no_grad():
        for Xbatch, ybatch in test_loader:
            Xbatch = Xbatch.to(device)
            outputs = model(Xbatch)
            pred = torch.argmax(outputs, dim = 1)
            preds.extend(pred.cpu().numpy())
            truths.extend(ybatch.numpy())
            
    f1 = f1_score(truths, preds, average = "weighted", zero_division=0)
    cm = confusion_matrix(truths, preds)
            
    print(f"\nbTest F1: {f1:.5f}")
    print(classification_report(truths, preds))
    
    # Plot Confusion
    cm_percents = cm.astype(float)
    cm_percents /= cm_percents.sum(axis=1, keepdims=True)
    plt.figure(figsize = (8,8))
    plt.imshow(cm_percents, cmap = "viridis", vmin = 0, vmax = 1)
    for i in range(len(classes)):
        plt.text(i, i, f"{100*cm_percents[i,i]:.1f}", color = "firebrick", ha = "center", va = "center", fontweight = "bold")
    plt.colorbar()
    plt.title(f"{name} Confusion Matrix")   
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{name}_CM.png")
    plt.close() 
    
    return f1, cm

def trainPlot(history, name):
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label = "Train")
    plt.plot(history["val_loss"], label = "Validation")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history["val_f1"])
    plt.title("Validation F1")
    plt.tight_layout()
    plt.savefig(f"{name}_ValF1.png")
    plt.close()