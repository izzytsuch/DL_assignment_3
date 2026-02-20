"""
This project is for CSCI 3485, Deep learning, project #3. 
This code was written by Izzy Tsuchitori and Riley Nelson
February 20th, 2026

This project compares MLPs and CNNs on the FashionMNIST datatset, as well as:
-The effect of Kernel size on CNN performance
-The effect of #of Filters on CNN performance
-The impact of Batch Normalization vs Dropout

For each experiement, a plot is created

"""

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchsummary import summary
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable 
#used https://www.geeksforgeeks.org/python/how-to-make-a-table-in-python/ to find prettytable


device = 'cuda' if torch.cuda.is_available() else 'cpu'
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets

class FMNISTDataset(Dataset):
   """
   Convert raw tensors into correct format for experimentation"
   """
   def __init__(self, x, y):
      """
      inputs:
      self
      x- image tensor shape of (N, 28, 28)
      y- label tensor or shape (N,)
      """
       x = x.view(-1, 1, 28, 28)
       x = x.float()/255
       self.x, self.y = x, y
   def __getitem__(self, ix):
       return self.x[ix].to(device), self.y[ix].to(device)
      """ returns a sample from the dataset """
   def __len__(self):
      """ returns total #of samples """
       return len(self.x)

train_dataset = FMNISTDataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)


class MLP(nn.Module):
   """
   MLP model format for testing
   Includes Batch Normalization and Dropout
   """
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(28 * 28, 1000)
        self.batch_norm = nn.BatchNorm1d(1000)
        self.hidden_layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.hidden_to_output_layer = nn.Linear(1000, 10)
    def forward(self, x):
       """ Forward pass of the MLP"""
        #must flatten layers for dataset to match 
        x = torch.flatten(x,1) #had to use chatgpt to find this fix 
        x = self.input_to_hidden_layer(x)
        x = self.batch_norm(x)
        x = self.hidden_layer_activation(x)
        x = self.dropout(x)
        x = self.hidden_to_output_layer(x)
        return x

class CNN(nn.Module):
   """
   CNN model format for testing. Allows for configuring:
   -Filters
   -Kernel size
   -Batch Normalization
   -Dropout
   """
    def __init__(self, 
                 filters=64, 
                 kernel_size=3, 
                 use_batchnorm=False, 
                 use_dropout=False, 
                 dropout_p=0.3):
        super().__init__()

        layers=[]
                    
        #Cov 1
        layers.append(nn.Conv2d(1,filters,kernel_size))
        if use_batchnorm:
           layers.append(nn.BatchNorm2d(filters))
        layers.append(nn.ReLU())
        if use_dropout:
           layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.MaxPool2d(2))

        #Conv 2
        layers.append(nn.Conv2d(filters,filters,kernel_size))
        if use_batchnorm:
           layers.append(nn.BatchNorm2d(filters))
        layers.append(nn.ReLU())
        if use_dropout:
           layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.MaxPool2d(2))
                      
        self.features = nn.Sequential(*layers) 

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(200), #used chat to find lazylinear fix
            nn.ReLU(),
            nn.Linear(200, 10)
        )
    
    def forward(self, x):
       """ forward pass of the CNN """
        x = self.features(x)
        x = self.classify(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train_epoch(model, dataloader, opt, loss_fn):
    """ 
    trains the model for one epoch.
    returns a tuple of the average loss, and the accruacy
    """
    model.train()
    total_loss=0
    samples=0
    correct=0

    for x,y in dataloader:
       opt.zero_grad()
       outputs=model(x)
       loss=loss_fn(outputs,y)
       loss.backward()
       opt.step()

       total_loss+=loss.item()*len(y)
       preds=outputs.argmax(dim=1)
       correct+=(preds==y).sum().item()
       samples+=len(y)                       

    return total_loss/samples, correct/samples



@torch.no_grad()
def accuracy(model, dataloader):
   """ checks arruracy of the model """
    model.eval()
    correct=0
    samples=0

    for x,y in dataloader:
       outputs=model(x)
       preds=outputs.argmax(dim=1)
       correct+=(preds==y).sum().item()
       samples+=len(y)

    return correct/samples



def do_experiment(model, train_dl, test_dl, table, name):
    """
    This is where each experiment is actually run.
    Each experiment is called using do_experiment, which then calls on
    the proper format depending on the model. This then sends the proper data to cuda/cpu
    and then runs and tests each experiment, and then logs accuracy and time into a table.
    """
    model.to(device)
    #separate test and train functions so it doesn't train every time? 
    
    loss_func = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)

    losses=[]
    accuracies=[]
    n_epochs = 5
    start_time = time.perf_counter()
    
    for epoch in range (n_epochs):
       train_loss, train_acc=train_epoch(model, train_dl, opt, loss_func)
       losses.append(train_loss)
       accuracies.append(train_acc)
       print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    train_time = time.perf_counter()-start_time
   
    print(f"Training time for {name}: {train_time:.6f} seconds")


    test_acc = accuracy(model, test_dl)
    print(f"Test accuracy for {name}: {test_acc:.4f}")

    table.add_row([name, f"{train_time:.6f}", f"{test_acc:.4f}", count_params(model)])

    return (losses, accuracies, table, test_acc)


def visualize(n_epochs, losses, accuracies, i, name):
    plt.figure(figsize=(13,3))
    plt.subplot(121)
    plt.title(f'Training Loss value over epochs - {name}')
    plt.plot(np.arange(n_epochs) + 1, losses)
    plt.subplot(122)
    plt.title(f'Training Accuracy value over epochs - {name}') #specify version
    plt.plot(np.arange(n_epochs) + 1, accuracies)
    plt.savefig(f"plot_{i}.png")

def main():
    i = 0 
    table=PrettyTable(["Experiment Name", "Training Time (seconds)", "Test Accuracy", "Number of Parameters"])

    best_test_acc=0
    best_filters=None
    best_kernel=None

    #study 1, compare CNN to MLP

    #CNN compare
    model=CNN().to(device)
    losses, accuracies, table, test_acc=do_experiment(model, train_dl, test_dl, table, name="CNN")
    visualize(5, losses, accuracies, i, name="CNN")
    i+=1
    if test_acc>best_test_acc:
       best_test_acc=test_acc
       best_filters=64
       best_kernel=3
   
    #MLP compare
    mlp = MLP().to(device)
    losses, accuracies, table, test_acc = do_experiment(mlp, train_dl, test_dl, table, name="MLP")
    visualize(5, losses, accuracies, i, name="MLP")
    i += 1

   
    #study 2, vary kernel size
    kernel_size= [2, 3, 5, 7, 9]
    for k in kernel_size:
        model = CNN(kernel_size=k).to(device)
        losses, accuracies, table, test_acc = do_experiment(model, train_dl, test_dl, table, name=(f"kernel size: {k}"))
        visualize(5, losses, accuracies, i, name=(f"kernel size: {k}"))
        i += 1
        if test_acc>best_test_acc:
           best_test_acc=test_acc
           best_filters=64
           best_kernel=k


    #study 3, vary #of filters
    filters = [5, 10, 15, 20, 25]
    for f in filters:
        model = CNN(filters=f).to(device)
        losses, accuracies, table, test_acc = do_experiment(model, train_dl, test_dl, table, name=(f"filter size: {f}"))
        visualize(5, losses, accuracies, i, name=(f"filter size: {f}"))
        i += 1
        if test_acc>best_test_acc:
           best_test_acc=test_acc
           best_filters=f
           best_kernel=3
       
      
    print(f"\nBest architecture selected: Filters={best_filters}, Kernel={best_kernel}")
   
    #study 4, test with Batch norm vs Dropout

    #Batch Norm compare
    model_bn=CNN(filters=best_filters, kernel_size=best_kernel, use_batchnorm=True, use_dropout=False).to(device)
    losses, accuracies, table, test_acc=do_experiment(model_bn, train_dl, test_dl, table, name="Chosen CNN with Batch Normalization")
    visualize(5, losses, accuracies, i, name="Chosen CNN with Batch Normalization")
    i+=1

    #Dropout compare
    model_dropout=CNN(filters=best_filters, kernel_size=best_kernel, use_batchnorm=False, use_dropout=True).to(device)
    losses, accuracies, table, test_acc=do_experiment(model_dropout, train_dl, test_dl, table, name="Chosen CNN with Dropout")
    visualize(5, losses, accuracies, i, name="Chosen CNN with Dropout")
    i+=1
    
    


if __name__ == "__main__":
    main()
