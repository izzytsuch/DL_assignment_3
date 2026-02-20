import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import numpy as np
from torchsummary import summary
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable 
#used https://www.geeksforgeeks.org/python/how-to-make-a-table-in-python/ to find prettytable


#next steps:
#1. implement model_parameters function specified in lab
#2. run BN and droupout experiment 
#3. split up do_experiment function into separate for training 
#.  and testing?
#4. clean up visualization

#notes:
#i've used a lot of the code that we were given in lectures, but
#it's a little messy so sorry if it's confusing!

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets

class FMNISTDataset(Dataset):
   def __init__(self, x, y):
       x = x.view(-1, 1, 28, 28)
       x = x.float()/255
       self.x, self.y = x, y
   def __getitem__(self, ix):
       return self.x[ix].to(device), self.y[ix].to(device)
   def __len__(self):
       return len(self.x)

train_dataset = FMNISTDataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(28 * 28, 1000)
        self.batch_norm = nn.BatchNorm1d(1000)
        self.hidden_layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.hidden_to_output_layer = nn.Linear(1000, 10)
    def forward(self, x):
        #must flatten layers for dataset to match 
        x = torch.flatten(x,1) #had to use chatgpt to find this fix 
        x = self.input_to_hidden_layer(x)
        x = self.batch_norm(x)
        x = self.hidden_layer_activation(x)
        x = self.dropout(x)
        x = self.hidden_to_output_layer(x)
        return x

class CNN(nn.Module):
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
        x = self.features(x)
        x = self.classify(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train_epoch(x, y, model, opt, loss_fn):
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

    return loss/samples, correct/samples



@torch.no_grad()
def accuracy(model, dataloader):
    model.eval()
    correct=0
    samples=0

    for x,y in dataloader:
       total_loss+=loss.item()*len(y)
       preds=outputs.argmax(dim=1)
       correct+=(preds==y).sum().item()
       samples+=len(y)        

    return correct/samples



def do_experiment(model, train_dl, test_dl, table, name):
    model.to(device)
    #separate test and train functions so it doesn't train every time? 
    
    loss_func = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)

    loss=[]
    acc=[]
    losses, accuracies, n_epochs = [], [], 5
    start_time = time.perf_counter()
    
    for epoch in range (n_epochs):
       train_loss, train_acc=train_epoch(model, train_dl, train_dl, loss_func)
       losses.append(train_loss)
       acc.append(train_acc)
       print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    final_time = time.perf_counter()
    train_time = final_time-start_time
    print(f"Training time for {name}: {train_time:.6f} seconds")

    epoch_accuracies = []
    for batch in test_dl:
        x, y = batch
        batch_acc = accuracy(x, y, model)
        epoch_accuracies.append(batch_acc)
    print(f"Test accuracy for {name}: {np.mean(epoch_accuracies)}")

    table.add_row([name, f"{train_time:.6f}", f"{test_acc:.4f}", count_params(model)])

    return (losses, accuracies, table)


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
    #study 1
    # model = nn.Sequential(
    # nn.Conv2d(1, 64, kernel_size=3),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Conv2d(64, 128, kernel_size=3),
    # nn.ReLU(),
    # nn.MaxPool2d(2),
    # nn.Flatten(),
    # nn.Linear(3200, 200),
    # nn.ReLU(),
    # nn.Linear(200, 10)
    # ).to(device)
    table = PrettyTable(["Experiment Name", "Training Time (seconds)", "Test Accuracy"]) 
    
    model = CNN().to(device)
    n_epochs, losses, accuracies, curr_table = do_experiment(model, train_dl, test_dl, table, name="CNN")
    visualize(n_epochs, losses, accuracies, i, name="CNN")
    i += 1

   #call function 
    mlp = MLP().to(device)
    n_epochs, losses, accuracies, curr_table = do_experiment(mlp, train_dl, test_dl, table, name="MLP")
    visualize(n_epochs, losses, accuracies, i, name="MLP")
    i += 1
    
    #study 2
    kernel_size= [2, 3, 5, 7, 9]
    for k in kernel_size:
        model = CNN(kernel_size=k).to(device)
        n_epochs, losses, accuracies, curr_table = do_experiment(model, train_dl, test_dl, table, name=(f"kernel size: {k}"))
        visualize(n_epochs, losses, accuracies, i, name=(f"kernel size: {k}"))
        i += 1


    #study 3
    filters = [5, 10, 15, 20, 25]
    for f in filters:
        model = CNN(filters=f).to(device)
        n_epochs, losses, accuracies, curr_table = do_experiment(model, train_dl, test_dl, table, name=(f"filter size: {f}"))
        visualize(n_epochs, losses, accuracies, i, name=(f"filter size: {f}"))
        i += 1
    
    print(curr_table)
    #study 4


if __name__ == "__main__":
    main()
