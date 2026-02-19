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
    def __init__(self, filters=64, kernel_size=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ) #.to(device)??

        self.classify = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(3200, 200),
            nn.LazyLinear(200), #used chat to find lazylinear fix
            nn.ReLU(),
            nn.Linear(200, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x

def train_batch(x, y, model, opt, loss_fn):
    model.train()

    opt.zero_grad()                    # Flush memory
    batch_loss = loss_fn(model(x), y)  # Compute loss
    batch_loss.backward()              # Compute gradients
    opt.step()                         # Make a GD step

    return batch_loss.detach().cpu().numpy()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()

    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float())/len(y)

    return s.cpu().numpy()

def do_experiment(model, train_dl, test_dl):
    #model = model.to(device)
    #separate test and train functions so it doesn't train every time? 
    loss_func = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)

    losses, accuracies, n_epochs = [], [], 5
    start_time = time.perf_counter()
    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")

        epoch_losses, epoch_accuracies = [], []
        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, opt, loss_func)
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        for batch in train_dl:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        epoch_accuracy = np.mean(epoch_accuracies)

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.6f} seconds")
    
    epoch_accuracies = []
    for batch in test_dl:
        x, y = batch
        batch_acc = accuracy(x, y, model)
        epoch_accuracies.append(batch_acc)
    print(f"Test accuracy: {np.mean(epoch_accuracies)}")

    
    #plt.show()
    return n_epochs, losses, accuracies

def visualize(n_epochs, losses, accuracies, i, name):
    plt.figure(figsize=(13,3))
    plt.subplot(121)
    plt.title(f'Training Loss value over epochs - {name}')
    plt.plot(np.arange(n_epochs) + 1, losses)
    plt.subplot(122)
    plt.title(f'Training Accuracy value over epochs - {name}') #specify version
    plt.plot(np.arange(n_epochs) + 1, accuracies)
    plt.savefig(f"plot_{i}.png")


def count_parameters(model):
    return model.parameters()

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
    model = CNN().to(device)
    n_epochs, losses, accuracies = do_experiment(model, train_dl, test_dl)
    visualize(n_epochs, losses, accuracies, i, name="CNN")
    i += 1

   #call function 
    mlp = MLP().to(device)
    n_epochs, losses, accuracies = do_experiment(mlp, train_dl, test_dl)
    visualize(n_epochs, losses, accuracies, i, name="MLP")
    i += 1
    
    #study 2
    kernel_size= [2, 3, 5, 7, 9]
    for k in kernel_size:
        model = CNN(kernel_size=k).to(device)
        n_epochs, losses, accuracies = do_experiment(model, train_dl, test_dl)
        visualize(n_epochs, losses, accuracies, i, name=(f"kernel size: {k}"))
        i += 1


    #study 3
    filters = [5, 10, 15, 20, 25]
    for f in filters:
        model = CNN(filters=f).to(device)
        n_epochs, losses, accuracies = do_experiment(model, train_dl, test_dl)
        visualize(n_epochs, losses, accuracies, i, name=(f"filter size: {f}"))
        i += 1

    #study 4


if __name__ == "__main__":
    main()
