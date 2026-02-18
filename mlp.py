import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim import SGD
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

data_folder = '~/data/FMNIST'
fmnist_train = datasets.FashionMNIST(data_folder, download=True, train=True)
x_train = fmnist_train.data
y_train = fmnist_train.targets

fmnist_test = datasets.FashionMNIST(data_folder, download=True, train=False)
x_test = fmnist_test.data
y_test = fmnist_test.targets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float()/255
        x = x.view(-1,28*28)
        self.x, self.y = x, y
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)
    def __len__(self):
        return len(self.x)

class FMNISTNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(28 * 28, 1000)
        #self.batch_norm = nn.BatchNorm1d(1000)
        self.hidden_layer_activation = nn.ReLU()
        #self.dropout = nn.Dropout(0.25)
        self.hidden_to_output_layer = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        #x = self.batch_norm(x)
        x = self.hidden_layer_activation(x)
        #x = self.dropout(x)
        x = self.hidden_to_output_layer(x)
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
    argmaxes = prediction.argmax(dim=1) # Compute if the location of maximum in each row coincides
    s = torch.sum((argmaxes == y).float())/len(y)         # with ground truth
    return s.cpu().numpy()

train_dataset = FMNISTDataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = FMNISTNeuralNet().to(device)
loss_func = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=1e-2)

losses, accuracies, n_epochs = [], [], 5
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

fmnist_test = datasets.FashionMNIST(data_folder, download=True, train=False)
x_test = fmnist_test.data
y_test = fmnist_test.targets

test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

epoch_accuracies = []
for batch in test_dl:
    x, y = batch
    batch_acc = accuracy(x, y, model)
    epoch_accuracies.append(batch_acc)

print(f"Test accuracy: {np.mean(epoch_accuracies)}")

epochs = np.arange(n_epochs) + 1
plt.figure(figsize=(20,3))
plt.subplot(121)
plt.title('Training Loss value over epochs')
plt.plot(epochs, losses)
plt.subplot(122)
plt.title('Training Accuracy value over epochs')
plt.plot(epochs, accuracies)
plt.show()
