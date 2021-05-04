from preprocessing import preprocess
import pickle
import pandas as pd
from model import DeepFactorizationMachineModel
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import time
import numpy as np
import torch.utils.data as data_utils
import scipy

def get_from_files():
    print('Importing data...')
    with open("embedding_dims.txt", "rb") as fp:
        embedding_dims = pickle.load(fp)

    with open("groups.txt", "rb") as fp:
        groups = pickle.load(fp)

    data = pd.read_pickle('data_merged.pkl')

    print('Done')
    return data, embedding_dims, groups

def dataframe_to_sparse(df):
    data_sub = df.iloc[0:10000]
    scipy.sparse.csr_matrix(df.values)


def get_dataloaders(data, groups):
    print('creating loaders')
    X = data.iloc[:,:-1]

    X_indices = np.arange(0, len(X))
    y = data.iloc[:,-1:]
    train = data_utils.TensorDataset(torch.Tensor(np.array(X)), torch.Tensor(np.array(y)))
    train_length = len(X)*0.9
    val_length = len(X)-train_length
    trainset, valset = torch.utils.data.random_split(train, [train_length, val_length])
    # gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42)
    # trainset, testset = gss.split(X, y, groups)
    # print('train test set')
    # X_train, X_test, y_train, y_test = train_test_split(X_indices, y, random_state=42, test_size=0.1)
    #
    # X_train = X[X_train]
    # X_test = X[X_test]

    print('Creating dataloaders...')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                             shuffle=False)

    print('Done')
    return trainloader, testloader

def main():
    # data_merged, embedding_dims, groups = preprocess()
    data_merged, embedding_dims, groups = get_from_files()
    trainloader, testloader = get_dataloaders(data_merged, groups)

    mlp_dims = [32, 32]
    model = DeepFactorizationMachineModel(embedding_dims, 4, mlp_dims, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = model(input)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
    main()