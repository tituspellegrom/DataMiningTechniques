from preprocessing import preprocess
import pickle
import pandas as pd
from model import DeepFactorizationMachineModel
import torch
import torch.optim as optim
import torch.nn as nn
from data import ExpediaDataset
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm

def get_from_files():
    with open("embedding_dims.txt", "rb") as fp:
        embedding_dims = pickle.load(fp)

    return embedding_dims

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

def computeValidationLoss(device, net, criterion, valloader):
    val_loss=0.0
    for i, data in enumerate(valloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    return val_loss


def main():
    embedding_dims = get_from_files()
    trainset = ExpediaDataset(train=True, filename='df_temporary')
    valset = ExpediaDataset(train=False, filename='df_temporary')

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    val_loader = DataLoader(valset, batch_size=128, shuffle=True)


    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"


    mlp_dims = [200, 200, 200]
    num_feature_columns = 14

    model = DeepFactorizationMachineModel(embedding_dims, 4, mlp_dims, 0.5, num_feature_columns=num_feature_columns)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, 0)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                print()
                running_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor).to(device)
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze(1)).sum().item()

        print('Accuracy per epoch: %d %%' % (100 * correct / total))


    print('Finished Training')



if __name__ == '__main__':
    main()