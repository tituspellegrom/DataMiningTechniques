import pickle
import pandas as pd
from model import DeepFactorizationMachineModel
import torch
import torch.optim as optim
import torch.nn as nn
from data import ExpediaDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import csv
from tqdm import tqdm
import numpy as np
import cProfile

torch.manual_seed(5)

def get_from_files(path=None):
    if path == None:
        with open("Data/embedding_dims.txt", "rb") as fp:
            embedding_dims = pickle.load(fp)
    else:
        with open(f"{path}embedding_dims.txt", "rb") as fp:
            embedding_dims = pickle.load(fp)

    return embedding_dims

def sampler_weights(data_name, path=None):
        if path ==None:
            target = pd.read_pickle(f'Data/{data_name}_train.pkl')['label']
        else:
            target = pd.read_pickle(f'{path+data_name}_train.pkl')['label']

        class_sample_count = np.unique(target, return_counts=True)[1]
        weight = 1. / class_sample_count

        return weight[target]

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

def computeValidationLoss(device, net, criterion, valloader):
    val_loss=0.0
    for i, data in enumerate(tqdm(valloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

    return val_loss


def main(path=None):
    embedding_dims = get_from_files(path=path)
    trainset = ExpediaDataset(train=True, data_name='df_temporary', path=path)
    valset = ExpediaDataset(train=False, data_name='df_temporary', path=path)

    samples_weight = sampler_weights(data_name='df_temporary', path=path)
    sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # train_loader = DataLoader(trainset, batch_size=256, sampler=sampler_train, num_workers=2)
    val_loader = DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    mlp_dims = [200, 200, 200]
    # mlp_dims = [32, 32]
    # mlp_dims = [128, 128]
    num_feature_columns = 14

    model = DeepFactorizationMachineModel(embedding_dims, 4, mlp_dims, 0.2, num_feature_columns=num_feature_columns)
    model.to(device)
    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.40872054e-06, 9.02983457e-06, 1.49875603e-05]).to(device))
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(30):  # loop over the dataset multiple times
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
            if i % 3467 == 3466:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 3467))
                running_loss = 0.0
                print(loss.item())
                print(np.unique(labels.cpu(), return_counts=True)[1])
                print(torch.bincount(torch.max(outputs.data, 1)[1]))
                # for name, param in model.named_parameters():
                #     print(name, param.grad.norm())

        correct = 0
        total = 0

        model.eval()

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