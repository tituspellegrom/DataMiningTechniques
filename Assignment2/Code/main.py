from preprocessing import preprocess
import pickle
import pandas as pd
from model import DeepFactorizationMachineModel
import torch
import torch.optim as optim
import torch.nn as nn
from data import SparseDataset
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm

def get_from_files():
    print('Importing data...')
    with open("df_temporary_embedding_dims.csv", "r") as fp:
        embedding_dims = list(csv.reader(fp))

    with open("groups.txt", "rb") as fp:
        groups = pickle.load(fp)

    print('Done')

    embedding_dims = [int(emb) for emb in embedding_dims[0]]
    return embedding_dims, groups


def main():
    embedding_dims, groups = get_from_files()
    dataset = SparseDataset('df_temporary_data_merged.npz')
    trainloader = DataLoader(dataset, batch_size=128, shuffle=True)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    mlp_dims = [200, 200, 200]

    TEMP_embedding_dims = [332786, 35, 232, 231, 140822, 28417]
    # TODO correct retrieval of embedding dims (should be max + 1)
    model = DeepFactorizationMachineModel(TEMP_embedding_dims, 4, mlp_dims, 0.5)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, 0)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

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

            # TODO validation loss

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         inputs, labels = data
    #         # calculate outputs by running images through the network
    #         outputs = model(input)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))

if __name__ == '__main__':
    main()