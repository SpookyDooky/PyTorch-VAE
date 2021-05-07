import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import sys
import torch

class CellDataset(Dataset):

    def __init__(self, input_size, root: str, split: str):
        self.data_folder = root
        self.split = split
        self.data = np.load(self.data_folder + 'cell_expression.npy')
        self.k_highest_variance = input_size
        #self.filter_out_insignificant()
        #self.normalize_data()
        self.train = self.data[0:15000:1]
        self.test = self.data[15000:18000:1]

    def __len__(self):
        if self.split == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        if self.split == 'train':
            return self.train[index]
        else:
            return self.test[index]

    def filter_out_insignificant(self):
        data = pd.read_csv(self.data_folder + 'gene_data.csv')
        dataframe = pd.DataFrame(data, columns=['gene', 'dispersions_norm'])
        dataframe.sort_values(by=['dispersions_norm'], inplace=True, ascending=False)
        index_array = dataframe.index.to_numpy()[0:self.k_highest_variance:1]
        temp = np.zeros((len(self.data), self.k_highest_variance))
        count = 0
        for significant_index in np.nditer(index_array):
            temp[:,count] = self.data[:,significant_index]
            count = count + 1
        self.data = temp

    def normalize_data(self):
        self.norm_values = np.zeros((len(self.data[0]), 2), dtype=float)
        for x in range(self.k_highest_variance):
            mean_val = torch.mean(torch.FloatTensor(self.data[0::1][x]))
            self.norm_values[x][0] = mean_val.data
            std = torch.std(torch.FloatTensor(self.data[0::1][x]), True)
            self.norm_values[x][1] = std

        for idx1, idx2 in np.ndenumerate(self.data):
            self.data[idx1] = (self.data[idx1] - self.norm_values[idx1[1]][0]) / self.norm_values[idx1[1]][1]


