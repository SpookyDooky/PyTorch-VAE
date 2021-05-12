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
        self.filter_out_insignificant()
        #self.normalize_data()
        self.normalize_per_feature()
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
        self.lowest_value = 2.728561 # negative value but made positive here to negate negative values
        self.biggest_value = 48.944336
        self.data = self.data + self.lowest_value
        self.data = self.data / self.biggest_value
        print('done')

    def normalize_per_feature(self):
        #min_per_column = np.min(self.data, axis=0)
        #for index in range(len(self.data[0])):
        #    factor_positive = -1 * min_per_column[index]
        #    for row_idx in range(len(self.data)):
        #        self.data[row_idx][index] = self.data[row_idx][index] + factor_positive

        max_per_column = np.max(self.data, axis=0)
        for index in range(len(self.data[0])):
            #self.data[index] = self.data[index] / max_per_column[index]
            #for row_idx in range(len(self.data)):
            #    self.data[row_idx][index] = self.data[row_idx][index] / max_per_column[index]
            self.data[:,index] = self.data[:,index] / max_per_column[index]
            #print(self.data[index])
