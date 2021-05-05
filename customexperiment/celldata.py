import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class CellDataset(Dataset):

    def __init__(self, input_size, root: str, split: str):
        self.data_folder = root
        self.split = split
        self.data = np.load(self.data_folder + 'cell_expression.npy')
        self.k_highest_variance = input_size
        self.filter_out_insignificant()
        self.train = self.data[0:10000:1]
        self.test = self.data[13000:16000:1]
        self.count = 0

    def __len__(self):
        if self.split == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        self.count = self.count + 1
        return self.data[index]

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