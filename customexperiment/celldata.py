import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class CellDataset(Dataset):

    def __init__(self, root: str, split: str):
        self.data_folder = root
        self.data = np.load(self.data_folder + 'cell_expression.npy')
        print(len(self.data))
        self.k_highest_variance = 32
        if self.k_highest_variance > 0:
            self.filter_out_insignificant()

        print('I ran')
        #k = 13
        #temp = np.split(self.data, k)
        #train=temp[0]
        #for x in range(1,10):
        #    train = np.concatenate((train, temp[x]))
#
        #print('making validation set')
        #validate = temp[10]
        #for x in range(10, 13):
        #    validate = np.concatenate((validate, temp[x]))
#
        #if split == 'train':
        #    self.data = train
        #else:
        #    self.data = validate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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