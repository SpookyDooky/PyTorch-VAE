import pytorch_lightning as pl
from torch.utils.data import DataLoader
from celldata import CellDataset


class CellExperiment(pl.LightningModule):

    def __init__(self, parameters: dict, k_most_significant):
        super().__init__()
        self.parameters = parameters
        self.k_most_significant = k_most_significant

    def train_dataloader(self):
        if self.parameters['dataset'] != 'celldata':  # Just here so that I know when I filled in the wrong dataset
            raise ValueError('Undefined dataset')

        dataset = CellDataset(self.parameters['data_path'], self.k_most_significant)
        train_dataloader = DataLoader(dataset,
                                      batch_size = self.parameters['batch_size'],
                                      shuffle=False,
                                      drop_last=True)  # To drop the last batch that is not completely filled
        return train_dataloader

    def val_dataloader(self):
        if self.parameters['dataset'] != 'celldata':
            raise ValueError('Undefined dataset')

        dataset = CellDataset(self.parameters['data_path'], self.k_most_significant)
        val_dataloader = DataLoader(dataset,
                                    batch_size= 64,
                                    shuffle=False,
                                    drop_last=True)
        return val_dataloader
