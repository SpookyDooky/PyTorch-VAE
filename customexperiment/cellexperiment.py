import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from customexperiment.celldata import CellDataset
from models import BaseVAE
from torch import optim
from utils import data_loader
from models.types_ import *


class CellExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(CellExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.k_most_significant = 300
        self.current_device = None
        self.hold_graph = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        newdata2 = batch.clone().detach().float().unsqueeze(1)
        results = self.model(newdata2)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.sample_length,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # self.current_device = self.device
        print(batch)
        #newdata2 = batch.clone().detach().float()[None,...].reshape(-1, 1, 10)
        newdata2 = batch.clone().detach().float().unsqueeze(1)
        #newdata = torch.tensor(batch).float()[None, ...]
        print(newdata2)
        results = self.model(newdata2)
        print('?hhhhhhhhhhhhhhhhhhhhhhh')
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.sample_length,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_end(self, outputs):
        print('????')
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    @data_loader
    def train_dataloader(self):
        if self.params['dataset'] != 'celldata':  # Just here so that I know when I filled in the wrong dataset
            raise ValueError('Undefined dataset')

        dataset = CellDataset(root=self.params['data_path'], split="train")
        train_dataloader = DataLoader(dataset,
                                      batch_size = self.params['batch_size'],
                                      shuffle=False,
                                      drop_last=True)  # To drop the last batch that is not completely filled
        return train_dataloader

    @data_loader
    def val_dataloader(self):
        if self.params['dataset'] != 'celldata':
            raise ValueError('Undefined dataset')

        dataset = CellDataset(root=self.params['data_path'], split="test")
        val_dataloader = DataLoader(dataset,
                                    batch_size= 64,
                                    shuffle=False,
                                    drop_last=True)
        self.sample_length = len(val_dataloader)
        return val_dataloader

    def configure_optimizers(self):
        current_optimizer = optim.Adam(self.model.parameters(),
                                       lr=self.params['LR'],
                                       weight_decay=self.params['weight_decay'])
        optimizers = []
        schedulers = []
        optimizers.append(current_optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                learning_scheduler = optim.lr_scheduler.ExponentialLR(optimizers[0],
                                                                      gamma=self.params['scheduler_gamma'])
                schedulers.append(learning_scheduler)
                return optimizers, schedulers
        except:
            return optimizers
