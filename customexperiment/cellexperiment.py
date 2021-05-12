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
        self.count = 0
        self.batch_index = 0
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if batch_idx < self.batch_index:
            self.batch_index = batch_idx
            print('\nlogging epoch stats')

        self.batch_index = batch_idx
        newdata2 = batch.clone().detach().float()
        results = self.forward(newdata2)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.train_samples,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        #train_loss = {**train_loss, 'Learning-rate': torch.FloatTensor(self.schedulers[0].get_lr())}
        self.logger.experiment.log({key: val.item() for key, val in dict(train_loss).items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # self.current_device = self.device
        batch = batch.type(torch.FloatTensor)
        results = self.forward(batch.cuda())
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.val_samples,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        val_log = {'validation loss': val_loss['loss'].data}
        self.logger.experiment.log({key: val.item() for key, val in dict(val_log).items()})
        return val_loss


    @data_loader
    def train_dataloader(self):
        if self.params['dataset'] != 'celldata':  # Just here so that I know when I filled in the wrong dataset
            raise ValueError('Undefined dataset')

        dataset = CellDataset(self.params['input_size'], root=self.params['data_path'], split="train")
        train_dataloader = DataLoader(dataset,
                                      batch_size = self.params['batch_size'],
                                      shuffle=True,
                                      drop_last=True)  # To drop the last batch that is not completely filled
        self.train_samples = len(train_dataloader)
        return train_dataloader

    @data_loader
    def val_dataloader(self):
        if self.params['dataset'] != 'celldata':
            raise ValueError('Undefined dataset')

        dataset = CellDataset(self.params['input_size'], root=self.params['data_path'], split="test")
        val_dataloader = DataLoader(dataset,
                                    batch_size= 144,
                                    shuffle=False,
                                    drop_last=True)
        self.val_samples = len(val_dataloader)
        return val_dataloader

    def configure_optimizers(self):
        print(self.model.parameters())
        current_optimizer = optim.Adam(self.model.parameters(),
                                       lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(current_optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                learning_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizers[0],
                                                               gamma=self.params['scheduler_gamma'])
                self.schedulers.append(learning_scheduler)
                return self.optimizers, self.schedulers
        except:
            return self.optimizers