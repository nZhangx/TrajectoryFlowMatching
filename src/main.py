# training script for lightning model
"""
run via:
(changing data and model)
python main.py data=data/data3.yaml model=model/model3.yaml

to test:
python main.py skip_training=true
"""
import pytorch_lightning as pl
import torch
from torch import nn
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate

from pytorch_lightning.strategies import DDPStrategy

@hydra.main(config_path="conf", config_name="config")
def train_model(cfg):
    # set seed
    pl.seed_everything(cfg.seed)

    print(cfg)
    if 'memory' in cfg['model_module'].keys(): # if is key
        memory = cfg['model_module']['memory']
        cfg.data_module.memory = memory

    data_module = instantiate(cfg.data_module)

    # correct dim
    x_dim = data_module.dims[0]
    if 'dim' in cfg.model_module.keys():
        cfg.model_module.dim = x_dim
    elif 'input_dim' in cfg.model_module.keys():
        cfg.model_module.input_dim = x_dim
        cfg.model_module.output_dim = x_dim


    if 'treatment_cond' in cfg.model_module.keys():
        # for conditional models, need this to configure
        cfg.model_module.treatment_cond = len(data_module.cond_headings)

    model = instantiate(cfg.model_module)

    # conditional models need train_consecutive false!
    if not('Cond' in model.naming):
        cfg.data_module.train_consecutive = True
        data_module = instantiate(cfg.data_module)
    else:
        cfg.data_module.train_consecutive = False
        data_module = instantiate(cfg.data_module)
    
    wandb_config = {key: value for key, value in cfg.model_module.items() if key not in ['_target_']}
    wandb_config['model'] = model.naming
    wandb_config['data'] = data_module.naming
    wandb_config['mode'] = 'batch_run'
    wandb_config['x_headings'] = data_module.x_headings
    wandb_config['cond_headings'] = data_module.cond_headings
    wandb_config['t_headings'] = data_module.t_headings
    wandb_config['seed'] = cfg.seed

    if cfg.wandb_logging and not(cfg.skip_training):
        wandb.init(project="clinical_trajectory",
                    dir = '/home/mila/x/xi.zhang/scratch/shung_ICU/wandb_log/',
                    config = wandb_config
                    )
        wandb_logger = WandbLogger()

    ckpt_savedir = '/home/mila/x/xi.zhang/scratch/shung_ICU/checkpoints/'+model.naming+'_'+data_module.naming+'/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)

    checkpoint_callback = ModelCheckpoint(
        dirpath='/home/mila/x/xi.zhang/scratch/shung_ICU/checkpoints/'+model.naming+'_'+data_module.naming+'/',
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )

    strategy_ddps = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_time=cfg.max_time, 
        check_val_every_n_epoch=50,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger if cfg.wandb_logging else None,
        limit_train_batches=0 if cfg.skip_training else 1.0,
        strategy=strategy_ddps,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    wandb.finish()

def main():
    train_model()

if __name__ == '__main__':
    main()