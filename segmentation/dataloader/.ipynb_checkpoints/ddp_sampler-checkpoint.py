import torch
from torch.utils import data

def getDataLoaders(config,train_dataset,val_dataset,rank=0,world_size=1):
    dist_sampler = data.distributed.DistributedSampler(
        train_dataset,
        rank = rank,
        num_replicas = world_size
    )
    train_batch_size = config.getint('DATA_LOADER','train_batch_size')
    val_batch_size = config.getint('DATA_LOADER','val_batch_size')
    num_workers = config.getint('DATA_LOADER','num_workers_cpu')
    # loaders
    train_loader = data.DataLoader(train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=dist_sampler,
        pin_memory=False
    )
    val_loader = data.DataLoader(train_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    return train_loader, val_loader

