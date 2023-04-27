from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int)
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(args.scale, config['batch_size'], **config['dataset'])
    '''
    dataset: Class DataSetWrapper
        function:
            get_data_loaders: get train_loader & valid_loader
                
        passed into simclr.train(), call function: get_data_loaders.
    '''
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
