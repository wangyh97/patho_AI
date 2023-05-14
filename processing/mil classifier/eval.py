import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import time

'''
similar to train_tcga, use testing set to eval
'''