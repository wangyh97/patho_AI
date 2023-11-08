import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve,f1_score,confusion_matrix
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import time

'''
similar to train_tcga, use testing set to eval
changes:
    test_index: grouping['val_list'].index --> grouping['test_list'].index
    del training_index
    del saving_path
    del train()
    model loading path:../init.pth --> pth file path
    change to eval mode:
        model.eval()
        with torch.no_grad()
    del arg: epoch;run
    result saving file: result_{run}.txt  -->  test_result_{run}.txt
    
TODO:remove f1 score
'''


# load data

# =====>>>> current working dir is where the bash file located, not the executed py file <<<<==============

def load_features(scale, layer, load):
    # load PtRes / Ptsimclr features
    assert load in ['none', 'TCGA_high', 'TCGA_low',
                    'c16_high'], 'invalid load, should be selected from [''none'',''TCGA_high'',''TCGA_low'',''c16_high'']'
    if load == 'none':
        features = np.load(f'../../data/pretrained_resnet{layer}/{scale}X_full_slide_features_PtRes{layer}.npy',
                           allow_pickle=True).item()
        print(f'load features from :  ../../data/pretrained_resnet{layer}/{scale}X_full_slide_features_PtRes{layer}.npy')
    else:
        features = np.load(f'../../data/{load}/{scale}X_full_slide_features.npy',
                           allow_pickle=True).item()
        print(f'load features from : ../../data/{load}/{scale}X_full_slide_features.npy')
    tv_t = np.load(f'../../config/data_segmentation_csv/{scale}X_tv_grouping.npy', allow_pickle=True).item()
    full = np.load(f'../../config/data_segmentation_csv/{scale}X_full.npy', allow_pickle=True).item()
    full_index = list(full['full_list'].index)
    test_index = list(tv_t['test_list'].index)  # 用于独立验证的test的数据的原始index
    index_dict = {ind: i for i, ind in enumerate(full_index)}
    return features, test_index, index_dict


def timer(func):
    def wrapper(*args, **kws):
        tick = time.time()
        func(*args, **kws)
        print(f'{func.__name__} comsumes {time.time() - tick} s')

    return wrapper


def set_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    print('seed set')


#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


def get_bag_feats(features, index, index_dict, args):
    """
    example of label/feature : get_bag_feats(test_index[2],args)
    label:[1. 0.]
    feature:[[2.5330880e+00 1.6554745e-01 8.0470458e-02 ... 1.3655423e+00
      5.7236932e-02 5.2817654e-02]
     [2.8611205e+00 2.9038048e-01 6.0187571e-02 ... 1.8358935e+00
      5.3482568e-01 6.2871813e-03]
     [3.3081994e+00 7.3715396e-02 1.2616467e+00 ... 1.9259404e+00
      1.4681002e-01 2.8594225e-03]
     ...
     [2.9693909e+00 3.2868910e-01 1.3055435e-01 ... 2.4260533e+00
      1.7651926e-01 1.2930447e-01]
     [2.8800142e+00 3.0109720e-02 8.2876140e-01 ... 2.4528553e+00
      5.6700967e-03 0.0000000e+00]
     [1.4685658e+00 1.6393182e-01 6.0487707e-04 ... 1.2453270e+00
      0.0000000e+00 4.1464632e-03]]
    """
    label_dict = {'L': 0, 'H': 1}
    feats_og = pd.DataFrame(features[f'index{index_dict[index]}'][2])
    feats = shuffle(feats_og).reset_index(drop=True).to_numpy()
    label_og = label_dict[features[f'index{index_dict[index]}'][1]]  # transformed label in form of int,[0,1]

    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = label_og
    else:
        if int(label_og) <= (len(label) - 1):
            label[int(label_og)] = 1
    return label, feats


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(features, test_df, milnet, criterion, index_dict, args):
    """
    demo of values of intermediate variables:

    original label:tensor([[0., 1.]], device='cuda:0'),shape of feats:torch.Size([1, 53, 512])
    shape of feats after view:torch.Size([53, 512])
    shape of ins_pred:torch.Size([53, 2]),original bag_pred:tensor([[-2.9899,  2.7934],
            [-3.7077,  3.5751],
            [-3.6781,  3.7495],
            [-3.8786,  3.5615]], device='cuda:0'),shape of original bag_pred:torch.Size([4, 2])
    max pred:tensor([ 0.9809, -2.4702], device='cuda:0'),bag pred after mean:tensor([-3.5636,  3.4199], device='cuda:0')
     Testing bag [0/73] bag loss: 0.9777
    test laels:[0. 1.],test prediction : [array([0.02755601, 0.9683193 ], dtype=float32)]
    first 5 class_pred_bag:[1. 0. 0. 0. 0.],
    first 5 test_pred:[[0. 1.]
     [0. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]],
     first 5 labels:[[0. 1.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]]
     first 5 single char labels:[1, 0, 0, 0, 0],first 5 single char pred:[1, 0, 0, 1, 0]
     """
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    #     Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(features, test_df[i], index_dict, args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_prediction = torch.mean(bag_prediction, dim=0)
            # bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            # loss = 0.5 * bag_loss + 0.5 * max_loss
            # l2_loss = 0
            # if args.reg == True:
            #     for param in milnet.parameters():
            #         l2_loss += torch.norm(param)
            # loss += args.reg_coef*l2_loss
            # total_loss = total_loss + loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal, fpr, tpr, precision, recall = multi_label_roc(test_labels, test_predictions,
                                                                                    args.num_classes, pos_label=1)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    # get confusion matrix
    one_hot_labels = [np.argmax(i) for i in test_labels]
    one_hot_preds = [np.argmax(i) for i in test_predictions]
    c = confusion_matrix(one_hot_labels, one_hot_preds)
    # tn, fp, fn, tp = c.ravel()
    f1 = f1_score(one_hot_labels, one_hot_preds)
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)
    return avg_score, auc_value, c, fpr, tpr, precision, recall, f1 # fpr,tpr for ROC curve plotting,precision,recall for PR-curve plotting


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        precision, recall, thr_pr = precision_recall_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal, fpr, tpr, precision, recall


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def load_model(args, weight_path):
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    weight = torch.load(weight_path)

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                   dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    milnet = torch.nn.DataParallel(milnet)
    milnet = milnet.cuda()
    milnet.load_state_dict(weight, strict=True)
    return milnet


def weight_parser(weight_path,fold):
    path= Path(weight_path)
    pic_path = weight_path.replace('pth','png')
    metric = path.stem
    if fold == -1:
        lrwdT = path.parents[0].name
    else:
        lrwdT = path.parents[1].name
    _, lr, weight, _ = lrwdT.split('_')
    return lr, weight, metric, pic_path


def metrics_visulization(fpr, tpr, roc_auc, precision, recall, confusion_matrix, saving_path, figsize=(10, 20)):
    # roc curve: fpr,tpr
    # metrics
    # F1 score
    # TODO: add metrics
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    axes[0].set_title('ROC curve')
    axes[0].plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % roc_auc)
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc="lower right")

    axes[1].set_title('P-R curve')
    axes[1].plot(recall, precision, color='darkorange')
    axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1].set_xlabel('recall')
    axes[1].set_ylabel('precision')

    axes[2].set_title('confusion matrix')
    axes[2].imshow(confusion_matrix,interpolation='nearest')
    axes[2].set_xlabel('predicted label')
    axes[2].set_ylabel('true label')
    axes[2].set_xticks(np.arange(2), [0, 1])
    axes[2].set_yticks(np.arange(2), [0, 1])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

    plt.savefig(saving_path)
    plt.close()


@timer
def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')

    # select features
    parser.add_argument('--scale', type=int,
                        help='select magnificant scale of the extracted patches,select form [10,20]')
    parser.add_argument('--layer', type=int, help='select resnet18/50 as feature extractor,[18]')
    parser.add_argument('--load', type=str, default='none',
                        help='select pretrained embedder,[none,TCGA_high,TCGA_low,c16_high]')

    # model structure
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index',default=0, type=str, help='GPU ID(s) [0]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')

    # select the model
    parser.add_argument('--description', type=str,
                        help='short description for the trial, saving results in file: results_{description}.txt')
    parser.add_argument('--fold', default=0, type=str, help='specific fold selected, run all models saved')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    set_seed()

    # TODO: working dir
    log_path = os.path.join(args.description, 'metrics.log')

    # initiate logger: show info in stream & save in file
    log = logging.getLogger('recorder')
    log.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    handler1 = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler1.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    handler1.setFormatter(formatter1)
    log.addHandler(handler)
    log.addHandler(handler1)

    # load data
    features, test_index, index_dict = load_features(args.scale, args.layer, args.load)
    # load weight lists
    fold_list = args.fold
    print(f'fold:{args.fold}')
    for fold in fold_list:
        weight_path = os.path.join(args.description, f'weights/*/fold{fold}/*.pth')
        weight_lists = glob.glob(weight_path)
        for weight_path in weight_lists:
            lr, weight, metric, pic_path = weight_parser(weight_path,fold)
            milnet = load_model(args, weight_path)
            criterion = nn.BCEWithLogitsLoss()
            milnet.eval()
            with torch.no_grad():
                avg_score, aucs, conf_mat, fpr, tpr, precision, recall, f1 = test(features, test_index, milnet, criterion,
                                                                              index_dict, args)
                log.info(f'lr:{lr},weight:{weight},fold:{fold},metric:{metric},avg_score: {avg_score},auc for TMB-H: {aucs[1]},f1:{f1},tn_fp_fn_tp:{conf_mat.ravel()[0], conf_mat.ravel()[1], conf_mat.ravel()[2], conf_mat.ravel()[3]}')
                metrics_visulization(fpr, tpr, aucs[1], precision, recall, conf_mat, pic_path)
                # log.info(f'avg_score: {avg_score}')
                # log.info(f'auc for TMB-H: {aucs[1]}')
                # log.info(f'tn, fp, fn, tp:{conf_mat.ravel()[0],conf_mat.ravel()[1],conf_mat.ravel()[2],conf_mat.ravel()[3]}')


if __name__ == '__main__':
    main()
