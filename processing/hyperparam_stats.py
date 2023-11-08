import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import os
from pathlib import Path

'''
readers: convert result file into dataframe
'''

def reader(result_dir:str) -> pd.DataFrame:
    results = glob.glob(f'{result_dir}/re*.txt')
    run = Path(result_dir).stem[-1]
    result_list = [pd.read_csv(i,header=None) for i in results]
    result = pd.concat(result_list)
    result.columns = [
    "learning rate",
    "weight_decay",
    "Tmax",
    "best score",
    "best auc",
    "best avg auc",
    ]
    return result,run

#new columns in result file
def reader_kfold(result_dir:str) -> pd.DataFrame:
    results = glob.glob(f'{result_dir}/re*.txt')
    run = Path(result_dir).stem[-1]
    result_list = [pd.read_csv(i,header=None) for i in results]
    result = pd.concat(result_list)
    result.columns = [
    "learning rate",
    "weight_decay",
    "Tmax",
    "avg best score",
    "avg best auc",
    "best avg epoch",
    'avg_loss_score',  # best score where loss reaches its best
    'avg_loss_auc',
    'avg_loss'
    ]
    return result,run



'''
visualization
'''

def vis_best_metrics(results,metrics,top=5,run=None,figsize = (50, 50),output=False):
    """
    args:
        stats_df：tuple: (dataframe of result, run number)
        metrics: best auc,best score,best avg auc statistical data in different settings of hyperparameter
        top: show top N best point with stars
        run:if results are pd.DataFrame, assign run manually for correct saving of output figs. specially used in result 1-3
    """
    assert metrics in ['best auc','best score','best avg auc'], "metrics should be choosen from ['best auc','best score','best avg auc']"
    if type(results) == pd.DataFrame:
        stats_df = results
        run = run
    elif type(results) == tuple:
        stats_df,run = results
    else:
        raise TypeError('type of results should be pd.DataFrame or tuple')
    labels_lr = [str(i) for i in np.sort(stats_df["learning rate"].unique())]
    labels_wd = [str(i) for i in np.sort(stats_df["weight_decay"].unique())]
    labels_Tmax = [str(i) for i in np.sort(stats_df["Tmax"].unique())]
    auc_list = []
    score_list = []
    avg_auc_list = []
    # NOTE:auc/score/avg_auc_list is wrongly named, they have no relation with names but only a recorder of metrics for different hyperparameters seperately
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    for i in labels_lr:
        lr = eval(i)
        auc_list.append(stats_df[stats_df["learning rate"] == lr][metrics])
    axes[0].boxplot(auc_list, labels=labels_lr)
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel(metrics)
    
    for i in labels_wd:
        wd = eval(i)
        score_list.append(stats_df[stats_df["weight_decay"] == wd][metrics])
    axes[1].boxplot(score_list, labels=labels_wd)
    axes[1].set_xlabel("weight_decay")
    axes[1].set_ylabel(metrics)
    for i in labels_Tmax:
        Tmax = eval(i)
        avg_auc_list.append(stats_df[stats_df["Tmax"] == Tmax][metrics])
    axes[2].boxplot(avg_auc_list, labels=labels_Tmax)
    axes[2].set_xlabel("Tmax")
    axes[2].set_ylabel(metrics)
    if output:
        plt.savefig(f'./mil classifier/hyperparam_select_batch_{run}/{metrics}.png')
    print(stats_df[metrics].describe())
    
def top_comb(results,metrics,top=10):
    '''
    get hyperparameter combinations that reach top N auc/score/avg auc
    args:
        results:get from reader function
        metrics: choose from [best auc,best score,best avg auc]
        top: get top N combinations, 10 by default
    return:
        3 list containing best hyperparameter combinations ---- best_auc_comb,best_score_comb,best_avg_auc_comb
        each list is composed of a tuple:(comb:series,value of metric)
        here is an example:
            ((
            learning rate    0.000920
            weight_decay     0.000005
            Tmax             50.000000
            Name: 343, dtype: float64
            ),metric:1.2994971264367816)
    '''
    if type(results) == pd.DataFrame:
        stats_df = results
    elif type(results) == tuple:
        stats_df,_ = results
    else:
        raise TypeError('type of results should be pd.DataFrame or tuple')
    top_df = stats_df.sort_values([metrics],ascending = False).head(10)
    col_index_dict = {
        'best score':3,
        'best auc':4,
        'best avg auc':5
    }
    best_comb = [(top_df.iloc[i,:3],top_df.iloc[i,col_index_dict[metrics]]) for i in range(top)]
    return best_comb

def view_boosting(results,title,figsize=(15,30),output=False):
    '''
    resutls:get from reader function,a list of result, must be arranged in correct order
    title:choose from ['simclr','pretrained Res18','pretrained Res50']
    '''
    best_auc_list = [__extract_metrics(top_comb(result,'best auc')) for result in results]
    best_score_list = [__extract_metrics(top_comb(result,'best score')) for result in results]
    N_auc = len(best_auc_list)
    N_score = len(best_score_list)
    fig,axes = plt.subplots(1,2,figsize = figsize)
    if title:
        fig.suptitle(title)
    b0=axes[0].boxplot(best_auc_list,labels = ['trial 1','trial 2','trial 3'][:N_auc],patch_artist=True)
    for patch, color in zip(b0['boxes'], ['coral','orange','yellow'][:N_auc]):
        patch.set_facecolor(color)
    axes[0].set_title('best auc')
    
    b1=axes[1].boxplot(best_score_list,labels = ['trial 1','trial 2','trial 3'][:N_score],patch_artist=True)
    for patch, color in zip(b1['boxes'], ['coral','orange','yellow'][:N_score]):
        patch.set_facecolor(color)
    axes[1].set_title('best score')
    if output:
        save_dir = '../out/5_classifier/classifier_comparison'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'../out/5_classifier/classifier_comparison/{title}.png')
    print(f'pic saved at :/out/5_classifier/classifier_comparison/{title}.png')

def __extract_metrics(top_comb):
    metrics = [i[1] for i in top_comb]
    return metrics

def vis_scatter(description,result,factor_list,metrics,figsize = (10,20),output = True):
    num = len(factor_list)
    result_df = result[0]
    run = result[1]
    fig, axes = plt.subplots(num,1,figsize=figsize)
    for i in range(num):
        axes[i].scatter(result_df[f'{factor_list[i]}'],result_df[f'{metrics}'])
        axes[i].set_xlabel(factor_list[i])
        axes[i].set_ylabel(metrics)
    if output:
        plt.savefig(f'./mil classifier/{description}_{run}/{metrics}.png')

def vis_lr_on_metrics(result,metrics,figsize = (40,20),output = True):
    result_df = result[0]
    run = result[1]
    colors = []
    lr = result_df['learning rate']
    wd = result_df['weight_decay']
    for i in range(len(lr)):
        if lr[i] >= 0.0001 and lr[i] < 0.001:
            colors.append('red')
        elif lr[i] >= 0.00001 and lr[i] < 0.0001:  
            colors.append('blue')
        else:
            colors.append('green')
    plt.figure(figsize=figsize)
    plt.scatter(wd,result_df[f'{metrics}'],c = colors)
    plt.xlabel('Weight Decay')
    plt.ylabel(f'{metrics}')
    if output:
#         plt.savefig(f'./mil classifier/hyperparam_select_batch_{run}/{metrics}.png')
        pass

def vis_wd_on_metrics(result,metrics,figsize = (20,10),output = True):
    result_df = result[0]
    run = result[1]
    colors = []
    lr = result_df['learning rate']
    wd = result_df['weight_decay']
    for i in range(len(wd)):
        (1e-5,1e-4),(1e-6,1e-5),(1e-7,1e-6)
        if wd[i] >= 0.00001 and wd[i] < 0.0001:
            colors.append('red')
        elif wd[i] >= 0.000001 and wd[i] < 0.00001:  
            colors.append('blue')
        else:
            colors.append('green')
    plt.figure(figsize=figsize)
    plt.scatter(lr,result_df[f'{metrics}'],c = colors)
    plt.xlabel('learning rate')
    plt.ylabel(f'{metrics}')