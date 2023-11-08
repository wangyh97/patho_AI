import argparse
import os

import numpy as np
from tqdm import tqdm

import faiss

from clustering import Kmeans
def reduce(args, train_list):
    # 传进来的train_list是一个dict_of_features
    # add try
    new_feat_dict = {}
    for key in tqdm(train_list.keys()):
        uuid = train_list[key][0]
        label = train_list[key][1]
        feats = train_list[key][2]
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        kmeans = Kmeans(k=args.num_prototypes, pca_dim=args.pca_dim)
        kmeans.cluster(feats, seed=66)  # for reproducibility
        assignments = kmeans.labels.astype(np.int64)
        print(f'shape of assignments:{assignments.shape}\nassignment{assignments}')
        # compute the centroids for each cluster
        centroids = np.array([np.mean(feats[assignments == i], axis=0)
                              for i in range(args.num_prototypes)])

        # compute covariance matrix for each cluster
        covariance = np.array([np.cov(feats[assignments == i].T)
                               for i in range(args.num_prototypes)])
        # the semantic shift vectors are enough.
        semantic_shift_vectors = []
        for cov in covariance:
            semantic_shift_vectors.append(
                # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
                np.random.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                              size=args.num_shift_vectors))
        semantic_shift_vectors = np.array(semantic_shift_vectors)
        new_feat_tuple = (uuid,label,centroids,semantic_shift_vectors)
        new_feat_dict[key] = new_feat_tuple
        del feats
    return new_feat_dict

def main():
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--extractor', type=str, choices=['res','sim','saved','retccl'])
    parser.add_argument('--load',type=str,default='TCGA_high',choices=['TCGA_high','TCGA_low','c16_high'])
    parser.add_argument('--scale',type=int,default=10)
    parser.add_argument('--layer', type=int, default=18, help='layers of resnet,choose from[18,34,50,101],[18]')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--pca_dim', type=int, default=-1)
    parser.add_argument('--num_shift_vectors', type=int, default=200)

    parser.add_argument('--chucksize',type=int,help='chucksize for every single cpu to process')
    parser.add_argument('--cpu_index',type=int,help='start from 1')
    args = parser.parse_args()

    train_dir = {
        'res': f'../data/pretrained_resnet{args.layers}',
        'sim': '../data/simclr_extracted_feats',
        'saved': f'../data/{args.load}',
        'retccl': '../data/retccl_res50_2048'}
    train_list = np.load(f'{train_dir[args.extractor]}/{args.scale}X_full_slide_features.npy',allow_pickle=True).item()
    training_keys_slice = train_list.keys()[args.chucksize * (args.cpu_index-1):args.chucksize * args.cpu_index]
    train_list_slice = {train_list[key] for key in training_keys_slice}
    save_dir = os.path.join(train_dir[args.extractor],f'{args.scale}X_reduce_{args.num_prototypes}_features_{args.cpu_index}.npy')
    os.makedirs(save_dir,exist_ok=True)
    reduced_feats = reduce(args, train_list_slice)
    np.save(save_dir, reduced_feats)
if __name__ == '__main__':
    main()