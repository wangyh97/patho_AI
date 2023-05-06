nohup python train_tcga.py --lr 0.0001 --weight_decay 1e-05 --Tmax 50 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_0.out 
nohup python train_tcga.py --lr 0.0001 --weight_decay 1e-05 --Tmax 100 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_1.out 
nohup python train_tcga.py --lr 0.0001 --weight_decay 1e-05 --Tmax 200 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_2.out 
nohup python train_tcga.py --lr 0.0001 --weight_decay 1.9e-05 --Tmax 50 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_3.out 
nohup python train_tcga.py --lr 0.0001 --weight_decay 1.9e-05 --Tmax 100 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_4.out 
nohup python train_tcga.py --lr 0.0001 --weight_decay 1.9e-05 --Tmax 200 --gpu_index 0 > ../../out/5_classifier/hyperparam_select/train_tcga_5.out 