from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from resnext import *
import argparse
from load_data import myData,imb_Dataloader,patient_Dataloader
import numpy as np
import gc
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class ArgsError(Exception): pass

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    loss_data = {'train':[],
                'val':[]}
    acc_data = {'train':[],
                'val':[]}

    for epoch in range(args.start_epoch+1,num_epochs):
        # Each epoch has a training and validation phase
        
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):             
                # wrap them in Variable1
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
#                 running_loss += loss.data[0]
                running_loss = loss.item()
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects / ((i+1)*args.batch_size)

                if phase == 'train' and i%args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                        args.print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            if phase == "val" and epoch_acc>best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            loss_data[phase].append(epoch_loss)
            acc_data[phase].append(epoch_acc)

        if (epoch+1) % args.save_epoch_freq == 0:
            if args.loader == 'balanced':
                save_dir = os.path.join(args.save_path,'class_weight',str(args.scale))
            else:
                save_dir = os.path.join(args.save_path,args.loader,str(args.scale))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model, os.path.join(save_dir, "epoch_" + str(epoch) + ".pth.tar"))
    if args.save_loss:
        np.save(os.path.join(save_dir,'lossdata.npy'),loss_data)
        np.save(os.path.join(save_dir,'accdata.npy'),acc_data)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    torch.save(model.state_dict(),os.path.join(save_dir,f'best_model_weights_at_epoch{best_epoch}.pth'))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    
    #notice: --loader unbalanced cannot be used with --class_weight
    parser = argparse.ArgumentParser(description="PyTorch implementation of resnext")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--save-loss',type=bool,default=True)
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--scale',type=int,help='scale of selected dataset, 5/10/20/40')
    parser.add_argument('--loader',type = str,default='unbalanced',help = 'method solving imbalanced classification, choose from: unbanlanced/shuffled/imb/over/under')
    parser.add_argument('--class-weight',action='store_true',help='using class-weight to balance the influence of H/L patches,as H/L is approximately 0.2, class-weight should be assigned as torch.tensor([0.84,0.16])')
    args = parser.parse_args()
    
    #check if the args are passed correctly
    if args.loader in ['unbalanced','shuffled']:
        args.class_weight = True
        print('mode: unbalanced dataset & class weighted loss function')
    elif args.loader in ['imb','over','under']:
        if args.class_weight:
            raise ArgsError('if loader is not unbalanced, class weight is no longer needed')
        else:
            print('mode: patient or patch wise balanced dataset and normal loss function')
    else:
        raise ArgsError('invalid loader type.loader should be choosen from unbalanced/imb/combine')
        
    # read data
    if args.loader in ['unbalanced','shuffled']:
        dataloders, dataset_sizes = myData(args)
    elif args.loader == 'imb':
        dataloders,dataset_sizes = imb_Dataloader(args)
    elif args.loader in ['over','under']:
        dataloders,dataset_sizes = patient_Dataloader(args)
    else:
        raise ArgsError('invalid loader type.loader should be choosen from unbalanced/imb/combine')

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = resnext50(num_classes = args.num_class)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    if args.class_weight:
        class_weight = torch.tensor([0.84,0.16])
        class_weight = class_weight.cuda()
        criterion = nn.CrossEntropyLoss(weight = class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # better using AdamW
#     optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    optimizer_ft = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0001)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    # better using CosineAnnelingLR
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max = args.num_epochs)

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes)
    
