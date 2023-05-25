from __future__ import print_function
import argparse
import os
import sys
import shutil
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
#from model.dataloaders.triplet_image_loader import TripletImageLoader, TripletImageLoader_celeba
from model.networks.Resnet_18 import resnet18
from model.models.baseline import MMetric

from tensorboardX import SummaryWriter
from model.utils import adjust_learning_rate, AverageMeter, accuracy, accuracy_id, set_gpu

def get_para():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Conditional Similarity Learning')
    parser.add_argument('--dataset', type=str, default='zapper', choices = {'zapper', 'celeba'}) 
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_mul', type=int, default=1,
                        help='the multiplier over the learning rate')            
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='30')
    parser.add_argument('--gamma', type=float, default=0.1)        
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 90)')    
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')    
    parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                        help='how many dimensions in embedding (default: 64)')    
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')    
    parser.add_argument('--ncondition', type=int, default=4)
    parser.add_argument('--tao', type=float, default=1.0, help='the temperature to control the hardness of the weight')
    parser.add_argument('--num_traintriplets', type=int, default=400000, metavar='N',
                        help='how many unique training triplets (default: 200000)')

    parser.add_argument('--metric_type', type=str, default='projection', choices = {'mask', 'projection'}, 
                        help='which type of metric to use')    
    parser.add_argument('--concept_type', type=str, default='DS', choices = {'FC', 'DS', 'TM', 'BiLSTM'}, 
                        help='which type of select concept')    
    parser.add_argument('--concept_normalize', action='store_true', default=False, 
                        help='which to normalize embedding before concept choice')       
    parser.add_argument('--residual', action='store_true', default=False, 
                        help='which to add residual after the projected embedding')  
    parser.add_argument('--weight_dist', action='store_true', default=True, 
                        help='weight embedding or distance')
    parser.add_argument('--weight_loss', action='store_true', default=False, 
                        help='weight embedding or distance')
    parser.add_argument('--with_label', action='store_true', default=False, 
                        help='with_label or not in DS method')  
    parser.add_argument('--reverse_test', action='store_true', default=False, 
                        help='if true, reverse p and n of triplets in testset')  
    parser.add_argument('--fix_BN', action='store_true', default=False, 
                        help='do not update running mean and variance in BN')    
    parser.add_argument('--freeze_BN', action='store_true', default=False, 
                        help='do not update any parameters in BN')              
    parser.add_argument('--resume', default=False, type=bool,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='whether pretrain resnet18')
    parser.add_argument('--pi', action='store_true', default=False, 
                        help='if true, use set module')  
    parser.add_argument('--vg', action='store_true', default=True, 
                        help='if true, valid by greedy acc, not TP acc')
    parser.add_argument('--HIK', action='store_true', default=False,
                        help='whether use Histagram Intersection Kernel to compute semantic regularization')
    parser.add_argument('--lamda', type=float, default=0.1, help='balance train loss and regularization')
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()

    set_gpu(args.gpu)
    save_path1 = '-'.join(['DiscoverNET', args.dataset, str(args.lr), str(args.batch_size), str(args.dim_embed), str(args.ncondition)])
    save_path2 = '-'.join(['fixed', str(args.metric_type), str(args.concept_type)])

    if args.pretrain:   
        save_path1 += '-pretrain'
    else:
        save_path1 += '-nopretrain'
    if args.vg:
        save_path1 += '-vg'  
    if args.pi:
        save_path1 += '-pi' 
    if args.HIK:
        save_path1 += '-HIK-lamda{}'.format(args.lamda)   


    if args.concept_normalize:
        save_path2 += '-CNorm'
    if args.residual:
        save_path2 += '-PRes'   
    if args.weight_dist:
        save_path2 += '-WD'
    if args.weight_loss:
        save_path2 += '-WL'
    if args.fix_BN:
        save_path2 += '-FixBN'
    if args.freeze_BN:
        save_path2 += '-FrzBN'
    if args.with_label:
        save_path2 += '-with_label'     
    args.save_path = os.path.join(save_path1, save_path2)
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)    
        
    return args

def get_loader(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'zapper':
        from model.dataloaders.triplet_image_loader import TripletImageLoader
        trainset = TripletImageLoader(args.ncondition, 'train', n_triplets=200000, aug=True)
        valset = TripletImageLoader(args.ncondition, 'val', n_triplets=20000, aug=False)
        testset = TripletImageLoader(args.ncondition, 'test', n_triplets=40000, aug=False)
        trainset_half = TripletImageLoader(args.ncondition, 'train_half', n_triplets=200000, aug=True)
        valset_half = TripletImageLoader(args.ncondition, 'val_half', n_triplets=20000, aug=False)
        testset_half = TripletImageLoader(args.ncondition, 'test_half', n_triplets=40000, aug=False)
    else:        
        from model.dataloaders.triplet_image_loader import TripletImageLoader_celeba as TripletImageLoader
        trainset = TripletImageLoader(args, 'train', n_triplets=args.num_traintriplets, aug=True)    
        valset = TripletImageLoader(args, 'val', n_triplets=80000, aug=False)        
        testset = TripletImageLoader(args, 'test', n_triplets=160000, aug=False)
        trainset_half = TripletImageLoader(args, 'train_half', n_triplets=args.num_traintriplets, aug=True)    
        valset_half = TripletImageLoader(args, 'val_half', n_triplets=80000, aug=False)        
        testset_half = TripletImageLoader(args, 'test_half', n_triplets=160000, aug=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False, **kwargs)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, **kwargs)
    train_half_loader = DataLoader(trainset_half, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_half_loader = DataLoader(valset_half, batch_size=256, shuffle=False, **kwargs)
    test_half_loader = DataLoader(testset_half, batch_size=256, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, train_half_loader, val_half_loader, test_half_loader

def build_model(args):
    emb_model = resnet18(pretrained=args.pretrain, embedding_size=args.dim_embed)
    if args.freeze_BN:
        for m in emb_model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False    
    
    model = MMetric(emb_model, args)
    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model.cuda()
    return model

def build_optimizer(model, args):
    optimizer = optim.Adam([{'params': model.embeddingnet.parameters()}], lr=args.lr)
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.epochs,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')    
    
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    return optimizer, lr_scheduler

def train(train_loader, model, criterion, optimizer, epoch, writer):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()
    if args.fix_BN or args.freeze_BN:
        model.embeddingnet.embeddingnet.eval()

    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()

        # compute output    
        dista, distb, weight = model(data1, data2, data3, c)
        with torch.no_grad():
            dista_r, distb_r, weight_r = model(data1, data3, data2, c)
        # 1 means dista should be larger than distb
        target = torch.ones_like(dista)
        if args.cuda:
            target = target.cuda()
        loss = criterion(dista, distb, target)  #(anchor, dissim, sim)  so dista > distb 

        if args.HIK:
            weight = weight[(dista>distb)&(dista_r>distb_r)]
            weight_r = weight_r[(dista>distb)&(dista_r>distb_r)]
            if weight.shape[0] == 0:
                reg = 0.0
            else:            
                hik = torch.min(weight,weight_r)
                reg = (torch.sum(hik, 1).mean())*weight.shape[0]
            loss = loss + args.lamda* reg
        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'lr={:.4g}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, optimizer.param_groups[0]['lr']))
            
            if writer is not None:
                writer.add_scalar('train_data/loss', float(losses.val), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train_data/loss', float(accs.val), epoch * len(train_loader) + batch_idx)        
            
def test(test_loader, model, criterion, epoch = None, writer = None):
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = {}
    for condition in list(range(args.ncondition)):
        accs_cs[condition] = AverageMeter()

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, c) in enumerate(test_loader):
            if args.cuda:
                data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
            c_test = c       
            dista, distb, weight = model(data1, data2, data3, c)
            target = torch.ones_like(dista)
            if args.cuda:
                target = target.cuda()
            if args.weight_loss:
                test_loss = criterion(dista, distb, target)
                test_loss = test_loss.unsqueeze(-1).view(-1, args.ncondition)
                test_loss = torch.mul(test_loss, weight)
                condition_loss = torch.sum(test_loss, 0)      #1*c
                best_condition = torch.argmin(condition_loss).item()  #scalar
                dista = dista[data1.size(0) * best_condition : data1.size(0) * (best_condition + 1)]
                distb = distb[data1.size(0) * best_condition : data1.size(0) * (best_condition + 1)]
                test_loss = test_loss[:,best_condition]
                test_loss = torch.sum(test_loss, 0).item()
            else:
                test_loss = criterion(dista, distb, target).item() 

            # measure accuracy and record loss
            acc = accuracy(dista, distb)
            accs.update(acc, data1.size(0))
            losses.update(test_loss, data1.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        losses.avg, 100. * accs.avg))
    if writer is not None:
        writer.add_scalar('val_data/loss', float(losses.avg), epoch)
        writer.add_scalar('val_data/acc', float(accs.avg), epoch)
    return accs.avg

def compute_cost(data_loader, model, conditions):
    all_sample = np.zeros((conditions, conditions))
    correct_sample = np.zeros((conditions, conditions))
    model.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, label, c) in enumerate(data_loader):
            label = torch.tensor(np.array(label, dtype=float))
            c = c.numpy().tolist()
            from collections import Counter
            res = Counter(c)
            for key,value in res.items():
                all_sample[key,:] += value
            if args.cuda:
                data1, data2, data3, label = data1.cuda(), data2.cuda(), data3.cuda(), label.cuda()
            for l in range(conditions):
                embedded_x = model.embeddingnet(data1)[0][:,l,:]
                embedded_y = model.embeddingnet(data2)[0][:,l,:]
                embedded_z = model.embeddingnet(data3)[0][:,l,:]
                dist_a = torch.norm(embedded_x - embedded_y, p=2, dim=1) 
                dist_b = torch.norm(embedded_x - embedded_z, p=2, dim=1)
                res = dist_a > dist_b
                index = torch.where(label.bool(), res.float(), (~res).float()).bool()
                index = index.cpu().numpy()
                c_select = np.array(c)[index].tolist()
                res = Counter(c_select)
                for key,value in res.items():
                    correct_sample[key][l] += value
    acc = correct_sample/all_sample
    cost = 1-acc
    print('\n cost_matrix:{}'.format(cost))
    return cost

def compute_T(conditions, cost):
    import ot
    a = (1/conditions)*np.ones((conditions,))
    b = (1/conditions)*np.ones((conditions,))
    T = ot.sinkhorn(a,b,cost, 1e-3)
    print('T:{}'.format(T))
    return T

def apply_T(C, T):
    acc = 1 - np.sum(C*T)
    id = T.argmax(1)
    for i in range(T.shape[0]):
        T[i,:] = T[i,:]/T.sum(1)[i]
    cost_id = (C*T).sum(1)
    acc_id = 1 - cost_id
    return acc_id, id, acc

def val_greedy(val_half_loader, model, n_embed, epoch = None, writer = None):
    cost = compute_cost(val_half_loader, model, n_embed)
    acc = 1-cost
    acc_id =acc.max(1)
    acc_mean = np.mean(acc_id)
    print('/n Val_greedy.Acc:{:.4f}'.format(acc_mean))
    if writer is not None:
        writer.add_scalar('val_data/acc', float(acc_mean), epoch)
    return acc_mean

def test_greedy_match(test_half_loader, model, n_embed):
    cost = compute_cost(test_half_loader, model, n_embed)
    acc = 1-cost
    acc_id =acc.max(1)
    id = acc.argmax(1)
    acc_mean = np.mean(acc_id)
    print('/n Test_greedy_id.Acc:{} Test_greedy.Acc:{:.4f}'.format(acc_id, acc_mean))
    return acc_id, id, acc_mean

def test_ot_match(val_half_loader, test_half_loader, model, n_embed):
    cost_valid = compute_cost(val_half_loader, model, n_embed)
    T = compute_T(n_embed, cost_valid)
    cost_test = compute_cost(test_half_loader, model, n_embed)
    acc_id, id, acc_mean = apply_T(cost_test, T)
    print('/n Test_ot_id.Acc:{} Test_ot.Acc:{:.4f}'.format(acc_id, acc_mean))
    return acc_id, id, acc_mean

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    cudnn.benchmark = True
    
if __name__ == '__main__':
    args = get_para()
    print(vars(args))
            
    args.cuda = torch.cuda.is_available()
    train_loader, val_loader, test_loader, train_half_loader, val_half_loader, test_half_loader = get_loader(args)
    model = build_model(args)
    if args.weight_loss:        
        criterion = torch.nn.MarginRankingLoss(reduce = False, margin = args.margin)
    else:
        criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer, lr_scheduler = build_optimizer(model, args)
    writer = SummaryWriter(logdir=args.save_path)

    if args.resume == True:
        # load checkpoint
        state = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        model.load_state_dict(resumed_state)
        best_acc = state['best_acc']
        best_epoch = state['best_epoch']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
    else:
        init_epoch = 1
        best_acc, best_epoch = 0,  1
        initial_lr = args.lr
    
    for epoch in range(init_epoch, args.epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch = epoch, writer = writer)
        lr_scheduler.step() 
        # evaluate on validation set
        if args.vg:
            acc = val_greedy(val_half_loader, model, args.ncondition, epoch, writer)
        else:
            acc = test(val_loader, model, criterion, epoch, writer)
        # remember best acc and save checkpoint
        is_best = False
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            is_best = True
        # save the best model
        state = {'epoch': epoch + 1,
                'args': args,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'optimizer' : optimizer.state_dict()}
        torch.save(state, os.path.join(args.save_path, 'checkpoint.pth.tar'))        
        print('i have saved................................................')
        if is_best:
            shutil.copyfile(os.path.join(args.save_path, 'checkpoint.pth.tar'), os.path.join(args.save_path, 'model_best.pth.tar'))
        print('Best Epoch: {} Best Val Accuracy: {:.2f}%'.format(best_epoch, 100. * best_acc))
    writer.close()
    # test the model
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_best.pth.tar'))['state_dict'])

    # # test the model with the old TP metric
    # test_acc = test(test_loader, model, criterion)

    # # test the model with our proposed new metrics
    acc_id_ot, id_ot, acc_mean_ot = test_ot_match(val_half_loader, test_half_loader, model, args.ncondition)
    acc_id_gd, id_gd, acc_mean_gd = test_greedy_match(test_half_loader, model, args.ncondition)










