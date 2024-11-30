import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import utils.evaluators
import wandb
from termcolor import cprint
import copy
import numpy as np
from sam import SAM
from torchvision.transforms.functional import gaussian_blur


class robust_loss():
    def __init__(self, n_groups):
        self.n_groups = n_groups
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.normalize_loss = False
        self.adj = torch.zeros(self.n_groups).float().cuda()
        self.step_size = 0.01

    def compute_robust_loss(self, group_loss, group_count=None):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs * torch.exp(
            self.step_size*torch.tensor(adjusted_loss).data.cuda())
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = torch.zeros_like(group_loss[0])
        for ap, g_l in zip(self.adv_probs, group_loss):
            robust_loss += ap * g_l
        # robust_loss = group_loss @ self.adv_probs
        return robust_loss


def loss_function(config, labels, imgs, imgs_crpt_list, model_ntl, 
                  robust_loss_, criterion, return_all=False):
    
    def kld_criterion(crpt_out, clean_out_dt):
        return F.kl_div(F.log_softmax(crpt_out), 
                        F.log_softmax(clean_out_dt / config.dshift_group_Temp), 
                        reduction='mean', 
                        log_target=True)
    # forward
    clean_out = model_ntl(imgs)
    crpt_out_list = [model_ntl(imgs_crpt) for imgs_crpt in imgs_crpt_list]
    # compute CE loss
    loss_clean_CE = criterion(clean_out, labels)
    # KD loss
    clean_out_dt = clean_out.detach()
    loss_crpt_list = [kld_criterion(crpt_out, clean_out_dt) for crpt_out in crpt_out_list]
    loss_crpt = robust_loss_.compute_robust_loss(loss_crpt_list)

    loss = loss_clean_CE + config.loss_crpt_weight * loss_crpt

    if return_all: 
        return loss, loss_clean_CE, loss_crpt
    else: 
        return loss


def TransNTL(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
             model_ntl_bkp, model_ntl, datasets_name):
    for para in model_ntl.parameters():
        para.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model_ntl.parameters(), 
                    base_optimizer, 
                    rho=config.dshift_sam_rho, 
                    adaptive=True, 
                    lr=config.surrogate_lr, 
                    weight_decay=config.surrogate_weight_decay)
        
    if config.dshift_scheduler:
        if config.dshift_scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer.base_optimizer if 'sam' in config.dshift_optim else optimizer, 
            step_size=100)
        elif config.dshift_scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer if 'sam' in config.dshift_optim else optimizer, 
                T_max=config.surrogate_epochs)

    evaluators_val = [utils.evaluators.classification_evaluator(
        v) for v in dataloader_val]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v) for v in dataloader_test]
    # choose the model with the maximum Acc sum of source and target domain
    bestlogger = utils.evaluators.attack_ntl_logger_bestsum()

    robust_loss_ = robust_loss(3)

    for epoch in range(config.surrogate_epochs):
        model_ntl.train()
        for i, (imgs_src, labels_src, _, _) in enumerate(dataloader_train_srgt[0]):
            # clean image
            imgs = imgs_src.to(config.device)
            labels = torch.argmax(labels_src.to(config.device), dim=1)

            # corrupt image
            noise = torch.normal(mean=torch.zeros_like(imgs), std=config.dshift_gaussian_std)
            imgs_crpt_add = imgs + noise
            imgs_crpt_multi = (1 + noise) * imgs
            imgs_crpt_conv = gaussian_blur(imgs, kernel_size=config.dshift_blur_ks)
            imgs_crpt = [imgs_crpt_add, imgs_crpt_multi, imgs_crpt_conv]

            # dro loss
            loss, loss_clean_CE, loss_crpt = loss_function(
                config, labels, imgs, imgs_crpt, model_ntl, robust_loss_, criterion, 
                return_all=True)
            
            # sam
            loss.backward()
            optimizer.first_step(zero_grad=True)
            loss_function(config, labels, imgs, imgs_crpt, model_ntl, 
                          robust_loss_, criterion).backward()
            optimizer.second_step(zero_grad=True)
        
        if config.dshift_scheduler:
            scheduler.step()

        # validation
        acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, model_ntl)
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        # if best on validation, test 
        if epoch == 0 or bestlogger.log(acc1s[0], tgt_mean):
            test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model_ntl)
        
        wandb_log_dict = {
            'epoch_ft': epoch,
            'loss_ft': loss.item(),
            'loss_clean_CE': loss_clean_CE.item(),
            'loss_crpt': loss_crpt.item(),
            'Acc_ft_src': acc1s[0]}

        # print validation
        print('[Fine-tuning] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'Acc_ft_tgt_{dname}'] = acc_tgt
        print('')
        wandb.log(wandb_log_dict)
    
    wandb.run.summary['final_valbest_Acc_ft_src'] = bestlogger.result()['src']
    wandb.run.summary['final_valbest_Acc_ft_tgtmean'] = bestlogger.result()['tgt']
    wandb.run.summary['final_test_Acc_ft_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'final_test_Acc_ft_tgt_{dname}'] = acc_tgt
