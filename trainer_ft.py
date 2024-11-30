import trainer_normal
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import utils
import wandb


def train_src(config, dataloaders, valloaders, testloaders, model, print_freq=0, datasets_name=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.surrogate_lr,
                                momentum=config.surrogate_momentum,
                                weight_decay=config.surrogate_weight_decay)

    evaluators_val = [utils.evaluators.classification_evaluator(
        v) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v) for v in testloaders]
    # choose the model with the maximum Acc sum of source and target domain
    bestlogger = utils.evaluators.attack_ntl_logger_bestsum()    
    # bestlogger = utils.evaluators.attack_ntl_bestlogger_besttgt()    

    for epoch in range(config.surrogate_epochs):
        model.train()
        for i, (imgs_src, labels_src, _, _) in enumerate(dataloaders[0]):
            imgs = imgs_src.to(config.device)
            labels = labels_src.to(config.device)
            labels = torch.argmax(labels, dim=1)
            output = model(imgs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, model)
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        # if best, test 
        if epoch == 0 or bestlogger.log(acc1s[0], tgt_mean):
            test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

        wandb_log_dict = {
            'epoch_ft': epoch,
            'loss_ft': loss.item(),
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


def FTAL(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
         model_ntl_bkp, model_ntl, datasets_name):
    for para in model_ntl.parameters():
        para.requires_grad = True
    # fine-tune
    train_src(config, dataloader_train_srgt, dataloader_val, dataloader_test,
              model_ntl, datasets_name=datasets_name)
    

def RTAL(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
         model_ntl_bkp, model_ntl, datasets_name):
    # only re-initial classifier
    for m in model_ntl.classifier1:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    for para in model_ntl.parameters():
        para.requires_grad = True
    # fine-tune
    train_src(config, dataloader_train_srgt, dataloader_val, dataloader_test,
              model_ntl, datasets_name=datasets_name)
    
