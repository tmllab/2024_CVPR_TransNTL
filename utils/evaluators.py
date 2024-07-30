from tqdm import tqdm
import torch.nn.functional as F 
import torch
from . import metrics


class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):

                if inputs.shape[1] == 1: 
                    inputs = torch.repeat_interleave(inputs, repeats=3, dim=1)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)


def eval_func(config, evaluators, model_ntl):
    model_ntl.eval()
    acc1s, acc5s = [], []
    for evaluator in evaluators:
        eval_results = evaluator(model_ntl, device=config.device)
        (acc1, acc5), _ = eval_results['Acc'], eval_results['Loss']
        acc1s.append(acc1)
        acc5s.append(acc5)
    return acc1s, acc5s


# choose the model with the maximum Acc sum of source and target domain
class attack_ntl_logger_bestsum():
    def __init__(self):
        self.st_sum = {'src': 0, 'tgt': 0, 'sum': 0}
        pass
        
    def log(self, src, tgt):
        d_sum = src + tgt
        if d_sum > self.st_sum['sum']:
            self.st_sum['src'] = src
            self.st_sum['tgt'] = tgt
            self.st_sum['sum'] = d_sum
            return True
        else:
            return False
        
    def result(self):
        return self.st_sum


# choose the model with the maximum target domain Acc
class attack_ntl_bestlogger_besttgt():
    def __init__(self):
        self.tgt_max = {'src': 0, 'tgt': 0, 'sum': 0}
        pass
        
    def log(self, src, tgt):
        # d_sum = src + tgt
        if tgt > self.tgt_max['tgt']:
            self.tgt_max['src'] = src
            self.tgt_max['tgt'] = tgt
            # self.tgt_max['sum'] = d_sum
            return True
        else:
            return False
        
    def result(self):
        return self.tgt_max