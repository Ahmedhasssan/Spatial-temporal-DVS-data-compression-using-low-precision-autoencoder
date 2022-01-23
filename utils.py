"""
Utilities for training
"""

import torch
import time
import tabulate

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(trainloader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for batch_idx, (inputs, target) in enumerate(trainloader):
        inputs = inputs.cuda()
        target = target.cuda(non_blocking=True)
        outputs = model(inputs)

        # one hot encoding
        labels_ = torch.zeros(target.size(0), 10).cuda()
        target = target.long()

        labels_ = labels_.scatter(1, target.view(-1, 1), 1)

        loss = criterion(outputs, labels_)
        
        prec1, prec5 = accuracy(outputs.data, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    
    res = {
        "acc": top1.avg,
        "loss": losses.avg,
    }
    return res

def test(testloader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs = inputs.cuda()
            target = target.cuda(non_blocking=True)
            outputs = model(inputs)

            # one hot encoding
            labels_ = torch.zeros(target.size(0), 10).cuda()
            target = target.long()
            labels_ = labels_.scatter(1, target.view(-1, 1), 1)

            loss = criterion(outputs, labels_)
            prec1, prec5 = accuracy(outputs.data, target, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    
    res = {
        "acc": top1.avg,
        "loss": losses.avg,
    }
    return res

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)
