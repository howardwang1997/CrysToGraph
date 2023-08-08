# pretraining
import time
import os
from multiprocessing.dummy import Pool as ThreadPool
import random
import copy

import numpy as np
import torch
from torch import nn
from torch_geometric import nn as tgnn
from torch.nn import functional as F

import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss, SmoothL1Loss



class AverageRecorder(object):
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def classification_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score
        
    
class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        

class AtomRepresentationPretraining():
    def __init__(self, model, cuda=True):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()
        
        self.accs = AverageRecorder()
        self.precisions = AverageRecorder()
        self.recalls = AverageRecorder()
        self.f1s = AverageRecorder()
        self.aucs = AverageRecorder()
        
        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = model.cuda()
        
    def train(self, train_loader, criterion, optimizer, epochs, scheduler=None, verbose_freq=100):
        self.model.train()
        lrs = True
        if scheduler is None:
            lrs = False
        
        end = time.time()
        self.loss_list = []
        for epoch in range(epochs):
            loss_list = []
            for i, data in enumerate(train_loader):
                self.data_time.update(time.time() - end)
                
                inputs = data
                target = data.y
                
                in_var = inputs
#                 in_var = Variable(inputs)
                target_var = Variable(target.view(-1).long())

                if self.cuda:
                    in_var = in_var.cuda()
                    target_var = target_var.cuda()

                outputs = self.model(inputs)
                loss = criterion(outputs, target_var)
                loss_list.append(loss.data.cpu().item())
#                 acc, prec, rec, f1, auc = classification_eval(outputs.data.cpu(), target)
                self.losses.update(loss.data.cpu().item(), target.size(0))
#                 self.accs.update(acc, target.size(0))
#                 self.precisions.update(prec, target.size(0))
#                 self.recalls.update(rec, target.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.batch_time.update(time.time() - end)
                end = time.time()
                
                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(train_loader), classification=True)
            if lrs:
                scheduler.step()
            self.loss_list.append(loss_list)
            
    def verbose(self, epoch, i, total, classification=True):
        if classification:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                epoch, i, total, batch_time=self.batch_time,
                data_time=self.data_time, loss=self.losses, accu=self.accs,
                prec=self.precisions, recall=self.recalls, f1=self.f1s,
                auc=self.aucs)
            )
            
    def get_atomic_representation(self):
        return self.model.atoms, self.model.embeddings.cpu()
        

class GraphConvPretraining():
    def __init__(self, model, cuda=True, shuffle=False, random_seed=None):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = model.cuda()
        
    def train(self, mixed_train_loader, criterion, optimizer, epochs, scheduler=None, verbose_freq=100):
        self.model.train()
        lrs = True
        if scheduler is None:
            lrs = False
        if self.random_seed is None:
            self.random_seed = [random.random() for i in range(epochs)]
#            print(self.random_seed)
        mpcd = mixed_train_loader.dataset
        batch_size = mixed_train_loader.batch_size
        
        end = time.time()
        self.loss_list = []
        for epoch in range(epochs):
            loss_list = []
            
            if self.shuffle:
                mpcd_copy = copy.deepcopy(mpcd)
                mpcd_copy.shuffle_by_batch(batch_size=batch_size, random_seed=self.random_seed[epoch])
                mixed_train_loader = DataLoader(mpcd_copy, batch_size=batch_size, shuffle=False)
#                print(mpcd_copy.idx_map)
                
            for i, (data_i, data_j) in enumerate(mixed_train_loader):
                self.data_time.update(time.time() - end)
                
                inputs_i = data_i
                inputs_j = data_j

                if self.cuda:
                    inputs_i = inputs_i.cuda()
                    inputs_j = inputs_j.cuda()

                outputs_i = self.model(inputs_i)
                outputs_j = self.model(inputs_j)
                output_i = F.normalize(outputs_i, dim=1)
                output_j = F.normalize(outputs_j, dim=1)
                loss = criterion(outputs_i, outputs_j)
                loss_list.append(loss.data.cpu().item())
                self.losses.update(loss.data.cpu().item(), criterion.batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.batch_time.update(time.time() - end)
                end = time.time()
                
                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(mixed_train_loader))
            if lrs:
                scheduler.step()
            self.loss_list.append(loss_list)
            
    def validate(self):
        pass
            
    def verbose(self, epoch, i, total):
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, total, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.losses)
        )
            
    def save_model(self, path=''):
        if path == '':
            path = 'config/model.tch'
        torch.save(self.model, path)
    
    def load_model(self, path=''):
        if path == '':
            path = 'config/model.tch'
        return torch.load(path)
    
    def save_state_dict(self, path=''):
        if path == '':
            path = 'config/model.tsd'
        torch.save(self.model.state_dict(), path)


class MixedTargetGraphConvPretrainingWithDGL():
    def __init__(self, model, cuda=True, shuffle=False, random_seed=None):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        if self.cuda:
            self.model = model.cuda()

    def _step(self, inputs):
        if self.cuda:
            inputs = inputs.cuda()
        outputs = self.model(inputs)
        return outputs

    def _calc_loss(self, outputs):
        return self.criterion(outputs[0], outputs[1])

    def _make_target(self, targets, n_class=2):
        new = torch.zeros((len(targets), n_class))
        for i in range(len(targets)):
            new[i, targets[i]] = 1
        return new

    def train(self, mixed_train_loader, criterion, optimizer, epochs, scheduler=None, verbose_freq: int=100, grad_accum: int=1):
        self.model.train()
        self.criterion = criterion
        lrs = True
        if scheduler is None:
            lrs = False
        if self.random_seed is None:
            self.random_seed = [random.random() for i in range(epochs)]
        mpcd = mixed_train_loader.dataset
        batch_size = mixed_train_loader.batch_size

        self.criterion_task = BCEWithLogitsLoss()
        self.criterion_task = CrossEntropyLoss()
        self.criterion_task = MSELoss()
        
        end = time.time()
        self.loss_list = []
        for epoch in range(epochs):
            loss_list = []
            self.losses.reset()
            
            if self.shuffle:
                mpcd_copy = copy.deepcopy(mpcd)
                mpcd_copy.shuffle_by_batch(batch_size=batch_size-1, random_seed=self.random_seed[epoch])
                mixed_train_loader = DataLoader(mpcd_copy, batch_size=batch_size, shuffle=False)

            for i, data in enumerate(mixed_train_loader):
                self.data_time.update(time.time() - end)
                end = time.time()

                if len(data) == 3:
                    data = tuple([data[0].to(torch.device('cuda:0')), data[1].to(torch.device('cuda:0')), torch.unsqueeze(data[2], -1)])
                elif len(data) == 2:
                    data = tuple([data[0].to(torch.device('cuda:0')), torch.unsqueeze(data[-1], -1)])
                opor = self.model(data)
                target = data[-1]

#                target = self._make_target(data[0].y)
                target = Variable(target.float())
                if self.cuda:
                    target = target.cuda()
                loss = self.criterion_task(opor[1], target)
#                print(target)
#                print(loss)

                loss_list.append(loss.data.cpu().item())
                self.losses.update(loss.data.cpu().item(), criterion.batch_size)

                loss /= grad_accum
                loss.backward()

                if i % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.batch_time.update(time.time() - end)
                end = time.time()

                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(mixed_train_loader))
            if lrs:
                scheduler.step()
            self.loss_list.append(loss_list)
            self.save_model()

    def verbose(self, epoch, i, total):
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, total, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.losses)
        )

    def save_model(self, path=''):
        if path == '':
            if not hasattr(self, 'save_model_index'):
                names = list(filter(lambda name: 'model' in name, os.listdir('config/')))
                names = [x.split('.')[0] for x in names]
                names = list(filter(lambda name: name[:5] == 'model', names))
                names = list(filter(lambda name: len(name) > 5, names))
                index = max([int(x[5:]) for x in names]) + 1
                self.save_model_index = index
            path = 'config/model%d.tsd' % self.save_model_index
        torch.save(self.model.state_dict(), path)


    def load_model(self, path=''):
        if path == '':
            path = 'config/model.tch'
        return torch.load(path)

    def save_state_dict(self, path=''):
        if path == '':
            path = 'config/model.tsd'
        torch.save(self.model.state_dict(), path)
