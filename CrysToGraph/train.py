# pretraining
import time
import os
import random
import copy

import numpy as np
import torch
from torch.nn import functional as F

from sklearn import metrics
from torch.autograd import Variable
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
        

class AtomRepresentationPretraining():
    def __init__(self, model, cuda=True):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()
        
        self.cuda = cuda and torch.cuda.is_available()
        self.model = model

        self.loss_list = []
        if self.cuda:
            self.model = model.cuda()
        
    def train(self, train_loader, criterion, optimizer, epochs, scheduler=None, verbose_freq=100):
        self.model.train()
        lrs = True
        if scheduler is None:
            lrs = False
        
        end = time.time()
        for epoch in range(epochs):
            loss_list = []
            for i, data in enumerate(train_loader):
                self.data_time.update(time.time() - end)

                data = (data[0].to(torch.device('cuda:0')), data[-1])

                target = data[-1].view(-1, 1)

                if self.cuda:
                    target = target.cuda()

                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss_list.append(loss.data.cpu().item())
                self.losses.update(loss.data.cpu().item(), target.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.batch_time.update(time.time() - end)
                end = time.time()
                
                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(train_loader))
            if lrs:
                scheduler.step()
            self.loss_list.append(sum(loss_list) / len(loss_list))
            
    def verbose(self, epoch, i, total):
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            epoch, i, total, batch_time=self.batch_time,
               data_time=self.data_time, loss=self.losses)
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
        self.loss_list = []
        
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
                outputs_i = F.normalize(outputs_i, dim=1)
                outputs_j = F.normalize(outputs_j, dim=1)
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
            self.loss_list.append(sum(loss_list) / len(loss_list))
        self.save_state_dict()
            
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

    def save_state_dict(self, path=''):
        try:
            os.mkdir('results')
        except FileExistsError:
            pass

        if path == '':
            path = 'results/%s.pt' % self.name
        torch.save(self.model.state_dict(), path)


class FineTuningWithDGL():
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
        self.loss_list = []

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

    def train(self, mixed_train_loader, optimizer, epochs, scheduler=None, verbose_freq: int=100, grad_accum: int=1):
        self.model.train()
        lrs = True
        if scheduler is None:
            lrs = False
        if self.random_seed is None:
            self.random_seed = [random.random() for i in range(epochs)]
        mpcd = mixed_train_loader.dataset
        batch_size = mixed_train_loader.batch_size

        self.criterion_task = MSELoss()
        
        end = time.time()
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
                output = self.model(data)
                target = data[-1]

                target = Variable(target.float())
                if self.cuda:
                    target = target.cuda()
                loss = self.criterion_task(output, target)

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
            self.loss_list.append(sum(loss_list) / len(loss_list))
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
