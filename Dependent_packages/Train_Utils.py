from Dependent_packages import Dataload
import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import Model
import torch.utils.data
import csv
from sklearn.metrics import confusion_matrix


def save_train_data(epoch, train_loss, train_acc):
    train_data = [(epoch, train_loss, train_acc)]
    with open('./Model data/train_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_data)


def save_val_data(epoch, val_loss, val_acc):
    val_data = [(epoch, val_loss, val_acc)]
    with open('./Model data/val_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(val_data)


def save_confusion_matrix(phase, epoch, conf_matrix):
    file_path = os.path.join('./Model data/Confusion Matrix',
                             f'{phase}_confusion_matrix_epoch_{epoch}.csv')
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(conf_matrix)


class train_utils(object):

    def __init__(self, args, save_dir):

        self.args = args
        self.save_dir = save_dir

    def setup(self):

        args = self.args
        
        '''
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            print('gpu is available')
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        '''
        
        warnings.warn("gpu is not available")
        self.device = torch.device("cpu")
        self.device_count = 1
        logging.info('using {} cpu'.format(self.device_count))

        Dataset = getattr(Dataload, args.data_name)

        self.datasets = {}
        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir, args.normalize_type).data_prepare()
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}

        self.model = getattr(Model, args.model_name)(in_channel=Dataset.input_channel, out_channel=Dataset.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):

        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            for phase in ['train', 'val']:

                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Initialize the variables of the confusion matrix
                all_labels = []
                all_predictions = []

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):

                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Adding real and predicted labels to the confusion matrix
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(pred.cpu().numpy())

                        if phase == 'train':

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # calculate confusion matrix
                conf_matrix = confusion_matrix(all_labels, all_predictions)
                print(f'{phase} confusion matrix：')
                print(conf_matrix)
                save_confusion_matrix(phase, epoch, conf_matrix)

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                if phase == 'train':
                    train_loss = epoch_loss
                    train_acc = epoch_acc
                    save_train_data(epoch, train_loss, train_acc)
                elif phase == 'val':
                    val_loss = epoch_loss
                    val_acc = epoch_acc
                    save_val_data(epoch, val_loss, val_acc)

                if phase == 'val':
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    if epoch_acc > best_acc or epoch > args.max_epoch - 2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()













