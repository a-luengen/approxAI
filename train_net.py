import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import os
import sys
#from datetime import datetime
import time
import numpy as np 
from net.ResNet import resnet50
from net.WarmUpLR import WarmUpLR
from utils import getDataLoaders

import time

# Hyper-parameters

lr = 0.01
momentum = 0.9
weight_decay = 1e-5 # or 5e-4
epochs = 10
batch_size_train = 64# 4 or 256
batch_size_test = 64 # 4 or 256

# settings
warm_up = 1 # default = 1
save_epoch = 3 # save model after this amount of epochs
milestones = [60, 120, 160]
use_cuda = False
use_colab = False


def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(trainLoader):
        if epoch <= warm_up:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)
        if use_cuda:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        #n_iter = (epoch - 1) * len(trainLoader) + batch_index + 1
        #last_layer = list(net.children())[-1]
        #for name, para in last_layer.named_parameters():
        #    if 'weight' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #    if 'bias' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size_train + len(images),
            total_samples=len(trainLoader.dataset)
        ))

        #update training loss for each iteration
        #writer.add_scalar('Train/loss', loss.item(), n_iter)

    #for name, param in net.named_parameters():
    #    layer, attr = os.path.splitext(name)
    #    attr = attr[1:]
    #    writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch, testLoader):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in testLoader:
        images = Variable(images)
        labels = Variable(labels)

        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(testLoader.dataset),
        correct.float() / len(testLoader.dataset)
    ))
    print()

    #add informations to tensorboard
    #writer.add_scalar('Test/Average loss', test_loss / len(testLoader.dataset), epoch)
    #writer.add_scalar('Test/Accuracy', correct.float() / len(testLoader.dataset), epoch)

    return correct.float() / len(testLoader.dataset)

if __name__ == '__main__':
    print("Configuring...")

    net_name = "resnet_50"
    date_time_now = time.strftime("%H-%M-%S-%d-%m-%Y")

    if use_colab:
        checkpoint_path = os.path.join('drive/My Drive/ApproxAI/checkpoints', net_name, date_time_now)
    else:
        checkpoint_path = os.path.join('./checkpoints', net_name, date_time_now)

    print(checkpoint_path)
    net = resnet50()

    if use_cuda:
        net = net.cuda()

    testLoader, trainLoader = getDataLoaders(batch_size_train, batch_size_test)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # learning rate decay
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    iter_per_epoch = len(trainLoader)

    # use warmup scheduler to avoid numerical instability in the beginning of the training
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_up)

    # create checkpoint folder to save models
    if not os.path.exists(os.path.join(checkpoint_path, net_name)):
        os.makedirs(os.path.join(checkpoint_path, net_name))    
    checkpoint_path = os.path.join(checkpoint_path, net_name, '{net}-{epoch}-{type}.pth')

    print('Started training...')
    best_acc = 0.0
    start = time.time()
    for epoch in range(1, epochs):
        if epoch > warm_up:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch, testLoader)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > milestones[1] and best_acc < acc:
            print("Milestone save!")
            torch.save(net.state_dict(), checkpoint_path.format(net=net_name, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % save_epoch:
            print("Checkpoint save!")
            torch.save(net.state_dict(), checkpoint_path.format(net=net_name, epoch=epoch, type='regular'))
    print('Train time for one epoch: ', (time.time() - start) / epochs )
    print("Done with training!")