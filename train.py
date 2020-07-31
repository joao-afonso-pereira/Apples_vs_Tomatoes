from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import mean
import sys

from data import get_data_loaders
from cnn import CNN

EPOCHS = 50

loss_fn = F.cross_entropy

LEARNING_RATE = 1e-04
REG = 1e-04

def fit(model, data, device):
    
    # train and validation loaders
    train_loader, valid_loader = data
    
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=REG)

    # Start training
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Best validation params
    best_val = -float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        
        print('\nEPOCH {}/{}\n'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, y) in enumerate(train_loader):  # iterations loop
            # send mini-batch to gpu
            x = x.to(device)
            
            y = y.type(torch.LongTensor)
            y = y.to(device)
            
            # forward pass
            y_pred = model(x)
            
            ypred_ = torch.argmax(y_pred, dim=1)

            # Compute vae loss
            loss = loss_fn(y_pred, y)

            # Backprop and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            sys.stdout.write("\r" + '........{}-th mini-batch loss: {:.3f}'.format(i, loss.item()))
            sys.stdout.flush()
            
        # Validation
        tr_loss, tr_acc = eval_model(model, train_loader, device)
        train_history['train_loss'].append(tr_loss.item())
        train_history['train_acc'].append(tr_acc)
   
        val_loss, val_acc = eval_model(model, valid_loader, device)
        train_history['val_loss'].append(val_loss.item())
        train_history['val_acc'].append(val_acc)


        # save best validation model
        if best_val < val_acc:
            torch.save(model.state_dict(), 'outputs/cnn.pth')
            best_val = val_acc
            best_epoch = epoch

        # display the training loss
        print()
        print('\n>> Train loss: {:.3f}  |'.format(tr_loss.item()) + ' Train Acc: {:.3f}'.format(tr_acc))

        print('\n>> Valid loss: {:.3f}  |'.format(val_loss.item()) + ' Valid Acc: {:.3f}'.format(val_acc))

        print('\n>> Best model: {} / Acc={:.3f}'.format(best_epoch+1, best_val))
        print()

    # save train/valid history
    plot_fn = 'outputs/train_history.png'
    plot_train_history(train_history, plot_fn=plot_fn)

    # return best validation model
    model.load_state_dict(torch.load('outputs/cnn.pth'))

    return model


def plot_train_history(train_history, plot_fn=None):
    plt.switch_backend('agg')

    best_val_epoch = np.argmin(train_history['val_loss'])
    best_val_acc = train_history['val_acc'][best_val_epoch]
    best_val_loss = train_history['val_loss'][best_val_epoch]
    plt.figure(figsize=(7, 5))
    epochs = len(train_history['train_loss'])
    x = range(epochs)
    plt.subplot(211)
    plt.plot(x, train_history['train_loss'], 'r-')
    plt.plot(x, train_history['val_loss'], 'g-')
    plt.plot(best_val_epoch, best_val_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val loss')
    plt.legend(['train_loss', 'val_loss'])
    plt.axis([0, epochs, 0, max(train_history['train_loss'])])
    plt.subplot(212)
    plt.plot(x, train_history['train_acc'], 'r-')
    plt.plot(x, train_history['val_acc'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.axis([0, epochs, 0, 1])
    if plot_fn:
        plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def eval_model(model, data_loader, device, debug=False):
    with torch.no_grad():

        model.eval()
        
        loss_eval = 0
        N = 0
        n_correct = 0
        
        for i, (x, y) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            
            y = y.type(torch.LongTensor)
            y = y.to(device)

            # forward pass
            y_pred = model(x)           

            # Compute cnn loss
            loss = loss_fn(y_pred, y)
            loss_eval += loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_pred, dim=1)
            n_correct += torch.sum(1.*(ypred_ == y)).item()
            
            y = y.cpu().numpy()
            ypred_ = ypred_.cpu().numpy()
  

        loss_eval = loss_eval / N
        acc = n_correct / N
        
        
        return loss_eval, acc 


def main():

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU")
    
                
    model = CNN().to(DEVICE)
    
        
    (train_loader, valid_loader, test_loader) = get_data_loaders()
    
    
    # Fit model
    model, train_history, _, best_epoch = fit(model=model, data=(train_loader, valid_loader), device=DEVICE)
            
                
    # Test results
    test_loss, test_acc = eval_model(model, test_loader, DEVICE)
    
    print('\nTest loss: {:.3f}            |'.format(test_loss.item()) + ' Test Acc: {:.3f}'.format(test_acc))
    
    results_test = [test_loss.item(), test_acc]
       
    np.savetxt('results.txt', results_test, fmt='%.3f', delimiter=',')
                
    
    print("\n\nDONE!")

if __name__ == '__main__':
    main()
