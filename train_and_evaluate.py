import copy
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from utils.construct_hypergraph import construct_G_from_fts
from torch import nn

from config import get_config
from datasets import source_select
from models import model_select


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_model(model, fts, lbls, idx_train, idx_val,
                criterion, optimizer, scheduler, device,
                num_epochs=100, print_freq=500):
    """
    training method
    :param model: model to be trained
    :param fts: input features
    :param lbls: input labels
    :param idx_train: list of cross validation train set indicies
    :param idx_val: list of cross validation validation set indices
    :param criterion: loss function
    :param optimizer:
    :param scheduler:
    :param device: CUDA device
    :param num_epochs: epochs to train for
    :param print_freq:
    :return: best model on validation set
    """
    since = time.time()

    model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device)
    best_acc = 0.0
    loss_min = 100

    for epoch in range(num_epochs):

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(feats=fts)
                loss = criterion(outputs[idx], lbls[idx]) * len(idx)
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_wts_best_val_acc = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss
                model_wts_lowest_val_loss = copy.deepcopy(model.cpu().state_dict())
                model = model.to(device)

            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return model_wts_best_val_acc, model_wts_lowest_val_loss, best_acc, loss_min


def test_model(model, best_model_wts, fts, lbls, idx_test, device):
    """
    testing method
    :param model_best:
    :param fts:
    :param lbls:
    :param idx_test:
    :param edge_dict:
    :param device:
    :return:
    """
    model.load_state_dict(best_model_wts)
    model = model.to(device)
    model.eval()

    running_corrects = 0.0
    with torch.no_grad():
        outputs = model(feats=fts)

    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds[idx_test] == lbls.data[idx_test])
    test_acc = running_corrects.double() / len(idx_test)

    test_sens = torch.sum(
        preds[(min(idx_test) + lbls.data[idx_test].nonzero()).clone().detach().long()] == 1).item() / len(
        lbls.data[idx_test].nonzero().data)
    test_spec = torch.sum(
        preds[(min(idx_test) + (lbls.data[idx_test] != 1).nonzero()).clone().detach().long()] == 0).item() / len(
        (lbls.data[idx_test] != 1).nonzero().data)
    print('*' * 20)
    print('Test accuracy: %.2f' % test_acc)
    print('Test sensitivity: %.2f' % test_sens)
    print('Test specificity: %.2f' % test_spec)
    print('*' * 20)
    return test_acc


def get_source():
    cfg = get_config('config/config.yaml')
    source = source_select(cfg)
    if cfg['data_type'] == 'simulated':
        return source(cfg)


def train_test_HUNET():
    device = torch.device('cuda:0')
    cfg = get_config('config/config.yaml')
    fts, lbls, idx_trains, idx_vals, idx_test, n_category = get_source()
    H = construct_G_from_fts([fts], [cfg['k_construct_nn']])
    fts = torch.Tensor(fts).to(device)  # Convert to tensor and pass to device
    lbls = torch.Tensor(lbls).squeeze().long().to(
        device)  # Squeeze along axis that are 1 and convert the values to 64 bit integers and pass to device
    model = model_select(cfg['model']) \
        (dim_feat=fts.size(1),
         n_categories=n_category,
         n_stack=cfg['n_stack'],
         layer_spec=cfg['layer_spec'],
         pool_ratios=cfg['pool_ratios'],
         dropout_rate=cfg['drop_out'],
         H_for_hunet=H,
         hunet_depth=cfg['hunet_depth']
         )

    # initialize model
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()
    # Wieght decay : prevents wieghts from going too large
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.95, weight_decay=cfg['weight_decay'])
    # Adaptive learning rate with 1 milestone
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()

    trained_models = []
    for idx_set in range(0, len(idx_trains)):
        trained_models += [
            train_model(model, fts, lbls, idx_trains[idx_set], idx_vals[idx_set], criterion, optimizer,
                        schedular, device,
                        cfg['max_epoch'], cfg['print_freq'])]
    model_wts_best_val_acc = trained_models[0][0]
    model_wts_lowest_val_loss = trained_models[0][1]
    best_accuracy = trained_models[0][2]
    loss_min = trained_models[0][3]
    for trained_model in trained_models:
        if best_accuracy < trained_model[2]:
            model_wts_best_val_acc = trained_model[0]
        if loss_min > trained_model[3]:
            loss_min = trained_model[3]
            model_wts_lowest_val_loss = trained_model[1]
    print("stacked:{}\ndepth:{} \n pool_ratio: {}".format(cfg['n_stack'], cfg['hunet_depth'], cfg['pool_ratios']))
    print('**** Cross Validation accuracy results ****')
    print(np.asarray(trained_models)[:, 2])
    print('**** Cross Validation val loss results ****')
    print(np.asarray(trained_models)[:, 3])
    if idx_test is not None:
        print('**** Model of lowest val loss ****')
        test_acc_lvl = test_model(model, model_wts_lowest_val_loss, fts, lbls, idx_test,
                                  device)
        print('**** Model of best val acc ****')
        test_acc_bva = test_model(model, model_wts_best_val_acc, fts, lbls, idx_test, device)
        return max(test_acc_lvl, test_acc_bva)


if __name__ == '__main__':
    setup_seed(10000)
    train_test_HUNET()
