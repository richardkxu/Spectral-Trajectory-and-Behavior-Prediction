import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from train_eval2 import *

import pickle


DATA = 'APOL'
train_seq_len = 6
pred_seq_len = 10

# DATA = 'LYFT'
# train_seq_len = 20
# pred_seq_len = 30

# DATA = 'ARGO'
# train_seq_len = 20
# pred_seq_len = 30

# change according to yr device
device = torch.device("cuda:0")
SUFIX = '1stS1new'

TRAIN = True
EVAL = True

DIR = '../resources/data/{}/'.format(DATA)
MODEL_DIR = '../resources/trained_models/'

train_epochs = 15
test_epochs = 10
save_per_epochs = 5


if __name__ == "__main__":
    print("Dataset: {}".format(DATA))
    print("Input seq length: {} | Pred seq length: {}".format(train_seq_len, pred_seq_len))
    print("Using Merge Stream Model")

    if TRAIN:
        # load stream1 preprocessed data
        f1 = open ( DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
        g1 = open ( DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
        f3 = open(DIR + 'stream2_obs_eigs_train.pkl', 'rb')  # 'r' for reading; can be omitted
        g3 = open(DIR + 'stream2_pred_eigs_train.pkl', 'rb')  # 'r' for reading; can be omitted

        tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
        pred_seq_1 = pickle.load ( g1 )  # load file content as mydict
        tr_eig_seq = pickle.load ( f3 )
        pred_eig_seq = pickle.load ( g3 )
        f1.close()
        g1.close()
        f3.close()
        g3.close()

        encoder1, decoder1 = trainIters(train_epochs, tr_seq_1 , pred_seq_1, tr_eig_seq, pred_eig_seq, DATA, SUFIX, save_every=save_per_epochs)

    if EVAL:
        print('start evaluating {}{}...'.format(DATA, SUFIX))
        f1 = open ( DIR + 'stream1_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        g1 = open ( DIR + 'stream1_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        f3 = open(DIR + 'stream2_obs_eigs_test.pkl', 'rb')  # 'r' for reading; can be omitted
        g3 = open(DIR + 'stream2_pred_eigs_test.pkl', 'rb')  # 'r' for reading; can be omitted

        train_seq_1 = pickle.load ( f1 )  # load file content as mydict
        pred_seq_1 = pickle.load ( g1 )  # load file content as mydict
        train_eig_seq = pickle.load(f3)
        pred_eig_seq = pickle.load(g3)
        f1.close ()
        g1.close ()
        f3.close()
        g3.close()

        # TODO: for test, eigen vector is of len 366, which is diff from train eigen of len 1063???
        # this eval code only takes traj input, not traffic graph input
        eval(test_epochs, train_seq_1, pred_seq_1, train_eig_seq, pred_eig_seq, DATA, SUFIX)