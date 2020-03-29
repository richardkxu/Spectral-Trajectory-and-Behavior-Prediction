import sys
import os
sys.path.append('..')
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from models import *
from sklearn.cluster import SpectralClustering , KMeans
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from torch.autograd import Variable


# change according to yr device
# device = torch.device("cuda:1")
device = torch.device("cuda:0")

TRAIN_SEQ_LEN = 6
PRED_SEQ_LEN = 10
FEAT_SIZE_IN = 1065
FEAT_SIZE_OUT = 2
BATCH_SIZE = 128
MU = 5
MODEL_LOC = '../resources/trained_models/mergeStream/{}'


def load_batch(index, batch_size, seq_ID, train_sequence_stream1, pred_sequence_stream_1, train_eig_seq, pred_eig_seq):
    '''
    load a batch of data
    :param index: index of the batch
    :param size: size of the batch of data
    :param seq_ID: either train sequence or a pred sequence, give as a str
    :param train_sequence: list of dicts of train sequences
    :param pred_sequence: list of dicts of pred sequences
    :return: Batch mof data
    '''

    i = index
    start_index = i * batch_size
    stop_index = (i+1) * batch_size

    if stop_index >= len(train_sequence_stream1):
        stop_index = len(train_sequence_stream1)
        start_index = stop_index - batch_size
    if seq_ID == 'train':
        stream1_train_batch = np.asarray([train_sequence_stream1[i]['sequence'] for i in range(start_index, stop_index)])
        eigs = train_eig_seq[start_index:stop_index]
        # input_tensor: 128, 6, 2; eigs: 128, 6, 1063
        # single_batch: 128, 6, 1065
        single_batch = np.concatenate((stream1_train_batch, eigs), axis=2)
    elif seq_ID == 'pred':
        stream1_pred_batch = np.asarray([pred_sequence_stream_1[i]['sequence'] for i in range(start_index, stop_index)])
        # eigs = pred_eig_seq[start_index:stop_index]
        # single_batch = np.concatenate((stream1_pred_batch, eigs), axis=2)
        single_batch = stream1_pred_batch
    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')
    return single_batch


def trainIters(n_epochs, train_dataloader, valid_dataloader, train_eig, val_eig, data, sufix, plot_every=1000, learning_rate=1e-3, save_every=5):
    print("Calling trainIters...")
    start = time.time()
    plot_losses_stream1 = []
    plot_losses_stream2 = []

    output_stream2_decoder = None
    num_batches = int(len(train_dataloader)/BATCH_SIZE)
    # Stream1 Data
    # inputs , labels = next ( iter ( train_dataloader ) )
    # [ batch_size , step_size , fea_size ] = inputs.size ()
    # input_dim = fea_size
    # hidden_dim = fea_size
    # output_dim = fea_size
    encoder_stream1 = None 
    decoder_stream1 = None
    encoder_stream2 = None
    decoder_stream2 = None
    encoder1loc = os.path.join(MODEL_LOC.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(MODEL_LOC.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix)) 

    train_eig_raw = train_eig
    pred_eig_raw = val_eig
    # Initialize encoder, decoders
    # stream1
    # TODO: double check the in, out, hidden size here !!!
    encoder_stream1 = Encoder ( FEAT_SIZE_IN , FEAT_SIZE_OUT , FEAT_SIZE_OUT ).to ( device )
    decoder_stream1 = Decoder ( 's1' , FEAT_SIZE_OUT , FEAT_SIZE_OUT , FEAT_SIZE_OUT, BATCH_SIZE, PRED_SEQ_LEN ).to ( device )
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)

    for epoch in range(0, n_epochs):
        print_loss_total_stream1 = 0  # Reset every print_every
        print_loss_total_stream2 = 0  # Reset every plot_every
        # Prepare train and test batch
        # train stream1
        for bch in range(num_batches):
            print('# {}/{} epoch {}/{} batch'.format(epoch, n_epochs, bch, num_batches))
            trainbatch = load_batch ( bch , BATCH_SIZE , 'train' , train_dataloader , valid_dataloader, train_eig_raw, pred_eig_raw )
            trainbatch_in_form = torch.Tensor(trainbatch).to(device)

            testbatch = load_batch ( bch , BATCH_SIZE , 'pred' , train_dataloader , valid_dataloader, train_eig_raw, pred_eig_raw )
            testbatch_in_form =  torch.Tensor(testbatch).to(device)

            input_stream1_tensor = trainbatch_in_form
            target_stream1_tensor = testbatch_in_form

            loss_stream1 = train_stream1(input_stream1_tensor, target_stream1_tensor, encoder_stream1, decoder_stream1, encoder_stream1_optimizer, decoder_stream1_optimizer)
            print_loss_total_stream1 += loss_stream1/num_batches

        # print( 'stream1 average loss:', print_loss_total_stream1/num_batches)
        print( 'stream1 average loss:', print_loss_total_stream1)

        if epoch % save_every == 0:
            save_model(encoder_stream1, decoder_stream1, data, sufix)

    compute_accuracy_stream1(train_dataloader, valid_dataloader, train_eig_raw, pred_eig_raw, encoder_stream1, decoder_stream1, n_epochs)

    # as of Mar 19, 2020, stream2 eval code is not available !

    # showPlot(plot_losses)
    save_model(encoder_stream1, decoder_stream1, data, sufix)

    return encoder_stream1, decoder_stream1


def eval(epochs, tr_seq_1, pred_seq_1, train_eig, val_eig, data, sufix, loc=MODEL_LOC):
    print("Calling eval...")
    encoder1loc = os.path.join(loc.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(loc.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix))


    # Initialize encoder, decoders
    encoder_stream1 = Encoder ( FEAT_SIZE_IN , FEAT_SIZE_OUT , FEAT_SIZE_OUT ).to ( device )
    decoder_stream1 = Decoder ( 's1' , FEAT_SIZE_OUT , FEAT_SIZE_OUT , FEAT_SIZE_OUT, BATCH_SIZE, PRED_SEQ_LEN ).to ( device )

    # the author save the whole model, which is bounded to cuda:1
    # remap to the current device on yr machine
    print("Loading trained model from:\n{}\n{}".format(encoder1loc, decoder1loc))
    encoder_stream1.load_state_dict(torch.load(encoder1loc, map_location=device))
    encoder_stream1.eval()
    decoder_stream1.load_state_dict(torch.load(decoder1loc, map_location=device))
    decoder_stream1.eval()

    compute_accuracy_stream1(tr_seq_1, pred_seq_1, train_eig, val_eig, encoder_stream1, decoder_stream1, epochs)
    # as of Mar 19, 2020, stream2 eval code is not available !


def train_stream1(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):
    # target_tensor: 128, 10, 2
    target_length = target_tensor.size(0)
    # input_tensor: 128, 6, 1065
    Hidden_State , _ = encoder.loop(input_tensor)
    # mu_1, mu_2: 128, 10, 1; log_sigma_1, log_sigma_2: 128, 10, 1
    _, _, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = decoder.loop(Hidden_State)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor)
    loss = loss if loss >0 else -1*loss
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    norm_loss = loss.item() / target_length
    return norm_loss


def save_model(encoder_stream1, decoder_stream1, data, sufix, loc=MODEL_LOC):
    torch.save(encoder_stream1.state_dict(), os.path.join(loc.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix)))
    torch.save(decoder_stream1.state_dict(), os.path.join(loc.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix)))
    print('model saved at {}'.format(loc.format(data)))


def generate(inputs, encoder, decoder):
    with torch.no_grad():
        Hidden_State , Cell_State = encoder.loop(inputs)
        decoder_hidden , decoder_cell , mu_1 , mu_2 , log_sigma_1 , log_sigma_2 , rho = decoder.loop(Hidden_State)
        [ batch_size , step_size , fea_size ] = mu_1.size ()
        out = []
        for i in range(batch_size):
            mu1_current = mu_1[ i , : , : ]
            mu2_current = mu_2[ i , : , : ]
            sigma1_current = log_sigma_1[ i , : , : ]
            sigma2_current = log_sigma_2[ i , : , : ]
            rho_current = rho[ i , : , : ]
            out.append(sample(mu1_current , mu2_current , sigma1_current , sigma2_current , rho_current))

        return np.array(out)


def log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y):
    batch_size, step_size, fea_size = y.size()

    epoch_loss = 0
    for i in range(step_size):
        mu1_current = mu_1[:,i,:]
        mu2_current = mu_2[:,i,:]
        sigma1_current = log_sigma_1[:,i,:]
        sigma2_current = log_sigma_2[:,i,:]
        rho_current = rho[:,i,:]
        y_current = y[:,i,:]
        batch_loss = compute_sample_loss(mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current).sum()
        # if np.isinf(batch_loss.detach().cpu()):
        #     print(batch_loss, mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current)
        #     print(batch_loss.shape, mu1_current.shape, mu2_current, sigma1_current, sigma2_current, rho_current, y_current)
        # if np.isnan(batch_loss.detach().cpu()):
        #     print(batch_loss, mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current)
        #     print(batch_loss, mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current)
        #     sys.exit(0)
        batch_loss = batch_loss/batch_size
        epoch_loss += batch_loss
    return epoch_loss


def compute_sample_loss(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y):
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi]).to(device)

    y_1 = y[:,0]
    y_1 = (y_1-torch.mean(y_1))/y_1.max()
    y_2 = y[:,1]
    y_2 = (y_2 - torch.mean(y_2))/y_2.max()
    #mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1))
    #mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1)) * (torch.std(y_1))/(torch.std(mu_1))
    #mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2))
    #mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2)) * (torch.std(y_2))/(torch.std(mu_2))
    z = ( (y_1 - mu_1)**2/(log_sigma_1**2) + ((y_2 - mu_2)**2/(log_sigma_2**2)) - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 *log_sigma_2)) )
    mog_lik2 = ( (-1*z)/(2*(1-rho**2)) ).exp()
    mog_lik1 =  1/(pi_term * log_sigma_1 * log_sigma_2 * (1-rho**2).sqrt() )
    mog_lik = (mog_lik1*mog_lik2).log()
    return mog_lik


# ====================================== HELPER FUNCTIONS =========================================

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def sample(mu_1 , mu_2 , log_sigma_1 , log_sigma_2, rho):

    sample = []
    for i in range(len(mu_1)):
        mu = np.array([mu_1[i][0].item(), mu_2[i][0].item()])
        sigma_1 = log_sigma_1[i][0].item()
        sigma_2 = log_sigma_2[i][0].item()
        c = rho[i][0].item() * sigma_1 * sigma_2
        cov = np.array([[sigma_1**2, c],[c, sigma_2**2]])
        sample.append(np.random.multivariate_normal(mu, cov))

    return sample

def computeDist ( x1 , y1, x2, y2 ):
    return np.sqrt( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

def computeKNN ( curr_dict , ID , k ):
    import heapq
    from operator import itemgetter

    ID_x = curr_dict[ ID ][0]
    ID_y = curr_dict[ ID ][1]
    dists = {}
    for j in range ( len ( curr_dict ) ):
        if j != ID:
            dists[ j ] = computeDist ( ID_x , ID_y, curr_dict[ j ][0],curr_dict[ j ][1] )
    KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
    neighbors = list ( KNN_IDs.keys () )

    return neighbors

def compute_A ( frame ):
    A = np.zeros ( [ frame.shape[ 0 ] , frame.shape[ 0 ] ] )
    for i in range ( len ( frame ) ):
        if frame[ i ] is not None:
            neighbors = computeKNN ( frame , i , 4 )
        for neighbor in neighbors:
            A[ i ][ neighbor ] = 1
    return A


def compute_accuracy_stream1(traindataloader, labeldataloader, train_eig, val_eig, encoder, decoder, n_epochs):
    ade = 0
    fde = 0
    count = 0

    train_raw = traindataloader
    pred_raw = labeldataloader
    train_eig_raw = train_eig
    pred_eig_raw = val_eig

    # batch = load_batch(0, BATCH_SIZE, 'pred', train_raw, pred_raw, train_eig_raw, pred_eig_raw)
    # batch, _, _ = batch
    # batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BATCH_SIZE)])
    # batch_in_form = torch.Tensor(batch_in_form)
    # [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)

    print('computing accuracy...')
    for epoch in range(0, n_epochs):
        # Prepare train and test batch
        if epoch % (int(n_epochs/10) + 1) == 0:
            print("{}/{} in computing accuracy...".format(epoch, n_epochs))
        trainbatch = load_batch ( epoch , BATCH_SIZE , 'train' , train_raw , pred_raw, train_eig_raw, pred_eig_raw)
        trainbatch_in_form = torch.Tensor ( trainbatch )

        testbatch = load_batch ( epoch , BATCH_SIZE , 'pred' , train_raw , pred_raw, train_eig_raw, pred_eig_raw)
        testbatch_in_form = torch.Tensor ( testbatch )

        train = trainbatch_in_form.to(device)
        label = testbatch_in_form.to(device)

        pred = generate(train, encoder, decoder)
        mse = MSE(pred, label)
        # print(mse)
        mse = np.sqrt(mse)
        ade += mse
        fde += mse[-1]
        # count += testbatch_in_form.size()[0]
        count +=1

    ade = ade/count
    fde = fde/count
    print('ADE: {}'.format(ade))
    print("Mean ADE: {} FDE: {}".format(np.mean(ade), fde))


def makeplot(x, y, x_label, y_label, title, save_loc):
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    fig.savefig(os.path.join(save_loc, title+'.png'))
    fig.clear()

def MSE(y_pred, y_gt, device=device):
    # y_pred = y_pred.numpy()
    y_gt = y_gt.cpu().detach().numpy()
    acc = np.zeros(np.shape(y_pred)[:-1])
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = np.array(y_gt[:,:, 0])
    x = (x-np.mean(x))/x.max()
    y = np.array(y_gt[:,:, 1])
    #muX = np.mean(x) + (x - np.mean(muX)) * (np.std(x))/(np.std(muX))
    y = (y-np.mean(y))/y.max()
    #muY = np.mean(y) + (x -np.mean(muY)) * (np.std(y))/(np.std(muY))
    acc = np.power(x-muX, 2) + np.power(y-muY, 2)
    lossVal = np.sum(acc, axis=0)/len(acc)
    return lossVal

def compute_eigs ( train_stream2):
    N = train_stream2[0][list ( train_stream2[ 0 ].keys())[ -1]].shape[ 1 ]
    A = np.zeros ( [ N, N ] )
    frame = {}
    eig_batch = []
    for batch_idx in range(len(train_stream2)):
        eig_frame = []
        for which_frame in list ( train_stream2[ batch_idx ].keys())[ 2: ]:
            for j in range(N):
                frame[j] = [train_stream2[batch_idx][which_frame][0,j],train_stream2[batch_idx][which_frame][1,j]]
            for l in range (N):
                if frame[ l ] is not None:
                    neighbors = computeKNN ( frame , l , 4 )
                for neighbor in neighbors:
                    # if neighbor in labels:
                    # if idx < labels.index ( neighbor ):
                    dist_of_neighbor = computeDist(frame[l][0],frame[l][1], frame[neighbor][0],frame[neighbor][1])
                    if dist_of_neighbor <= MU:
                        A[ l ][ neighbor ] = np.exp(-1*computeDist(frame[l][0],frame[l][1], frame[neighbor][0],frame[neighbor][1]))
            d = [ np.sum ( A[ row , : ] ) for row in range ( A.shape[ 0 ] ) ]
            D = np.diag ( d )
            L = D - A
            _, vecs = eigs(L, k=2 )
            eig_frame.append(np.real(vecs[:,1]))
        eig_batch.append ( np.array(eig_frame) )
    return torch.Tensor(np.array(eig_batch))
