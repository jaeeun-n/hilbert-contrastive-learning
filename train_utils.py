import os.path
import re
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import math
from datetime import datetime
import gzip
import tqdm
import string

from sentence_transformers import models
from sentence_transformers import SentenceTransformer, InputExample

from loss_functions.simcse.simcse_loss import SimCSELoss
from loss_functions.simcse_hilbert.simcse_hilbert_loss import SimcseHilbertLoss
from loss_functions.simcse_hilbert_2branch.simcse_hilbert_loss_2branch import SimcseHilbertLoss2Branch


def get_train_samples(dataset_name, data_type, path_data):
    if data_type in ['par_doc']:
        path_par = path_data + '/' + dataset_name + '_par_par_idf_random'
        path_doc = path_data + '/' + dataset_name + '_doc_doc'
        train_samples = []
        with (open(path_par, encoding='utf8') as fpar, open(path_doc, encoding='utf8') as fdoc):
            for doc, par in zip(fdoc, fpar):
                doc = doc.strip()
                par = par.strip()
                train_samples.append(InputExample(texts=[doc, par]))
        return train_samples

    if data_type in ['2branch']:
        path_par = path_data + '/' + dataset_name + '_par_par_idf_random'
        path_doc = path_data + '/' + dataset_name + '_doc_doc'
        train_samples = []
        with (open(path_par, encoding='utf8') as fpar, open(path_doc, encoding='utf8') as fdoc):
            for doc, par in zip(fdoc, fpar):
                doc = doc.strip()
                par = par.strip()
                train_samples.append(InputExample(texts=[par, par, par, doc]))
        return train_samples

    path_file = path_data + '/' + dataset_name + '_' + data_type
    train_samples = []
    with open(path_file, encoding='utf8') as f:
        for example in tqdm.tqdm(f, desc='Read file'):
            example = example.strip()
            train_samples.append(InputExample(texts=[example, example]))
    return train_samples


def get_loss(loss_name, model: SentenceTransformer, device, loss_params):
    if loss_name == 'simcse':
        s_temp, lamb, emb = (loss_params['simcse_temperature'],
                             loss_params['lambda_value'],
                             loss_params['embedding_size'])
        hyperparameters = {'simcse_temperature': s_temp,
                           'lambda': lamb,
                           'embedding_size': emb}
        print('hyperparameter of simcse loss:', hyperparameters)
        return (SimCSELoss(backbone=model, temperature=s_temp, device=device, embedding_size=emb, lambda_value=lamb),
                hyperparameters)

    if loss_name == 'simcse_hilbert':
        s_temp, h_temp, lamb, emb = (loss_params['simcse_temperature'],
                                     loss_params['hilbert_temperature'],
                                     loss_params['lambda_value'],
                                     loss_params['embedding_size'])
        hyperparameters = {'simcse_temperature': s_temp,
                           'hilbert_temperature': h_temp,
                           'lambda': lamb,
                           'embedding_size': emb}
        print('hyperparameter of simcse_hilbert:', hyperparameters)
        return SimcseHilbertLoss(backbone=model, simcse_temperature=s_temp, hilbert_temperature=h_temp, device=device,
                                 lambda_value=lamb, embedding_size=emb), hyperparameters

    if loss_name == 'simcse_hilbert_2branch':
        s_temp, h_temp, lamb, emb = (loss_params['simcse_temperature'],
                                     loss_params['hilbert_temperature'],
                                     loss_params['lambda_value'],
                                     loss_params['embedding_size'])
        hyperparameters = {'simcse_temperature': s_temp,
                           'hilbert_temperature': h_temp,
                           'lambda': lamb,
                           'embedding_size': emb}
        print('hyperparameter of simcse_hilbert_2branch:', hyperparameters)
        return SimcseHilbertLoss2Branch(backbone=model, simcse_temperature=s_temp, hilbert_temperature=h_temp,
                                        device=device,
                                        lambda_value=lamb, embedding_size=emb), hyperparameters
    raise ValueError('Invalid loss function {}'.format(loss_name))
