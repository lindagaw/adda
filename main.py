"""Main script for ADDA."""

import params, pretty_errors
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from models import get_classifier
from utils import get_data_loader, init_model, init_random_seed

from datasets import obtain_office_31, obtain_office_home

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader, src_data_loader_eval = obtain_office_31('W')
    tgt_data_loader, tgt_data_loader_eval = obtain_office_31('D')

    model = get_classifier('inception_v3', pretrain=True)

    src_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).cuda()
    tgt_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).cuda()

    del model

    src_classifier = nn.Linear(2048, 31).cuda()
    critic = Discriminator(input_dims=params.d_input_dims,
                        hidden_dims=params.d_hidden_dims,
                        output_dims=params.d_output_dims)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)


    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source encoder on source <<<")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)
    print(">>> source encoder on target <<<")
    eval_src(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_src(tgt_encoder, src_classifier, tgt_data_loader_eval)
