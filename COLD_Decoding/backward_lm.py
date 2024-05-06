#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
import wandb
import argparse

import sys
sys.path.insert(0, './GPT2ForwardBackward')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from util import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bleuloss import batch_log_bleulosscnn_ae
from modeling_opengpt2 import OpenGPT2LMHeadModel
from padded_encoder import Encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"

path_to_backward = 'danyaljj/opengpt2_pytorch_backward'

encoder = Encoder()
model_backward = OpenGPT2LMHeadModel.from_pretrained(path_to_backward)

input = "until she finally won."
input_ids = encoder.encode(input)
input_ids = torch.tensor([input_ids[::-1] ], dtype=torch.int)
print(input_ids)

output = model_backward.generate(input_ids)

output_text = encoder.decode(output.tolist()[0][::-1])

print(output_text)