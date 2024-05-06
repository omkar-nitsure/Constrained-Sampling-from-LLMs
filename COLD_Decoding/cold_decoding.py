#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
import wandb
import argparse
import torch

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
from transformers import GPT2LMHeadModel, AutoTokenizer
from bleuloss import batch_log_bleulosscnn_ae
from modeling_opengpt2 import OpenGPT2LMHeadModel

stop_words = set(stopwords.words('english'))


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--beam-search", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--lexical-variants", action="store_true", help="")
    parser.add_argument("--if-zx", action="store_true")
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--output-dir", type=str, default="./data/commongen/")
    parser.add_argument("--fwd-model", type=str,
                        default="/var/karen/workspace/GPT2ForwardBackward/opengpt2_pytorch_forward")
    parser.add_argument("--back-model", type=str,
                        default="danyaljj/opengpt2_pytorch_backward")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch", type=int, default=1, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='constrained_langevin',
                        choices=['lexical_generation', 'counterfactual_langevin', 'abductive_langevin',
                                  'grammar', 'sentiment'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=50, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument("--abductive-c2-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-c2-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-c1-weight", type=float, default=1)
    parser.add_argument("--sentiment-c1-weight", type=float, default=1)
    parser.add_argument("--sentiment-c2-weight", type=float, default=0.05)
    parser.add_argument("--sentiment-c1-rev-weight", type=float, default=0.5)
    parser.add_argument("--abductive-filterx", action="store_true", help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=6)
    parser.add_argument("--sentiment-max-ngram", type=int, default=4)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1,
                        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='random', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters", type=int, default=1000,
                        help="add one more group of constraints from N iters")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")

    args = parser.parse_args()
    return args

### EDIT: Adding x_counter to the decode function in order to improve the counterfactual reasoning task
def decode(model, tokenizer, device, x="", z="", constraints=None, args=None, model_back=None, zz=None, x_counter="", model_clf=None, tokenizer_clf=None):
    '''
    x: left context   (prompt in lexical lexical task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''

    x_ = tokenizer.encode(x) # x_ is a list of token ids
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)

    if 'counterfactual' in args.mode and x_counter != "":
        x_counter_ = tokenizer.encode(x_counter)
        x_counter_t = torch.tensor(x_counter_, device=device, dtype=torch.long)
        x_counter_onehot = one_hot(x_counter_t, dimension=tokenizer.vocab_size)
        x_counter_t = x_counter_t.unsqueeze(0).repeat(args.batch_size, 1)
        x_counter_onehot = x_counter_onehot.repeat(args.batch_size, 1, 1)

    z_mask = None

    if 'counterfactual' in args.mode:
        z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
        z_t = torch.tensor(z_, device=device, dtype=torch.long)

        z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)

        if x_counter != "":
            z_counter = "<|endoftext|> "
            z_counter_ = tokenizer.encode(z_counter)
            z_counter_t = torch.tensor(z_counter_, device=device, dtype=torch.long)
            z_counter_onehot = one_hot(z_counter_t, dimension=tokenizer.vocab_size)
            z_counter_onehot = z_counter_onehot.repeat(args.batch_size, 1, 1)
            z_counter_t = z_counter_t.unsqueeze(0).repeat(args.batch_size, 1)

        length = args.length
        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nlength:\t%d\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), length, constraints))

        # z_mask: [batch_size, vocab_size]
        z_words = word_tokenize(z[2:])  # delete the ". " token we appended before
        z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
        z_nonstop_words += [z_words[0]]  # add the first token
        z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
        z_nonstop_ = tokenizer.encode(z_nonstop_words)
        print('|' + z_nonstop_words + '|')

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[z_nonstop_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    if 'abductive' in args.mode or 'lexical' in args.mode:
        length = args.length

        z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
        z_t = torch.tensor(z_, device=device, dtype=torch.long)
        z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
        # repeat batch_size times
        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

        zz_ = tokenizer.encode(zz)[1:]  # delete the "." token we appended before
        zz_t = torch.tensor(zz_, device=device, dtype=torch.long)
        zz_t = zz_t.unsqueeze(0).repeat(args.batch_size, 1)

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[zz_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nzz:\t|%s|\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), tokenizer.decode(zz_), constraints))

    if 'sentiment' in args.mode:
        length = args.length 
        z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
        z_t = torch.tensor(z_, device=device, dtype=torch.long)
        z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)
        length = args.length

        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nlength:\t%d\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), length, constraints))
            
        z_words = word_tokenize(z[2:])
        z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
        z_nonstop_words += [z_words[0]]
        z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
        z_nonstop_ = tokenizer.encode(z_nonstop_words)
        print('|' + z_nonstop_words + '|')

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[z_nonstop_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    model.eval()

    if args.init_mode == 'random':
        init_logits = initialize(model, x_t, length, args.init_temp, device)
    else:
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for bi in range(args.batch_size):
        print("[initial]: %s" % (text[bi]))

    if args.wandb:
        wandb.init(
            project='Hyperparameter Tuning',
            config=args)

    assert args.prefix_length <= 0  # Otherwise not compatible with batch mode

    # if args.prefix_length > 0:
    #     prefix_logits = torch.nn.Parameter(
    #         torch.rand(x_onehot.shape[0], args.prefix_length, x_onehot.shape[2], dtype=init_logits.dtype,
    #                    device=device))

    y_logits = init_logits
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    # if args.prefix_length > 0:
    #     optim = torch.optim.Adam([epsilon, prefix_logits], lr=args.stepsize)
    # else:
    optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=args.stepsize_iters,
                                                gamma=args.stepsize_ratio)
    # SCHEDULER -- Read Welling et al. -----------[OMKAR]
    frozen_len = args.frozen_length

    y_logits_ = None
    noise_std = 0.0

    ## Encode x beforehand
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"
    soft_forward_x = x_onehot[:, -1:, :]  # The last token of x is used in soft_forward
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [_.detach() for _ in x_model_past]

    if 'counterfactual' in args.mode and x_counter != "":
        soft_forward_x_counter = x_counter_onehot[:, -1:, :]
        if x_counter_t.shape[1] == 1:
            x_model_past_counter = None
        else:
            x_model_outputs_counter = model(x_counter_t[:, :-1])
            x_model_past_counter = x_model_outputs_counter.past_key_values
            x_model_past_counter = [_.detach() for _ in x_model_past_counter]

    # For right to left model
    rl_reverse_index = torch.arange(y_logits.shape[1] - 1, -1, -1)
    mask_t = None

    for iter in range(args.num_iters):
        optim.zero_grad()
        y_logits_ = y_logits + epsilon

        soft_forward_y = y_logits_ / 0.001
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask) / 0.001

        y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)
        # print(soft_forward_x.shape)
        # print(soft_forward_y.shape)
        # print(y_logits_t.shape)

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)

        # Compute loss, gradients, and update.
        lr_nll_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=z_mask),
            y_logits_ / args.input_lgt_temp)

        if args.lr_nll_portion == 1.0:
            rl_nll_loss = lr_nll_loss
        else:
            # add right-to-left model (rl)
            if "counterfactual" in args.mode or "sentiment" in args.mode:
                y_logits_rev = y_logits_[:, rl_reverse_index, :]
                y_logits_rev_t = model_back(y_logits_rev.argmax(-1) + 1).logits[:, :-1, :]
                y_logits_rev_t = y_logits_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
                rl_nll_loss = soft_nll(
                    top_k_filter_3d(y_logits_rev_t / args.output_lgt_temp, args.rl_topk),
                    y_logits_rev[:, 1:] / args.input_lgt_temp)
            elif "abductive" in args.mode or "lexical" in args.mode:
                yz_logits_rev = torch.flip(torch.cat([y_logits_, z_onehot], dim=1), [1])
                yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
                yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
                yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]

                tmp_logits = yz_logits_rev_rev_t_
                repetition_mask = torch.cat([F.softmax(tmp_logits[:, 1:, :], dim=-1),
                                             torch.zeros_like(tmp_logits[:, -1:, :])], dim=1)
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

                rl_nll_loss = soft_nll(
                    top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
                    y_logits_ / args.input_lgt_temp)


        if "counterfactual" in args.mode:
            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask).transpose(0, 1),
                target_idx=z_t,
                ngram_list=list(range(2, args.counterfactual_max_ngram + 1))
            )

            if x_counter != "":
                # pred(x_counter; y) that is given x_counter, the model loss to predict y
                output = model(x_counter_t)
                
                c_loss = -args.counterfactual_c1_weight*c_loss_1 + args.counterfactual_c2_weight * c_loss_2
            else:
                c_loss = c_loss_2        
        elif "abductive" in args.mode or "lexical" in args.mode:
            soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

            # Reshaping
            bz = args.batch_size
            lg = xyz_logits.shape[1]
            st = xy_length - 1
            ed = xyz_logits.shape[1] - 1
            xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
            z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

            c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
                z_logits,
                z_t.view(-1))
            c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=y_logits_.transpose(0, 1),
                target_idx=zz_t,
                ngram_list=[1]
            )
            c_loss = c_loss_1 + args.abductive_c2_weight * c_loss_2
        elif "sentiment" in args.mode:
            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask).transpose(0, 1),
                target_idx=z_t,
                ngram_list=list(range(2, args.sentiment_max_ngram + 1))
            )

            if "modelling-1" in args.mode:
            # pred(z; y)
                soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
                xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)
                bz, lg, st, ed = args.batch_size, xyz_logits.shape[1], xy_length - 1, xyz_logits.shape[1] - 1
                xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
                z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)
                c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
                    z_logits,
                    z_t.view(-1))
                # print(z_logits.shape, "z_logits")
                # print(z_t.view(-1).shape, "z_t")
                c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

                # pred(y;z) -- c_loss_1_rev is the cross entropy loss involved in predicting y given z
                soft_forward_z_ = (z_onehot.detach() / 0.3 - z_onehot).detach() + z_onehot
                y_logits_rev = soft_forward(model, soft_forward_z_, soft_forward_y_, x_past=x_model_past)
                # print(y_logits_rev.shape, "y logits_rev")
                # print(y_logits_.shape, "y logits_")
                y_logits_rev = y_logits_rev.reshape(-1, y_logits_rev.shape[-1])
                # print(y_logits_rev.shape, "y logits_rev after reshape")
                y_logits_open = y_logits_.reshape(-1, y_logits_.shape[-1])
                y_t_s = y_logits_open.argmax(-1)
                # print(y_t_hehe.shape, "y_t_hehe")
                c_loss_1_rev = torch.nn.CrossEntropyLoss(reduction='none')(
                    y_logits_rev,
                    y_t_s)
                c_loss_1_rev = c_loss_1_rev.mean(-1)
                
                # final loss values
                c_loss = - args.sentiment_c1_weight*c_loss_1 - args.sentiment_c1_rev_weight * c_loss_1_rev + args.sentiment_c2_weight * c_loss_2            
  
            elif "modelling-2" in args.mode:
                # Use model_clf to predict the sentiment of the generated text
                # This can be done by calculating the cross entropy loss between the predicted sentiment and the next token being Negative token
                expected_token = torch.tensor([tokenizer_clf.encode("negative")[1]], device="cuda")
                expected_token = expected_token.repeat(args.batch_size)
                predicted_sentiment = model_clf(y_logits_.argmax(-1)).logits
                predicted_sentiment = predicted_sentiment.view(-1, predicted_sentiment.shape[-1])/0.3
                predicted_sentiment = torch.nn.functional.softmax(predicted_sentiment, dim=-1)
                c_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                    predicted_sentiment,
                    expected_token)
                c_loss = c_loss.mean(-1) + args.sentiment_c2_weight * c_loss_2

        loss = (1.0 - args.constraint_weight) * args.lr_nll_portion * lr_nll_loss \
               + (1.0 - args.constraint_weight) * (1 - args.lr_nll_portion) * rl_nll_loss \
               + args.constraint_weight * c_loss      
        loss = loss.mean()

        if iter < args.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            text, _, _ = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)
            for bi in range(args.batch_size):
                if "abductive" in args.mode or "lexical" in args.mode:
                    print(
                        "%d, loss: %.4f, lr_nll_loss: %.4f, rl_nll_loss: %.4f,  c_loss_2: %.4f, lr: %.4f, |%s|" % (
                            iter + 1, loss.item(), lr_nll_loss[bi].item(), rl_nll_loss[bi].item(),
                            c_loss_2[bi].item(), last_lr, text[bi]))
                    # print("%d, loss: %.4f, lr_nll_loss: %.4f, rl_nll_loss: %.4f, c_loss_1: %.4f, c_loss_2: %.4f, lr: %.4f, |%s|" % (iter + 1, loss.item(), lr_nll_loss[bi].item(), rl_nll_loss[bi].item(), c_loss_1[bi].item(), c_loss_2[bi].item(), last_lr, text[bi]))
                else:
                    print("%d, loss: %.4f, lr_nll_loss: %.4f, c_loss: %.4f, lr: %.4f, |%s|" % (
                    iter + 1, loss.item(), lr_nll_loss[bi].item(), c_loss[bi].item(), last_lr, text[bi]))

            if "abductive" in args.mode or "lexical" in args.mode:
                pass

        if args.wandb:
            wandb.log(
                {"Loss": loss.item(),
                 "left-to-right nll loss": torch.mean(lr_nll_loss).item(),
                 "right-to-left nll loss": torch.mean(rl_nll_loss).item(),
                 "constraint loss": torch.mean(c_loss).item(),
                 "Gassian_Noise_STD": noise_std,
                 "LR": last_lr,
                 "Gradient": torch.norm(epsilon.grad).detach().clone().data.cpu().numpy()}
            )

        ## noise
        if iter < args.num_iters - 1:

            if 'grammar' in args.mode:
                continue

            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(),
                                     device='cuda', requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    if args.wandb:
        wandb.finish()


    if(args.beam_search > 0):
        text, last_text_ids = beam_search_decode(
            model, y_logits_, args.beam_search, soft_forward_x, x_model_past, tokenizer, args.batch_size,extra_mask=z_mask)
    else:
        text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)

    
    last_text_ids = last_text_ids.to(device)

    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    if args.verbose:
        for bi in range(args.batch_size):
            print("[final]: %s\n%.4f" % (text[bi], ppl_last))
            print("[final complete sentence]: %s\n" % text_post[bi])

    return ppl_last, text, text_post

# sentiment transfer without premise. Also assuming that the input is a single sentence
def sentiment_transfer(model, tokenizer, device, args, model_back=None):
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]

    outfile = "topk" + str(args.topk) + "_len" + str(args.length) + "_cw" + str(args.constraint_weight) + "_lrnll" + str(args.lr_nll_portion) + "_c1" + str(args.sentiment_c1_weight) + "_c1rev" + str(args.sentiment_c1_rev_weight) + "_c2" + str(args.sentiment_c2_weight) + "_ngram" + str(args.sentiment_max_ngram) + ".json"

    print("outputs: %s" % outfile)

    # If want to re-infer with same weights then change mode from 'w' to 'a'
    fw = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    if "modelling-2" in args.mode:
        tokenizer = AutoTokenizer.from_pretrained(args.sentiment_classifier)
        model_clf = GPT2LMHeadModel.from_pretrained(args.sentiment_classifier)
        model.resize_token_embeddings(len(tokenizer))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_clf.to(device, dtype=torch.float32)
    else:
        model_clf = None

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        x = "<|endoftext|>"
        z = ". " + d["positive_stmt"]

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            torch.cuda.empty_cache()
            ppl_last, text, text_post = decode(model, tokenizer, device, x, z, args=args, model_back=model_back, model_clf=model_clf)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'z': z,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)
        print('Output to: \t', outfile)

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)

def counterfactual_reasoning(model, tokenizer, device, args, model_back=None):
    fr = open(args.input_file, 'r')
    data = [json.loads(x) for x in fr.readlines()]
    loss_rerank = 'norerank' if args.no_loss_rerank else 'rerank'
    file_name = '%s_%s_seed%d_%d_%d_%s_ngram%d_cw%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
                '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_%s_%s_output.json' % (
                    args.version,
                    loss_rerank,
                    args.seed,
                    args.start,
                    args.end,
                    args.mode,
                    args.counterfactual_max_ngram,
                    args.constraint_weight,
                    args.lr_nll_portion,
                    args.length,
                    args.topk,
                    args.num_iters,
                    args.frozen_length,
                    args.win_anneal_iters,
                    args.noise_iters,
                    args.gs_std,
                    args.stepsize,
                    args.large_noise_iters,
                    args.large_gs_std)

    outfile = os.path.join(args.output_dir, file_name)
    fw = open(outfile, 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + file_name), 'w')
    fw_res = open(os.path.join(args.output_dir, 'res_' + file_name), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        if args.seed != -1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)
        print('output-lgt-temp:\t', args.output_lgt_temp)

        premise = d.get('premise', "")
        counterfactual = d.get('counterfactual', "")

        x = premise + ' ' + counterfactual
        x_counter = premise + ' ' + d.get('initial', "")
        ori_ending = d.get('original_ending', "")
        ori_endings = tokenize.sent_tokenize(ori_ending)

        # dont remove processed
        if x in procssed:
            continue
        else:
            procssed.add(x)

        x_text_so_far = [""]
        x_addon = [[x]]

        x_counter_text_so_far = [""]
        x_counter_addon = [[x_counter]]

        outputs = []
        for oi, z_sent in enumerate(ori_endings):
            print("Sentence %d" % oi)
            z_text_so_far = z_sent.strip()
            z_text_so_far = ". " + z_text_so_far

            assert len(x_text_so_far) == len(x_addon), "%d vs %d" % (len(x_text_so_far), len(x_addon))

            new_x_text_so_far = []
            new_x_addon = []
            for ii, text_i in enumerate(x_text_so_far):
                for text_j in x_addon[ii]:
                    text_ij = text_i.strip() + " " + text_j.strip()
                    new_x_text_so_far.append(text_ij)

                    text_ij = text_ij.strip()

                    ppl_last, text, text_post = decode(
                        model, tokenizer, device, text_ij, z_text_so_far, None, args, model_back=model_back, x_counter=x_counter)

                    outputs.append([text_ij, text_post])

                    #  Rank and filter text_post from util.py:
                    text_post = [post_sent(x) for x in text_post]
                    text_post = rank_and_filter(text_post, text_ij, z_text_so_far, model, tokenizer, device,
                                                args.no_loss_rerank)

                    if ii == len(x_text_so_far) - 1 and oi == len(ori_endings) - 1:
                        last_output = text_post
                        final_res = ' '.join([text_ij, last_output])
                        outputs.append(final_res)
                        fw_res.write(final_res + '\n')
                        fw_res.flush()

                    new_x_addon.append([text_post])

            x_text_so_far = new_x_text_so_far
            x_addon = new_x_addon

            break

        complete_output = outputs
        out = {
            'premise': premise,
            'initial': d.get('initial', ""),
            'counterfactual': counterfactual,
            'original_ending': ori_ending,
            'generation_complete': complete_output,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def grammar_correction(model, tokenizer, device, args, model_back=None):
    fr = open(args.input_file, 'r')
    data = [x.strip() for x in fr.readlines()]
    file_name = '%s_seed%d_%d_%d_%s_cw%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
                '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_%s_%s_output.json' % (
                    args.version,
                    args.seed,
                    args.start,
                    args.end,
                    args.mode,
                    args.constraint_weight,
                    args.lr_nll_portion,
                    args.length,
                    args.topk,
                    args.num_iters,
                    args.frozen_length,
                    args.win_anneal_iters,
                    args.noise_iters,
                    args.gs_std,
                    args.stepsize,
                    args.large_noise_iters,
                    args.large_gs_std)

    outfile = os.path.join(args.output_dir, file_name)
    fw = open(outfile, 'w')

    # Grammar
    data = [[' '.join(x.split()[:3]), ' '.join(x.split()[3:])] for x in data]
    print('#data: ', len(data))

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        if len(d[1].split()) <= 4:
            text = [d[1][2:]]
            text_post = [d[1][2:]]
            continue

        x = d[0]
        y = d[1]

        y = ". " + y

        ppl_last, text, text_post = decode(
            model, tokenizer, device, x, y, None, args, model_back=model_back)
        out = {
            'original': x + " " + y,
            'generation': text,
            'generation_complete': text_post,
        }

        fw.write(json.dumps(out) + '\n')

    print("outputs: %s" % outfile)



def _get_adverbs_and_nnps(z_words):
    pos = nltk.pos_tag(z_words)
    adverbs = [w[0] for w in pos if 'RB' in w[1]]
    nnps = [w[0] for w in pos if 'NNP' in w[1]]
    return adverbs, nnps

def _get_keywords(z, x, args):
    stop_words = set(stopwords.words('english'))
    z_words = word_tokenize(z)
    z_adverbs, z_nnps = _get_adverbs_and_nnps(z_words)
    ret_words = []
    for w in z_words:
        if w in z_nnps:
            if w not in ret_words:
                ret_words.append(w)
        else:
            w = w.lower()
            if w not in stop_words and w.isalnum() and w not in z_adverbs and w not in ret_words:
                ret_words.append(w)

    if args.abductive_filterx:
        x_words = word_tokenize(x)
        ret_words = [w for w in ret_words if w not in x_words]

    return ' '.join(ret_words)

def abductive_reasoning(model, tokenizer, device, args, model_back=None):
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]

    outfile = '%s_seed%d_%d_%d_%s_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
              '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.version,
                  args.seed,
                  args.start,
                  args.end,
                  args.mode,
                  args.constraint_weight,
                  args.abductive_c2_weight,
                  args.lr_nll_portion,
                  args.length,
                  args.topk,
                  args.num_iters,
                  args.frozen_length,
                  args.win_anneal_iters,
                  args.noise_iters,
                  args.gs_std,
                  args.stepsize,
                  args.stepsize_ratio,
                  args.stepsize_iters,
                  args.large_noise_iters,
                  args.large_gs_std)
    print("outputs: %s" % outfile)

    fw = open(os.path.join(args.output_dir, outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        if args.if_zx:
            x = d["obs2"].strip() + '<|endoftext|>' + d["obs1"].strip()
        else:
            x = d["obs1"].strip()
        z = d["obs2"].strip()
        z_keywords = _get_keywords(z, d["obs1"].strip(), args)

        if ' '.join([x, z]) in procssed:
            continue
        procssed.add(' '.join([x, z]))

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords

        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(model, tokenizer, device, x, z, None, args,
                                               model_back=model_back, zz=z_keywords)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)


        out = {
            'x': x,
            'z': z,
            'z_keywords': z_keywords,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()

    print("outputs: %s" % outfile)


def lexical_generation(model, tokenizer, device, args, model_back=None):
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]

    outfile = '%if_zx%s_seed%d_%d_%d_%s_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
              '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.if_zx,
                  args.version,
                  args.seed,
                  args.start,
                  args.end,
                  args.mode,
                  args.constraint_weight,
                  args.abductive_c2_weight,
                  args.lr_nll_portion,
                  args.length,
                  args.topk,
                  args.num_iters,
                  args.frozen_length,
                  args.win_anneal_iters,
                  args.noise_iters,
                  args.gs_std,
                  args.stepsize,
                  args.stepsize_ratio,
                  args.stepsize_iters,
                  args.large_noise_iters,
                  args.large_gs_std)
    print("outputs: %s" % outfile)

    # If want to re-infer with same weights then change mode from 'w' to 'a'
    fw = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d["concept_set"])
        constraints = d["concept_set"].split("#")

        constraints = ' '.join(constraints)
        x = "<|endoftext|>"
        z = constraints
        z_keywords = constraints

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords

        text_candidates = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            torch.cuda.empty_cache()
            ppl_last, text, text_post = decode(model, tokenizer, device, x, z, None, args, model_back=model_back,
                                               zz=z_keywords)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'constraints': constraints,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)
        print('Output to: \t', outfile)

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def main():
    args = options()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, output_hidden_states=True,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
    model.to(device)
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

    model_back = OpenGPT2LMHeadModel.from_pretrained(
        args.back_model, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_first_dropout=0)
    model_back.to(device)
    model_back.eval()
    # Freeze GPT-2 weights
    for param in model_back.parameters():
        param.requires_grad = False


    if "counterfactual" in args.mode:
        counterfactual_reasoning(model, tokenizer, device, args, model_back)
    if "abductive" in args.mode:
        abductive_reasoning(model, tokenizer, device, args, model_back)
    if "lexical" in args.mode:
        lexical_generation(model, tokenizer, device, args, model_back)
    if "grammar" in args.mode:
        grammar_correction(model, tokenizer, device, args, model_back)
    if "sentiment" in args.mode:
        sentiment_transfer(model, tokenizer, device, args, model_back)


if __name__ == "__main__":
    main()
