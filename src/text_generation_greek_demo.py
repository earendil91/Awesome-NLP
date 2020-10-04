#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:25:42 2020

@author: lighteternal
"""

# A simple text generation script for Greek, using the AUEB-BERT model. 

# Given a sequence (text) the model will predict the top5 most likely values of the {tokenizer.mask_token}.

# Note that BERT is a bi-directional masked language model (reads text both ways) and
# has limited success in language generation. 
# Causal/autoregressive models like GTP-2/3 who process text from left to right 
# are better-suited for this task. 

from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModelWithLMHead.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

sequence = f"Το αγαπημένο μου φαγητό είναι το {tokenizer.mask_token}."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
top_k_weights, top_k_indices = torch.topk(mask_token_logits, 5)

probabilities = torch.topk(torch.nn.functional.softmax(mask_token_logits, dim=1),5).values.tolist()[0]
decoded_tokens = list(tokenizer.decode([token]) for token in top_5_tokens)


for token,probability in zip(decoded_tokens, probabilities):
    #print(tokenizer.decode([token]), probability)
    print('{} {}'.format(token, probability))
    
