#!/usr/bin/env python
# coding: utf-8



# A Sentiment analyzer (5-class: 0 = strongly negative, 1 = negative, 2 = neutral, 3 = positive 4 = strongly positive) implemented with Elmo language model.

# Trained on stanfordSentimentTreebank
# Training and prediction code provided.


# Loading imports

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer

from examples.sentiment.sst_classifier import LstmClassifier
from realworldnlp.predictors import SentenceClassifierPredictor



#Model definition and training

HIDDEN_DIM = 512
CUDA_DEVICE = 0


# In order to use ELMo, each word in a sentence needs to be indexed with
# an array of character IDs.
elmo_token_indexer = ELMoTokenCharactersIndexer()
reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={'tokens': elmo_token_indexer})

train_dataset = reader.read('https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/train.txt')
dev_dataset = reader.read('https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/dev.txt')

# Initialize the ELMo-based token embedder using a pre-trained file.
# This takes a while if you run this script for the first time

# Original
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Medium
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

# Use the 'Small' pre-trained model
# options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
#                '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
# weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
#               '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                  min_count={'tokens': 3})

# Pass in the ElmoTokenEmbedder instance instead
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

# The dimension of the ELMo embedding will be 2 x [size of LSTM hidden states]
elmo_embedding_dim = 1024
lstm = PytorchSeq2VecWrapper(
    torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM , batch_first=True))

model = LstmClassifier(word_embeddings, lstm, vocab)
optimizer = optim.AdamW(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  patience= 1,
                  num_epochs=20,
                  cuda_device= CUDA_DEVICE)

trainer.train()


# In[5]:


#Save model (change filename if needed)

with open("/.../sentiment_model1", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/.../sentiment_vocab1")

print("Model saved. DONE")


# In[6]:


# Reload pretrained model

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer

from examples.sentiment.sst_classifier import LstmClassifier
from realworldnlp.predictors import SentenceClassifierPredictor

HIDDEN_DIM = 512
CUDA_DEVICE = 0

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo_embedding_dim = 1024
vocab = Vocabulary.from_files("/.../sentiment_vocab1")


lstm = PytorchSeq2VecWrapper(
    torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM , batch_first=True))
elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
model = LstmClassifier(word_embeddings, lstm, vocab)

# Reload the trained model.
with open("/.../sentiment_model1", 'rb') as f:
    model.load_state_dict(torch.load(f))
    model.eval()
    
print(model)



# Perform predictions

elmo_token_indexer = ELMoTokenCharactersIndexer()
reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={'tokens': elmo_token_indexer})

tokens = 'Severe acute respiratory syndrome SARS emerged in 2003 as a new epidemic form of life-threatening situations'
predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
logits = predictor.predict(tokens)['logits']
label_id = np.argmax(logits)

print(model.vocab.get_token_from_index(label_id, 'labels'))


# In[ ]:




