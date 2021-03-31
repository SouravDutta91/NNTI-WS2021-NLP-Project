import re
import string

import os
import time
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt

from preprocess import proc_all

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  print("=================================")
  print("GPU found")
  print("Using GPU at cuda:",torch.cuda.current_device())
  print("=================================")
  print(" ")

data = pd.read_csv('https://raw.githubusercontent.com/SouravDutta91/NNTI-WS2021-NLP-Project/main/data/hindi_hatespeech.tsv',sep='\t')

data_text = data['text']
textd = data_text

file = "~/NNTI-WS2021-NLP-Project/Project_files/hindi_corpus_cleaned.pkl"

if Path(file).is_file():
  text = pd.read_pickle(file)
  print("Loaded the clean corpus")
else:
  text = proc_all(textd)

text = text['text']
text = text.apply(lambda x: x.split())

print(text)

V = list(set(text.sum())) #List of unique words in the corpus
all_words = list(text.sum()) #All the words without removing duplicates

print("Total number of unique words are: ",len(V))

#Dictionaries of words and their indexes
word_index = {word: i for i,word in enumerate(V)}
index_word = {i: word for i,word in enumerate(V)}

def word_to_one_hot(word):
  id = V.index(word)
  onehot = [0.] * len(V)
  onehot[id] = 1.
  return torch.tensor(onehot)

get_onehot = dict((word, word_to_one_hot(word)) for word in V)

def sampling_prob(word):
  if word in all_words:
    count = all_words.count(word)
    zw_i = count / len(all_words)
    p_wi_keep = (np.sqrt(zw_i/0.001) + 1)*(0.001/zw_i)
  else:
    p_wi_keep = 0
  return p_wi_keep

def get_target_context(sentence,window):
  for i,word in enumerate(sentence):
    target = word_index[sentence[i]]

    for j in range(i - window, i + window):
      if j!=i and j <= len(sentence)-1 and j>=0:
        if sampling_prob(sentence[j]) > thres:
          context = word_index[sentence[j]]
          yield target,context

embedding_size = 300
learning_rate = 0.05
epochs = 300 
thres = np.random.random()


# Create model 
class Word2Vec(nn.Module):
  def __init__(self):
    super().__init__()
    self.v_len = len(V)
    self.es = embedding_size
    self.epochs = epochs
    
    self.w1 = nn.Linear(len(V),embedding_size,False)
    self.w2 = nn.Linear(embedding_size,len(V))
    self.soft = nn.LogSoftmax(dim = 1)

  def forward(self, one_hot):
    one_hot = self.w1(one_hot)
    one_hot=self.w2(one_hot)
    output=self.soft(one_hot)
    return output.cuda()

  def softmax(self,input):    
    output = self.soft(input)
    return output

# Define optimizer and loss
model = Word2Vec().cuda()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,nesterov=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

print("=====================================")
print("The Word2Vec model: ")
print(model)
print("=====================================")

'''Gets the corpus and creates the training data with the target and its context
and returns a dataframe containing them in terms of their indexes.
'''
def get_training_data(corpus,window):
  t,c = [],[]
  for sentence in corpus:
    data = get_target_context(sentence,window)
    for i,j in data:
      x = get_onehot[index_word[i]]
      t.append(x)
      c.append(j)
  t_data = pd.DataFrame(list(zip(t,c)),columns=["target","context"])
  return t_data

def generateDirectoryName(batchsize,x=0):
  path = "m{}_e{}_lr{}_bs{}_es{}".format(len(text),epochs,learning_rate,batchsize,embedding_size)
  while True:
      dir_name = (path + ('_' + str(x) if x != 0 else '')).strip()
      if not os.path.exists(dir_name):
          os.mkdir(dir_name)
          print("Successfully created a directory at ",dir_name)
          return dir_name
      else:
          x = x + 1

def save_model(path,epoch):
  torch.save(model.state_dict(),"{}/epoch_{}.pt".format(path,epoch))

def save_vocab(path):
  with open(path+'/vocab.pkl','wb') as f:
    pickle.dump(V,f)
  print("Dumped vocabulary successfully")

def save_wordindex(path):
  with open(path+'/word_index.pkl','wb') as f:
    pickle.dump(word_index,f)

  with open(path+'/index_word.pkl','wb') as f:
    pickle.dump(index_word,f)
  print("Dumped word-index dictionaries successfully")

def save_loss(loss):
  with open(path+'/optimizer_loss.pkl','wb') as f:
    pickle.dump(loss,f)
    
  print("Written loss and optimizer successfully")

def train(traindata,batchsize):
  losses = []
  runs = trange(1,epochs+1)
  tqdm.write("Training started")
  for epoch in runs:
    total_loss = []
    for wt,wc in zip(DataLoader(traindata.target.values,batch_size=batchsize),
                     DataLoader(traindata.context.values,batch_size=batchsize)):
      wt = wt.cuda()
      wc = wc.cuda()
      optimizer.zero_grad()
      output = model(wt)
      loss = criterion(output,wc)
      total_loss.append(loss.item())
      loss.backward()
      optimizer.step()

    if epoch % 50 == 0 :
      start = time.time()
      tqdm.write("===========================================================")
      tqdm.write("Saving the model state")
      save_model(path,epoch)
      end = str(datetime.timedelta(seconds = time.time()-start))
      tqdm.write("Model state saved. It was completed in {}".format(end))      
      tqdm.write("===========================================================")

    time.sleep(0.1)
    tqdm.write("At epoch {} the loss is ({})".format(epoch ,round(np.mean(total_loss),3)))
    losses.append(np.mean(total_loss))

  plt.xlabel("Epochs")
  plt.ylabel("LOSS")
  save_loss(losses)
  plt.plot(losses)
  plt.savefig(path+'/Plot.png')

# Set hyperparameters
# window_size = 2 defined where the model is being trained
batch_size = 60
window_size = 2

start2 = time.time()
print("=================================================")
print("Collecting training data")
starte = time.time()
data = get_training_data(text,window_size)
ende = str(datetime.timedelta(seconds = time.time()-starte))
print("It took {} to collect the data".format(ende))
print("The training data has {} target-context pairs".format(len(data)))
print("===================================================")

print("Sampling Threshold: ",thres)

path = generateDirectoryName(batch_size)
train(data,batch_size)

end2 = str(datetime.timedelta(seconds = time.time()-start2))
print("Training finished.")
print("It took {} to finish training the model".format(end2))

print("===========================================================")
print("Saving Vocabulary and Word-Index dictionaries ")
print("===========================================================")
start1 = time.time()
save_vocab(path)
save_wordindex(path)
end1 = str(datetime.timedelta(seconds = time.time()-start1))
print("===========================================================")
print("Saved Successfully in {}".format(end1))
print("===========================================================")
