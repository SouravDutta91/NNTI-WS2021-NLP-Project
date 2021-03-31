import re
import string

import os
import time
import torch
import pickle
import datetime
import numpy as np
import pandas as pd

def punctuations_remove(input):
  output = "".join([x for x in input if x not in string.punctuation])
  return output

def numbers_remove(input):
  output = re.sub(r"[0-9]+", "", input)
  return output

def usernames_remove(input):
  output = re.sub(r"@\S+", "", input)
  return output

def hashtag_remove(input):
  output = re.sub(r"#\S+", "", input)
  return output

def http_remove(input):
  output = re.sub(r"http\S+", "", input)
  return output

def emojis_remove(input):
  EMOJI_PATTERN = re.compile(
      "["
      "\U0001F1E0-\U0001F1FF"  # flags (iOS)
      "\U0001F300-\U0001F5FF"  # symbols & pictographs
      "\U0001F600-\U0001F64F"  # emoticons
      "\U0001F680-\U0001F6FF"  # transport & map symbols
      "\U0001F700-\U0001F77F"  # alchemical symbols
      "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
      "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
      "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
      "\U0001FA00-\U0001FA6F"  # Chess Symbols
      "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
      "\U00002702-\U000027B0"  # Dingbats
      "\U000024C2-\U0001F251" 
      "]+"
  )
  
  output = EMOJI_PATTERN.sub(r'',input)
  return output

def extra_whitespaces(input):
  output = input.replace('\s+', ' ')
  return output

def stopwords_remove(m):
  hindi_stopwords = pd.read_csv('https://raw.githubusercontent.com/stopwords-iso/stopwords-hi/master/stopwords-hi.txt').stack().tolist()
  english_stopwords = pd.read_csv('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt').stack().tolist()
  stopwords = hindi_stopwords + english_stopwords

  output = pd.Series(m).apply(lambda x: [item for item in x.split() if item not in stopwords])
  return output

def tolower(input):
  output = input.lower()
  return output

def corpus_preprocess(corpus):
  corpus = corpus.apply(lambda x: tolower(x))
  corpus = corpus.apply(lambda x: emojis_remove(x))
  corpus = corpus.apply(lambda x: http_remove(x))
  corpus = corpus.apply(lambda x: hashtag_remove(x))
  corpus = corpus.apply(lambda x: numbers_remove(x))
  corpus = corpus.apply(lambda x: usernames_remove(x))
  corpus = corpus.apply(lambda x: punctuations_remove(x))
  corpus = corpus.apply(lambda x: stopwords_remove(x))
  corpus = corpus.apply(lambda x: extra_whitespaces(x))
  return corpus


def proc_all(text):
  #text = corpus_preprocess(text)
  print("Started Preprocessing")
  cleanstart = time.time()
  text = corpus_preprocess(text)
  cleanend = str(datetime.timedelta(seconds = time.time()-cleanstart))
  print("Preprocessing ended!")
  print("Pre-processing the text took {}".format(cleanend))
  print("===========================================================")
  print("-------")
  
  c = []
  for sent in text[0]:
    a = " ".join(sent)
    c.append(a)
  d = pd.DataFrame(c,columns=["text"])
  with open('hindi_corpus_cleaned.pkl','wb') as f:
      pickle.dump(d,f)
  return d

