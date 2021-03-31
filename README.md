# NNTI Final Project (Sentiment Analysis & Transfer Learning)
NNTI (WS-2021), Saarland University, Germany




## Introduction
Neural networks can be applied to many NLP tasks, such as text classification. In this report, ourgoal is to use a neural network architecture to correctly predict hate speech on the given dataset. Thereport is broadly composed of three main parts.Firstly, since neural networks only operate on numerical data and not on string or character, we needto convert our text data in some form of numerical representation before feeding it to the model.In contrast to traditional NLP approaches which associate words with discrete representations likeone-hot encoding method, we use word embeddings to represent words by dense, low-dimensionaland real-valued vectors.Then we use the word embeddings we get to make a binary neural sentiment classifier for the Hindidataset with a LSTM model. In this classifier, we are supposed to classify texts into two class, namely:Hate and Offensive(HOF) and Non-Hate and offensive(NOT). And then we apply this classifier toBengali dataset using the knowledge of transfer learning.Finally, we try to make improvement on our accuracy results by changing our model architecture intoconvolutional neural network(CNN) model.


### Files
Task 1 - Task1_Word_Embeddings.ipynb
Task 2 - LSTM-classifier-Task-2.ipynb
Task 3 - CNN-task-3.ipynb

##### Authors
Nishant Gajjar - 2577584 
[s8nigajj@stud.uni-saarland.de](s8nigajj@stud.uni-saarland.de)

Zhifei Li - 7010552
[zhli00001@stud.uni-saarland.de](zhli00001@stud.uni-saarland.de)

