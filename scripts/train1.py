from sentence_transformers import SentenceTransformer, models
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import numpy as np
import pickle
import itertools

df_rus = pd.read_csv('data/russian.csv', sep=';')
df_nord = pd.read_csv('data/nordic.csv', sep=';')
df_match = pd.read_csv('data/match.csv', sep=';')
match_nr = {}
match_rn = {}
for i in df_match.index :
    match_nr[df_match['nord'][i]] = df_match['rus'][i]
    match_rn[df_match['rus'][i]] = df_match['nord'][i]
with open('data/untrained.pkl', 'rb') as f :
    untrained = pickle.load(f)

train = {}
test = {}
for i in df_match.index[0:45] :
    train[df_match['nord'][i]] = [df_match['rus'][i]]

for i in df_match.index[45:] :
    test[df_match['nord'][i]] = df_match['rus'][i]




import sortedcontainers
for word1 in train :
    distances = sortedcontainers.SortedDict()
    for word2 in df_rus['word'] :
        distance = cosine(untrained[word1], untrained[word2])
        distances[distance] = word2
    for d, w in distances.items() :
        if len(train[word1]) == 15 :
            break
        if w not in train[word1] :
            train[word1].append(w)


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

train_examples = []

for key in train :
    train_examples.append(InputExample(texts=[key, train[key][0]], label=1))
    for word in train[key][1:] :
        train_examples.append(InputExample(texts=[key, word], label=0))


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.ContrastiveLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=5,
          warmup_steps=100,
          show_progress_bar=True,
          output_path='./models/model_trained_2')