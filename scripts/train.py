from sentence_transformers import SentenceTransformer, InputExample, losses, models
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from pymystem3 import Mystem
import itertools
import pandas as pd
import numpy as np




model = SentenceTransformer('distiluse-base-multilingual-cased-v1')




df_match = pd.read_csv('data/match.csv', sep=';')
df_match.insert(len(df_match.columns), 'similarity', pd.NA)
df_match.insert(len(df_match.columns), 'similarity trained', pd.NA)

for i in range(len(df_match)) :
    embeddings = model.encode([df_match['nord'][i], df_match['rus'][i]])
    df_match.loc[i, 'similarity'] = 1 - cosine(embeddings[0], embeddings[1])


df_nordic = pd.read_csv('data/nordic.csv', sep=';')
df_russian = pd.read_csv('data/russian.csv', sep=';')


match_nr = {df_match['nord'][i] : df_match['rus'][i] 
         for i in np.random.permutation(len(df_match))[:int(len(df_match) / 2)]}
match_rn = {df_match['rus'][i] : df_match['nord'][i] 
         for i in range(len(df_match))}

random_russian = np.random.permutation(df_russian['word'])

random_nordic = np.random.permutation(df_nordic['word'])

unmatch = []
for i, j in itertools.product(range(len(random_nordic)), range(len(random_russian))) :
    word = random_russian[j]
    if word not in match_rn or match_rn[word] != random_nordic[i] :
        unmatch.append((word, random_nordic[i]))
    if len(unmatch) == 100:
        break

print(len(unmatch))





train_examples = [InputExample(texts=[key, match_nr[key]], label=1) for key in match_nr]
train_examples = train_examples + [InputExample(texts=[x[0], x[1]], label=0) for x in unmatch]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.ContrastiveLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=2,
          warmup_steps=100,
          show_progress_bar=True,
          output_path='./models/model_trained')