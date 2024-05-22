from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

model_untrained = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model_trained = SentenceTransformer('./models/model_trained_2_bs8_epoch15/')

df_nord = pd.read_csv('data/nordic.csv', sep=';')
df_rus = pd.read_csv('data/russian.csv', sep=';')

num = df_rus.count()['word'] + df_nord.count()['word']

embeddings_untrained = {}
embeddings_trained = {}

count = 0
end = len(df_rus) + len(df_nord)
mean = 0.0
import sys
for x in df_rus['word']:
    print("{:3.2f}".format(count / end * 100) + '%', end='\r')
    if pd.notna(x) :
        embeddings_untrained[x] = (model_untrained.encode(x))
        embeddings_trained[x] = (model_trained.encode(x))
    count += 1
    sys.stdout.flush()

for x in df_nord['word']:
    print("{:3.2f}".format(count / end * 100) + '%', end='\r')
    if pd.notna(x) :
        embeddings_untrained[x] = (model_untrained.encode(x))
        embeddings_trained[x] = (model_trained.encode(x))
    count += 1
    sys.stdout.flush()

mean /= len(embeddings_untrained)
print('writing...')
import pickle
f1 = open('data/trained_5.pkl', 'wb')
f2 = open('data/untrained.pkl', 'wb')
pickle.dump(embeddings_trained, f1)
pickle.dump(embeddings_untrained, f2)
f1.close()
f2.close()


