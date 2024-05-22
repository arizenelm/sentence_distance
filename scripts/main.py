from pymystem3 import Mystem
#from ruwordnet import RuWordNet
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")





wn = RuWordNet()

from ruwordnet import RuWordNet
wn = RuWordNet(filename_or_session='ruwordnet.db')

for sense in wn.get_senses('налог'):
    print(sense.synset.causes)
    #print()
    #print(sense.synset.domains)


#print(wn.get_senses("число")[0].synset.title.split(sep=', '))




original_text = 'Выдано экземпляров из библиотечного фонда библиотек Минкультуры России'
m = Mystem()
lemmas = [word for word in m.lemmatize(original_text) if not word.isspace()]
print(lemmas)


my_dict = dict()
for word in lemmas:
    synonyms = [title.lower() for title in [sense.synset.title for sense in wn.get_senses(word)]]
    my_dict[word] = synonyms

