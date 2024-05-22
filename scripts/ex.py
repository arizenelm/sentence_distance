from sentence_transformers import SentenceTransformer
#from scipy.spatial.distance import cosine
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

#Our sentences we like to encode
sentences = ['Пахотные земли и постоянные пастбища',
    'Посевные площади сельскохозяйственных культур',
    'Зарегистрированные автомобили']
#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
#print(cosine(sentence_embeddings[0], sentence_embeddings[1]))
#print(cosine(sentence_embeddings[0], sentence_embeddings[2]))
#print(cosine(sentence_embeddings[1], sentence_embeddings[2]))