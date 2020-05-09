import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import operator 
from tensorflow.keras.layers import Input, Model, Dense, Reshape, dot, Embedding, subtract, multiply
from tensorflow.keras import layers
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nltk.download('abc')
stop_words = set(stopwords.words('english')) 

words = nltk.corpus.abc.words()

n_words = 10000
def makedata(words,n_words):
  Dictionary = {}

  for word in words:
    if word in stop_words or len(word) <= 2:
      continue
    if word in Dictionary:
      Dictionary[word] += 1
    else:
      Dictionary[word] = 1

    
  sorted_dictionary = dict(sorted(Dictionary.items(), key=operator.itemgetter(1),reverse=True))
  # print(sorted_dictionary)
  # print(sorted_dictionary.values())
  integer_dict = {}
  list_of_words = list(sorted_dictionary.keys())
  i = 1
  for word in list_of_words:
    if word in stop_words or len(word) <= 2:
      continue
    if(i<n_words):
      integer_dict[word] = i
      i+=1
    else:
      integer_dict[word] = 0

  reversed_dictionary = dict(zip(integer_dict.values(), integer_dict.keys()))
  # print(integer_dict)
  data = []
  for word in words:
    if word in stop_words or len(word) <= 2:
      continue
    # print(word, integer_dict[word])
    data.append(integer_dict[word])
  return data, reversed_dictionary, integer_dict
data, reverse_dict, integer_dict = makedata(words, n_words)
print(integer_dict)
print(type(data), len(data))
window_size = 5

from tensorflow import keras

sampling_table = keras.preprocessing.sequence.make_sampling_table(n_words)
couples, labels = keras.preprocessing.sequence.skipgrams(data, n_words, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")



vector_dim = 400

valid_examples = ['Iraq', 'wheat', 'letters', 'team', 'system', 'technology']


input_target = Input((1,))  
input_context = Input((1,))

embedding = Embedding(n_words, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

similarity = dot([target, context], axes=1)
# sub = subtract([target,context])
# similarity = multiply([sub,sub])



dot_product = dot([target, context], axes=1, normalize = True)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)

model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

embedding_model = Model(inputs=input_target, outputs=target)


def get_sim_array(valid_word):
    # for i in range(len(valid_examples)):
        # valid_word = valid_examples[i]
      valid_number = integer_dict[valid_word]
      top_k = 15  # number of nearest neighbors
      sim = _get_sim(valid_number)
      similar_words = []
      nearest = np.argsort(-sim)[1:top_k + 1]
      print("Nearest to " + valid_word)
      for k in range(top_k):
          close_word = reverse_dict[nearest[k]]
          similar_words.append(close_word)
          print(close_word)
      print(" ")
      return similar_words

def _get_sim(valid_word_idx):
    sim = np.zeros((n_words,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    for i in range(n_words):
        in_arr1[0,] = valid_word_idx
        in_arr2[0,] = i
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        # print(out.shape)
        sim[i] = out[0]
    return sim


def visualisation():
  embedding_clusters = []
  word_clusters = []
  in_arr1 = np.zeros((1,))
  for word in valid_examples:
      embeddings = []
      words = []
      for similar_word in get_sim_array(word):
          words.append(similar_word)
          in_arr1[0,] = integer_dict[similar_word]
          embeddings.append(embedding_model.predict_on_batch(in_arr1).flatten())
      embedding_clusters.append(embeddings)
      word_clusters.append(words)

  embedding_clusters = np.array(embedding_clusters)
  # print(embedding_clusters.shape)
  n, m, k = embedding_clusters.shape
  tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
  embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k)))
  embeddings_en_2d = embeddings_en_2d.reshape(n, m, 2)
  return word_clusters, embeddings_en_2d


def tsne_plot_similar_words(labels, embedding_clusters):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, color in zip(labels, embedding_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, label=label)
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()


epochs = 1000000
arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for epoch in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if epoch % 10000 == 0:
        print("Iteration ",epoch," loss= ", loss)
    if epoch % 100000 == 0:
        word_clusters, embeddings_en_2d = visualisation()
        tsne_plot_similar_words(valid_examples, embeddings_en_2d)
