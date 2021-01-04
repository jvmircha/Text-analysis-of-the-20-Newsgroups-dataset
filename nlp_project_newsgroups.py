!pip install certifi==2020.11.8
!pip install chardet==3.0.4
!pip install click==7.1.2
!pip install cycler==0.10.0
!pip install gensim==3.8.3
!pip install idna==2.10
!pip install joblib==0.17.0
!pip install kiwisolver==1.3.1
!pip install matplotlib==3.1.3
!pip install nltk==3.5
!pip install numpy==1.19.4
!pip install Pillow==8.0.1
!pip install pyparsing==2.4.7
!pip install python-dateutil==2.8.1
!pip install regex==2020.11.13
!pip install requests==2.25.0
!pip install scikit-learn==0.23.2
!pip install scipy==1.5.4
!pip install six==1.15.0
!pip install sklearn==0.0
!pip install smart-open==3.0.0
!pip install threadpoolctl==2.1.0
!pip install tqdm==4.51.0
!pip install urllib3==1.26.2
!pip install wordcloud
!pip install pyLDAvis

"""## **Import Packages**"""

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re
import nltk
from nltk import stem
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#import collections
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LdaModel, tfidfmodel
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.stem import WordNetLemmatizer
import pyLDAvis
import pyLDAvis.gensim
from sklearn.cluster import SpectralClustering
from prettytable import PrettyTable
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import statistics

"""## **Downloads**"""

corporus = fetch_20newsgroups(remove=('headers','footers','quotes'))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""## **Preprocessing Method**"""

def preprocess_data(corporus):
  targets = corporus.target
  target_names = corporus.target_names
  word_pattern = '[0-9A-Za-z]'
  not_word_pattern = '[^A-Za-z]'
  stop_words = [word for word in stopwords.words('english')]
  corporus_data = corporus.data

  #lower case
  corporus_data = [text.lower() for text in corporus_data]

  #tokenization
  corporus_data = [text.split(' ') for text in corporus_data]

  tokenizer = RegexpTokenizer(
    r'\d+,\d+[,\.]\d+\w*|'
    '\d+[,\.]\d+\w*|'
    '[A-Za-z]+\-[A-Za-z]+\-[A-Za-z]+\-[A-Za-z]+|'
    '[A-Za-z]+\-[A-Za-z]+\-[A-Za-z]+|'
    '[A-Za-z]+\-[A-Za-z]+|'
    '\d+[:/]\d+[:/]\d+|'
    '\d+[:/]\d+|'
    '[\w]+\+\+|'
    '\w\w\w+|'
    '[A-Za-z]+(?=\-)'
    
    '[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
    )
  
  stemmer = stem.lancaster.LancasterStemmer()
  lemmatizer = WordNetLemmatizer()

  preprocessed_corpus = [[lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(document) if len(lemmatizer.lemmatize(word.lower())) >= 3 and re.match(word_pattern,word) and (not re.match(not_word_pattern,word)) and word.lower() not in stop_words] for document in corporus.data]
  dictionary = gensim.corpora.Dictionary(preprocessed_corpus)
  preprocessed_documents = [' '.join(document) for document in preprocessed_corpus]

  return preprocessed_corpus, dictionary, preprocessed_documents


"""## **Corpus Statistics Visualization**"""

corporus_frequencies = [key_value for key_value in sorted(dictionary.cfs.items(), key=lambda t: -t[1])]

corporus_frequencies_word_only  = [dictionary[corporus_frequencies[i][0]] for i in range(len(corporus_frequencies))]
corporus_frequencies_count_only = [corporus_frequencies[i][1] for i in range(len(corporus_frequencies))]
corporus_frequencies_word_index = [(dictionary[corporus_frequencies[i][0]] + ('  (%04d)'%(i))) for i in range(len(corporus_frequencies))]

PLOT_NUM = 100
PLOT_STEP_SIZE = 20
PLOT_RANGE = PLOT_NUM * PLOT_STEP_SIZE

x_pos = [num for num, _ in enumerate(corporus_frequencies_word_only[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE])]
plt.subplots(figsize=(22,8.5))
plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.2)
plt.bar(x_pos,corporus_frequencies_count_only[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE])
plt.xticks(x_pos, corporus_frequencies_word_index[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE],rotation=90,fontsize=8)
plt.xlabel('Word (Index)')
plt.ylabel('Word Count')
plt.title(f'A Sampling of the {PLOT_RANGE} Most Common Words in the Dataset')


document_frequencies = [key_value for key_value in sorted(dictionary.dfs.items(), key=lambda t: -t[1])]
document_frequencies_word_only  = [dictionary[document_frequencies[i][0]] for i in range(len(document_frequencies))]
document_frequencies_count_only = [document_frequencies[i][1] for i in range(len(document_frequencies))]
document_frequencies_word_index = [(dictionary[document_frequencies[i][0]] + ('  (%04d)'%(i))) for i in range(len(document_frequencies))]

PLOT_NUM = 100
PLOT_STEP_SIZE = 20
PLOT_RANGE = PLOT_NUM * PLOT_STEP_SIZE

x_pos = [num for num, _ in enumerate(document_frequencies_word_only[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE])]
plt.subplots(figsize=(22,8.5))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)
plt.bar(x_pos,document_frequencies_count_only[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE])
plt.xticks(x_pos, document_frequencies_word_index[0:(PLOT_NUM*PLOT_STEP_SIZE)-1:PLOT_STEP_SIZE],rotation=90,fontsize=8)
plt.xlabel('Word (Index)')
plt.ylabel('Document Count')
plt.title(f'A Sampling of the {PLOT_RANGE} Most Common Words by Document Frequency')

"""## **Dictionary_2k**"""

corporus_frequencies_2k = corporus_frequencies[0:2000]
temp_keys = [pair[0] for pair in corporus_frequencies_2k]
temp_dictionary_2k = [dictionary[temp_keys[i]] for i in range(len(temp_keys))]

preprocessed_corpus_2k = [[word for word in document if word in temp_dictionary_2k] for document in preprocessed_corpus]

print('preprocessed_corpus_2k:')
for i in range(5):
  print(preprocessed_corpus_2k[i])
print('')
print('preprocessed_documents_2k:')
preprocessed_documents_2k = [' '.join(document) for document in preprocessed_corpus_2k]

dictionary_2k = gensim.corpora.Dictionary(preprocessed_corpus_2k)

for i in range(5):
  print(preprocessed_documents_2k[i])

"""## **Bag of Words**"""

counts = CountVectorizer()
bow_data = counts.fit_transform(preprocessed_documents)
bow_data.shape

"""## **Bag of Words 2k**"""

counts_2k = CountVectorizer()
bow_data_2k = counts_2k.fit_transform(preprocessed_documents_2k)
bow_data_2k.shape

"""## **TF-IDF Vectorizer**"""

tfidf_vectorizer = TfidfVectorizer()

tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed_documents)

tfidf_vectors.shape

"""## **TF-IDF Vectorizer 2k**"""

tfidf_vectorizer_2k = TfidfVectorizer()
tfidf_vectors_2k = tfidf_vectorizer_2k.fit_transform(preprocessed_documents_2k)
tfidf_vectors_2k.shape

"""## **LDA with TFIDF**"""

num_topics = len(target_names)
num_topics

lda_tfidf = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit(tfidf_vectors)

lda_tfidf.components_

terms_tfidf = tfidf_vectorizer.get_feature_names()

terms_count = 60

for idx,topic in enumerate(lda_tfidf.components_):
  # print('Topic# ',idx+1)
  abs_topic = abs(topic)
  topic_terms = [[terms_tfidf[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
  topic_terms_sorted = [[terms_tfidf[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
  topic_words = []
  for i in range(terms_count):
    topic_words.append(topic_terms_sorted[i][0])
    dict_word_frequency = {}
    for i in range(terms_count):
      dict_word_frequency[topic_terms_sorted[i][0]] = topic_terms_sorted[i][1]
  wcloud = WordCloud(background_color="white",prefer_horizontal=0.9,contour_color='black',width=1700,height=900)
  wcloud.generate_from_frequencies(dict_word_frequency)
  plt.figure()
  plt.imshow(wcloud)
  plt.axis("off")

"""## **LDA with TFIDF Visualization 2**"""

corpus_temp = gensim.matutils.Sparse2Corpus(tfidf_vectors,documents_columns=False)

num_topics = 20
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus_temp, id2word=id2word, chunksize=chunksize, \
                        alpha='auto', eta='auto', \
                        iterations=iterations, num_topics=num_topics, \
                        passes=passes, eval_every=eval_every)

tfidf_visualization = pyLDAvis.gensim.prepare(model, corpus_temp, dictionary, mds='tsne')

pyLDAvis.display(tfidf_visualization)

"""## **LDA with TFIDF 2k**"""

num_topics_2k = len(target_names)
num_topics_2k

lda_tfidf_2k = LatentDirichletAllocation(n_components=num_topics_2k, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit(tfidf_vectors_2k)

terms_tfidf_2k = tfidf_vectorizer_2k.get_feature_names()

terms_count_2k = 60

for idx,topic in enumerate(lda_tfidf_2k.components_):
  # print('Topic# ',idx+1)
  abs_topic_2k = abs(topic)
  topic_terms_2k = [[terms_tfidf_2k[i],topic[i]] for i in abs_topic_2k.argsort()[:-terms_count_2k-1:-1]]
  topic_terms_sorted_2k = [[terms_tfidf_2k[i],topic[i]] for i in abs_topic_2k.argsort()[:-terms_count_2k-1:-1]]
  topic_words_2k = []
  for i in range(terms_count_2k):
    topic_words_2k.append(topic_terms_sorted_2k[i][0])
    dict_word_frequency_2k = {}
    for i in range(terms_count_2k):
      dict_word_frequency_2k[topic_terms_sorted_2k[i][0]] = topic_terms_sorted_2k[i][1]
  wcloud = WordCloud(background_color="white",prefer_horizontal=0.9,contour_color='black',width=1700,height=900)
  wcloud.generate_from_frequencies(dict_word_frequency_2k)
  plt.figure()
  plt.imshow(wcloud)
  plt.axis("off")

"""## **LDA with TFIDF 2k Visualization 2**"""

corpus_temp = gensim.matutils.Sparse2Corpus(tfidf_vectors_2k,documents_columns=False)

num_topics = 20
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary_2k[0]  # This is only to "load" the dictionary.
id2word = dictionary_2k.id2token

model = LdaModel(corpus=corpus_temp, id2word=id2word, chunksize=chunksize, \
                        alpha='auto', eta='auto', \
                        iterations=iterations, num_topics=num_topics, \
                        passes=passes, eval_every=eval_every)

tfidf_visualization_2k = pyLDAvis.gensim.prepare(model, corpus_temp, dictionary_2k, mds='tsne')

pyLDAvis.display(tfidf_visualization_2k)

"""## **LDA with BOW**"""

lda_bow = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit(bow_data)

terms_bow = counts.get_feature_names()

terms_count = 40

for idx,topic in enumerate(lda_bow.components_):
  abs_topic = abs(topic)
  topic_terms = [[terms_bow[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
  topic_terms_sorted = [[terms_bow[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
  topic_words = []
  for i in range(terms_count):
    topic_words.append(topic_terms_sorted[i][0])
    dict_word_frequency = {}
    for i in range(terms_count):
      dict_word_frequency[topic_terms_sorted[i][0]] = topic_terms_sorted[i][1]
  wcloud = WordCloud(background_color="white",prefer_horizontal=0.9,contour_color='black',width=1700,height=900)
  wcloud.generate_from_frequencies(dict_word_frequency)
  plt.figure()
  plt.imshow(wcloud)
  plt.axis("off")

"""## **LDA with BOW Visualization 2**"""

# set training parameters
corpus_temp = [dictionary.doc2bow(doc) for doc in preprocessed_corpus]

num_topics = 20
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus_temp, id2word=id2word, chunksize=chunksize, \
                        alpha='auto', eta='auto', \
                        iterations=iterations, num_topics=num_topics, \
                        passes=passes, eval_every=eval_every)

bow_visualization = pyLDAvis.gensim.prepare(model, corpus_temp, dictionary)

pyLDAvis.display(bow_visualization)

"""## **LDA with BOW 2k**"""

lda_bow_2k = LatentDirichletAllocation(n_components=num_topics_2k, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit(bow_data_2k)

terms_bow_2k = counts_2k.get_feature_names()

terms_count_2k = 40

for idx,topic in enumerate(lda_bow_2k.components_):
  abs_topic_2k = abs(topic)
  topic_terms_2k = [[terms_bow_2k[i],topic[i]] for i in abs_topic_2k.argsort()[:-terms_count_2k-1:-1]]
  topic_terms_sorted_2k = [[terms_bow_2k[i],topic[i]] for i in abs_topic_2k.argsort()[:-terms_count_2k-1:-1]]
  topic_words_2k = []
  for i in range(terms_count_2k):
    topic_words_2k.append(topic_terms_sorted_2k[i][0])
    dict_word_frequency_2k = {}
    for i in range(terms_count_2k):
      dict_word_frequency_2k[topic_terms_sorted_2k[i][0]] = topic_terms_sorted_2k[i][1]
  wcloud = WordCloud(background_color="white",prefer_horizontal=0.9,contour_color='black',width=1700,height=900)
  wcloud.generate_from_frequencies(dict_word_frequency_2k)
  plt.figure()
  plt.imshow(wcloud)
  plt.axis("off")

"""## **LDA with BOW 2k Visualization 2**"""

# set training parameters
corpus_temp_2k = [dictionary_2k.doc2bow(doc) for doc in preprocessed_corpus_2k]

num_topics = 20
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus_temp_2k, id2word=id2word, chunksize=chunksize, \
                        alpha='auto', eta='auto', \
                        iterations=iterations, num_topics=num_topics, \
                        passes=passes, eval_every=eval_every)

bow_visualization_2k = pyLDAvis.gensim.prepare(model, corpus_temp_2k, dictionary)

pyLDAvis.display(bow_visualization_2k)

"""## **Doc2Vec**"""

documents_gensim = [TaggedDocument(token, [i]) for i, token in enumerate(preprocessed_corpus)] 
doc_to_vec_model = gensim.models.doc2vec.Doc2Vec(epochs=40, window=2)
doc_to_vec_model.build_vocab(documents_gensim)
doc_to_vec_model.train(documents_gensim, total_examples=doc_to_vec_model.corpus_count, epochs=doc_to_vec_model.epochs)

documents_gensim_2k = [TaggedDocument(token, [i]) for i, token in enumerate(preprocessed_corpus_2k)] 
doc_to_vec_model_2k = gensim.models.doc2vec.Doc2Vec(epochs=40, window=2)
doc_to_vec_model_2k.build_vocab(documents_gensim_2k)
doc_to_vec_model_2k.train(documents_gensim_2k, total_examples=doc_to_vec_model_2k.corpus_count, epochs=doc_to_vec_model_2k.epochs)

#@title
X = doc_to_vec_model.docvecs.vectors_docs 
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

color_values = ['#d3d3d3','#2f4f4f','#2e8b57','#191970','#8b0000',
               '#808000','#ff0000','#ff8c00','#ffd700','#c71585',
               '#7fff00','#00fa9a','#e9967a','#00ffff','#00bfff',
               '#0000ff','#ff00ff','#f0e68c','#dda0dd','#7b68ee']

color_list = [color_values[targets[i]] for i in range(len(targets))]

ax = plt.axes()
ax.scatter(x=X_tsne[:,0],y=X_tsne[:,1],c=color_list)

words = list(doc_to_vec_model.wv.vocab)
X = doc_to_vec_model[doc_to_vec_model.wv.vocab]
df = pd.DataFrame(X)

X_corr = df.corr()
values,vectors=np.linalg.eig(X_corr)
args = (-values).argsort()
values = vectors[args]
vectors = vectors[:, args]
new_vectors=vectors[:,:2]

neww_X=np.dot(X,new_vectors)

plt.figure(figsize=(13,7))
plt.scatter(neww_X[:,0],neww_X[:,1],linewidths=10,color='blue')
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space",size=20)
vocab=list(doc_to_vec_model.wv.vocab)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))

#@title
X_2k = doc_to_vec_model_2k.docvecs.vectors_docs 
tsne_2k = TSNE(n_components=2)
X_tsne_2k = tsne_2k.fit_transform(X_2k)

color_values = ['#d3d3d3','#2f4f4f','#2e8b57','#191970','#8b0000',
               '#808000','#ff0000','#ff8c00','#ffd700','#c71585',
               '#7fff00','#00fa9a','#e9967a','#00ffff','#00bfff',
               '#0000ff','#ff00ff','#f0e68c','#dda0dd','#7b68ee']

color_list = [color_values[targets[i]] for i in range(len(targets))]
ax = plt.axes()
ax.scatter(x=X_tsne_2k[:,0],y=X_tsne_2k[:,1],c=color_list)

words = list(doc_to_vec_model_2k.wv.vocab)
X_2k = doc_to_vec_model_2k[doc_to_vec_model_2k.wv.vocab]
df_2k = pd.DataFrame(X_2k)

X_corr_2k = df_2k.corr()
values_2k,vectors_2k=np.linalg.eig(X_corr_2k)
args_2k = (-values_2k).argsort()
values_2k = vectors_2k[args_2k]
vectors_2k = vectors_2k[:, args_2k]
new_vectors_2k=vectors_2k[:,:2]

neww_X_2k = np.dot(X_2k,new_vectors_2k)

plt.figure(figsize=(13,7))
plt.scatter(neww_X_2k[:,0],neww_X_2k[:,1],linewidths=10,color='blue')
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.title("Word Embedding Space",size=20)
vocab=list(doc_to_vec_model_2k.wv.vocab)
for i, word in enumerate(vocab):
  plt.annotate(word,xy=(neww_X_2k[i,0],neww_X_2k[i,1]))

"""## **K-Means**"""

nmi_kmeans_stats = []

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=True)

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(pca)
    
    #plt.legend()
    plt.show()

    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1])
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1])
    ax[1].set_title('TSNE Cluster Plot')

"""**Bag-Of-Words K-Means Clustering**

"""

from sklearn.metrics.cluster import normalized_mutual_info_score
km_bow_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
km_bow_fit = km_bow_model.fit(bow_data)

bow_order_centroids = km_bow_model.cluster_centers_.argsort()[:, ::-1]
bow_terms = counts.get_feature_names()

for i in range(num_topics):
    print("Cluster %d:" % i),
    for ind in bow_order_centroids[i, :10]:
        print(' %s' % bow_terms[ind])

bow_cluster_labels = km_bow_model.labels_
nmi_bow_kmeans = normalized_mutual_info_score(targets,bow_cluster_labels)
print(nmi_bow_kmeans)
nmi_kmeans_stats.append(nmi_bow_kmeans)

plot_tsne_pca(bow_data, bow_cluster_labels)

"""**TF-IDF K-Means Clustering**

"""

km_tfidf_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
km_tfidf_fit = km_tfidf_model.fit(tfidf_vectors)

order_centroids = km_tfidf_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()

for i in range(num_topics):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

tfidf_cluster_labels = km_tfidf_model.labels_
plot_tsne_pca(tfidf_vectors, tfidf_cluster_labels)

nmi_tfidf_kmeans = normalized_mutual_info_score(targets,tfidf_cluster_labels)
print(nmi_tfidf_kmeans)
nmi_kmeans_stats.append(nmi_tfidf_kmeans)

"""**Topic Distribution with K-Means Clustering**"""

lda_tfidf_data = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(tfidf_vectors)

lda_topics_kmeans_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
lda_topics_fit = lda_topics_kmeans_model.fit(lda_tfidf_data)

lda_topics_pca = PCA(n_components=2).fit(lda_tfidf_data)

lda_topics_datapoint = lda_topics_pca.transform(lda_tfidf_data)

plt.subplots(1, figsize=(14, 6))
plt.scatter(datapoint[:, 0], datapoint[:, 1])
lda_centroids = lda_topics_kmeans_model.cluster_centers_

nmi_lda_kmeans = normalized_mutual_info_score(targets,lda_topics_fit.labels_)
print(nmi_lda_kmeans)
nmi_kmeans_stats.append(nmi_lda_kmeans)

"""**Doc2Vec K-Means Clustering**"""

d2v_kmeans_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1) 
X_d2v = d2v_kmeans_model.fit(doc_to_vec_model.docvecs.vectors_docs)

labels_d2v = d2v_kmeans_model.labels_
l = d2v_kmeans_model.fit_predict(doc_to_vec_model.docvecs.vectors_docs)

doc2vec_kmeans_data = doc_to_vec_model.docvecs.vectors_docs
d2v_pca = PCA(n_components=2).fit(doc2vec_kmeans_data)

datapoint = d2v_pca.transform(doc_to_vec_model.docvecs.vectors_docs)

plt.subplots(1, figsize=(14, 6))
plt.scatter(datapoint[:, 0], datapoint[:, 1])
centroids = d2v_kmeans_model.cluster_centers_
centroidpoint = d2v_pca.transform(centroids)

nmi_d2v_kmeans = normalized_mutual_info_score(targets,labels_d2v)
print(nmi_d2v_kmeans)
nmi_kmeans_stats.append(nmi_d2v_kmeans)

nmi_table = PrettyTable()
nmi_table.field_names = ["Bag Of Words", "TFIDF", "LDA Topic Distribution", "Doc2Vec"]
nmi_table.add_row(nmi_kmeans_stats)
print(nmi_table)

"""## **K-Means 2k**"""

nmi_kmeans_stats = []

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=True)

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(pca)
    
    plt.show()

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=True)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1])
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1])
    ax[1].set_title('TSNE Cluster Plot')

"""**Bag-Of-Words K-Means Clustering**

"""

from sklearn.metrics.cluster import normalized_mutual_info_score
km_bow_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
km_bow_fit = km_bow_model.fit(bow_data_2k)
bow_terms = counts.get_feature_names()

bow_cluster_labels = km_bow_model.labels_
nmi_bow_kmeans = normalized_mutual_info_score(targets,bow_cluster_labels)
print(nmi_bow_kmeans)
nmi_kmeans_stats.append(nmi_bow_kmeans)

plot_tsne_pca(bow_data, bow_cluster_labels)

"""**TF-IDF K-Means Clustering**

"""

km_tfidf_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
km_tfidf_fit = km_tfidf_model.fit(tfidf_vectors_2k)

order_centroids = km_tfidf_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()

for i in range(num_topics):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

tfidf_cluster_labels = km_tfidf_model.labels_
plot_tsne_pca(tfidf_vectors, tfidf_cluster_labels)

nmi_tfidf_kmeans = normalized_mutual_info_score(targets,tfidf_cluster_labels)
print(nmi_tfidf_kmeans)
nmi_kmeans_stats.append(nmi_tfidf_kmeans)

"""**Topic Distribution with K-Means Clustering**"""

lda_tfidf_data = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(tfidf_vectors_2k)

lda_topics_kmeans_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
lda_topics_fit = lda_topics_kmeans_model.fit(lda_tfidf_data)

lda_topics_pca = PCA(n_components=2).fit(lda_tfidf_data)
lda_topics_datapoint = lda_topics_pca.transform(lda_tfidf_data)

plt.subplots(1, figsize=(14, 6))
plt.scatter(lda_topics_datapoint[:, 0], lda_topics_datapoint[:, 1])

lda_centroids = lda_topics_kmeans_model.cluster_centers_

nmi_lda_kmeans = normalized_mutual_info_score(targets,lda_topics_fit.labels_)
print(nmi_lda_kmeans)
nmi_kmeans_stats.append(nmi_lda_kmeans)

"""**Doc2Vec K-Means Clustering**"""

d2v_kmeans_model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1) 
X_d2v = d2v_kmeans_model.fit(doc_to_vec_model_2k.docvecs.vectors_docs)

labels_d2v = d2v_kmeans_model.labels_
l = d2v_kmeans_model.fit_predict(doc_to_vec_model_2k.docvecs.vectors_docs)

doc2vec_kmeans_data = doc_to_vec_model_2k.docvecs.vectors_docs
d2v_pca = PCA(n_components=2).fit(doc2vec_kmeans_data)

datapoint = d2v_pca.transform(doc_to_vec_model_2k.docvecs.vectors_docs)

plt.subplots(1, figsize=(14, 6))
plt.scatter(datapoint[:, 0], datapoint[:, 1])
centroids = d2v_kmeans_model.cluster_centers_
centroidpoint = d2v_pca.transform(centroids)

nmi_d2v_kmeans = normalized_mutual_info_score(targets,labels_d2v)
print(nmi_d2v_kmeans)
nmi_kmeans_stats.append(nmi_d2v_kmeans)

nmi_table = PrettyTable()
nmi_table.field_names = ["Bag Of Words", "TFIDF", "LDA Topic Distribution", "Doc2Vec"]
nmi_table.add_row(nmi_kmeans_stats)
print(nmi_table)

"""# **Part 7 : KNN**"""

#split into train and test sets
corporus_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
corporus_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

preprocessed_corpus_train, dictionary_train, preprocessed_documents_train = preprocess_data(corporus_train)
preprocessed_corpus_test, dictionary_test, preprocessed_documents_test = preprocess_data(corporus_test)

print(len(preprocessed_corpus_train))
print(len(preprocessed_corpus_test))

"""## **TF-IDF on BOW representation**"""

count_vectorizer_train_test = CountVectorizer()
count_vectors_train = count_vectorizer_train_test.fit_transform(preprocessed_documents_train)
count_vectors_test = count_vectorizer_train_test.transform(preprocessed_documents_test)

count_vectors_train.shape

count_vectors_test.shape

knn_classifier_bow =  KNeighborsClassifier(n_neighbors=5)
knn_classifier_bow.fit(count_vectors_train, corporus_train.target)
y_pred_bow = knn_classifier_bow.predict(count_vectors_test)

print('Predicted Class Labels = ', y_pred_bow)

"""**Performing Grid Search to find optimal K value for BOW**"""

k_values_range = range(1, 10)
weight_options = ['uniform', 'distance']
parameter_grid_bow = dict(n_neighbors=k_values_range, weights=weight_options)
grid_knn_bow = GridSearchCV(knn_classifier_bow, parameter_grid_bow, cv=10, scoring='accuracy')
grid_knn_bow.fit(count_vectors_train, corporus_train.target)

print(grid_knn_bow.best_score_)
print(grid_knn_bow.best_params_)
print(grid_knn_bow.best_estimator_)

"""**Running KNN on optimal K for BOW**"""

knn_classifier_optimal_k_bow =  KNeighborsClassifier(n_neighbors=grid_knn_bow.best_params_['n_neighbors'])
knn_classifier_optimal_k_bow.fit(count_vectors_train, corporus_train.target)
y_pred_optimal_k_bow = knn_classifier_optimal_k_bow.predict(count_vectors_test)

print('Predicted Class Labels = ', y_pred_optimal_k_bow)

"""## **KNN on TF-IDF representation**"""

tfidf_vectorizer_train_test = TfidfVectorizer()
tfidf_vectors_train = tfidf_vectorizer_train_test.fit_transform(preprocessed_documents_train)
tfidf_vectors_test = tfidf_vectorizer_train_test.transform(preprocessed_documents_test)

tfidf_vectors_train.shape

tfidf_vectors_test.shape

knn_classifier_tfidf =  KNeighborsClassifier(n_neighbors=5)
knn_classifier_tfidf.fit(tfidf_vectors_train, corporus_train.target)
y_pred_tfidf = knn_classifier_tfidf.predict(tfidf_vectors_test)

print('Predicted Class Labels = ', y_pred_tfidf)

"""**Performing Grid Search to find optimal K value for TF-IDF**"""

k_values_range = range(1, 10)
weight_options = ['uniform', 'distance']
parameter_grid_tfidf = dict(n_neighbors=k_values_range, weights=weight_options)
grid_knn_tfidf = GridSearchCV(knn_classifier_tfidf, parameter_grid_tfidf, cv=10, scoring='accuracy')
grid_knn_tfidf.fit(tfidf_vectors_train, corporus_train.target)

print(grid_knn_tfidf.best_score_)
print(grid_knn_tfidf.best_params_)
print(grid_knn_tfidf.best_estimator_)

"""**Running KNN on optimal K for TF-IDF**"""

knn_classifier_optimal_k_tfidf =  KNeighborsClassifier(n_neighbors=grid_knn_tfidf.best_params_['n_neighbors'])
knn_classifier_optimal_k_tfidf.fit(tfidf_vectors_train, corporus_train.target)
y_pred_optimal_k_tfidf = knn_classifier_optimal_k_tfidf.predict(tfidf_vectors_test)

print('Predicted Class Labels = ', y_pred_optimal_k_tfidf)

"""## **KNN on LDA TF-IDF representation**"""

num_topics = len(corporus.target_names)
lda_tfidf_train = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(tfidf_vectors_train)
lda_tfidf_test = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(tfidf_vectors_test)

lda_tfidf_train.shape

lda_tfidf_test.shape

knn_classifier_lda_tfidf =  KNeighborsClassifier(n_neighbors=5)
knn_classifier_lda_tfidf.fit(lda_tfidf_train, corporus_train.target)
y_pred_lda_tfidf = knn_classifier_lda_tfidf.predict(lda_tfidf_test)

print('Predicted Class Labels = ', y_pred_lda_tfidf)

"""**Performing Grid Search to find optimal K value for LDA TF-IDF**"""

k_values_range = range(1, 10)
weight_options = ['uniform', 'distance']
parameter_grid_lda_tfidf = dict(n_neighbors=k_values_range, weights=weight_options)
grid_knn_lda_tfidf = GridSearchCV(knn_classifier_lda_tfidf, parameter_grid_lda_tfidf, cv=10, scoring='accuracy')
grid_knn_lda_tfidf.fit(lda_tfidf_train, corporus_train.target)

print(grid_knn_lda_tfidf.best_score_)
print(grid_knn_lda_tfidf.best_params_)
print(grid_knn_lda_tfidf.best_estimator_)

"""**Running KNN on optimal K for LDA TF-IDF**"""

knn_classifier_optimal_k_lda_tfidf =  KNeighborsClassifier(n_neighbors=grid_knn_lda_tfidf.best_params_['n_neighbors'])
knn_classifier_optimal_k_lda_tfidf.fit(lda_tfidf_train, corporus_train.target)
y_pred_optimal_k_lda_tfidf = knn_classifier_optimal_k_lda_tfidf.predict(lda_tfidf_test)

print('Predicted Class Labels = ', y_pred_optimal_k_lda_tfidf)

"""## **KNN on LDA BOW representation**"""

num_topics = len(corporus.target_names)
lda_bow_train = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(count_vectors_train)
lda_bow_test = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50, random_state=0).fit_transform(count_vectors_test)

lda_bow_train.shape

lda_bow_test.shape

knn_classifier_lda_bow =  KNeighborsClassifier(n_neighbors=5)
knn_classifier_lda_bow.fit(lda_bow_train, corporus_train.target)
y_pred_lda_bow = knn_classifier_lda_bow.predict(lda_bow_test)

print('Predicted Class Labels = ', y_pred_lda_bow)

"""**Performing Grid Search to find optimal K value for LDA BOW**"""

k_values_range = range(1, 10)
weight_options = ['uniform', 'distance']
parameter_grid_lda_bow = dict(n_neighbors=k_values_range, weights=weight_options)
grid_knn_lda_bow = GridSearchCV(knn_classifier_lda_bow, parameter_grid_lda_bow, cv=10, scoring='accuracy')
grid_knn_lda_bow.fit(lda_bow_train, corporus_train.target)

print(grid_knn_lda_bow.best_score_)
print(grid_knn_lda_bow.best_params_)
print(grid_knn_lda_bow.best_estimator_)

"""**Running KNN on optimal K for LDA TF-IDF**"""

knn_classifier_optimal_k_lda_bow =  KNeighborsClassifier(n_neighbors=grid_knn_lda_bow.best_params_['n_neighbors'])
knn_classifier_optimal_k_lda_bow.fit(lda_bow_train, corporus_train.target)
y_pred_optimal_k_lda_bow = knn_classifier_optimal_k_lda_bow.predict(lda_bow_test)

print('Predicted Class Labels = ', y_pred_optimal_k_lda_bow)

"""## **KNN on Doc2Vec representation**"""

documents_gensim_train = [TaggedDocument(token, [i]) for i, token in enumerate(preprocessed_corpus_train)] 
doc_to_vec_model_knn_train = gensim.models.doc2vec.Doc2Vec(epochs=40, window=2)
doc_to_vec_model_knn_train.build_vocab(documents_gensim_train)
doc_to_vec_model_knn_train.train(documents_gensim_train, total_examples=doc_to_vec_model_knn_train.corpus_count, epochs=doc_to_vec_model_knn_train.epochs)

doc_to_vec_model_knn_train.docvecs.vectors_docs.shape

documents_gensim_test = [TaggedDocument(token, [i]) for i, token in enumerate(preprocessed_corpus_test)] 
doc_to_vec_model_knn_test = gensim.models.doc2vec.Doc2Vec(epochs=40, window=2)
doc_to_vec_model_knn_test.build_vocab(documents_gensim_test)
doc_to_vec_model_knn_test.train(documents_gensim_test, total_examples=doc_to_vec_model_knn_test.corpus_count, epochs=doc_to_vec_model_knn_test.epochs)

doc_to_vec_model_knn_test.docvecs.vectors_docs.shape

knn_classifier_docvec =  KNeighborsClassifier(n_neighbors=5)
knn_classifier_docvec.fit(doc_to_vec_model_knn_train.docvecs.vectors_docs, corporus_train.target)
y_pred_docvec = knn_classifier_docvec.predict(doc_to_vec_model_knn_test.docvecs.vectors_docs)

print('Predicted Class Labels = ', y_pred_docvec)

"""**Performing Grid Search to find optimal K value for Doc2Vec**"""

k_values_range = range(1, 10)
weight_options = ['uniform', 'distance']
parameter_grid_docvec = dict(n_neighbors=k_values_range, weights=weight_options)
grid_knn_docvec = GridSearchCV(knn_classifier_docvec, parameter_grid_docvec, cv=10, scoring='accuracy')
grid_knn_docvec.fit(doc_to_vec_model_knn_train.docvecs.vectors_docs, corporus_train.target)

print(grid_knn_docvec.best_score_)
print(grid_knn_docvec.best_params_)
print(grid_knn_docvec.best_estimator_)

"""**Running KNN on optimal K for Doc2Vec**"""

knn_classifier_optimal_k_docvec =  KNeighborsClassifier(n_neighbors=grid_knn_docvec.best_params_['n_neighbors'])
knn_classifier_optimal_k_docvec.fit(doc_to_vec_model_knn_train.docvecs.vectors_docs, corporus_train.target)
y_pred_optimal_k_docvec = knn_classifier_optimal_k_docvec.predict(doc_to_vec_model_knn_test.docvecs.vectors_docs)

print('Predicted Class Labels = ', y_pred_optimal_k_docvec)

"""**Table for optimal scores for KNN on the representations**"""

knn_score_table = PrettyTable()
knn_score_table.field_names = ["Bag Of Words", "TF-IDF", "LDA with BOW", "LDA with TF-IDF", "Doc2Vec"]
temp = [grid_knn_bow.best_score_, grid_knn_tfidf.best_score_, grid_knn_lda_bow.best_score_, grid_knn_lda_tfidf.best_score_, grid_knn_docvec.best_score_]
knn_score_table.add_row(temp)
print(knn_score_table)

"""**New clustering method to beat highest NMI from K-means**"""

clustering_label_feature_predict_data = []
clustering = SpectralClustering(n_clusters=len(corporus.target_names), assign_labels="discretize", affinity= "nearest_neighbors")
clustering_label_feature_predict_data.append(clustering.fit_predict(doc_to_vec_model.docvecs.vectors_docs))

pred = []
for val in clustering_label_feature_predict_data:
  pred.append(val)

clustering_label_feature_predict_data = np.array(pred)
clustering_label_feature_predict_data

clustering_label_feature_predict_data = clustering_label_feature_predict_data.flatten()

print(normalized_mutual_info_score(corporus.target, clustering_label_feature_predict_data))

percentage = (0.4415207454075793 - 0.3672520969703051)/0.3672520969703051
percentage

print(percentage * 100)
