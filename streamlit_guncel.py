import pandas as pd
from PIL import Image
import streamlit as st
import pickle
import json
# python -m pip install -U scikit-image
# import skimage
# print(skimage.__version__)
# pip install zipfile36
import zipfile
import json
from urllib.request import urlopen

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(page_title="Wine Recommender", page_icon="🍷", layout="centered") #mutlaka en başta olmak zorunda

# image = Image.open('/Users/oykucankusbulan/PycharmProjects/pythonProject4/Miuul Wine Proje/Wine Logo.jpeg')
# st.image(image, width=150)

df=pd.read_csv('https://github.com/nhandesvg/wine-recommender.main/preprocessed_wine_df.csv.zip')
df.head()

list_country = df["country"].unique()

print(list_country)

descriptor_mapping = pd.read_excel('/Users/handesevgi/PycharmProjects/pythonProject2/Proje/descriptor_mapping.xlsx').set_index(
    'raw descriptor')

descriptor_mapping.head(10)

st.title("Wine Recommender 🍷")

country_option = st.selectbox('What is your favorite Country ?', list_country)
st.write('Your selection for Country is ', country_option)

aroma = st.checkbox('Aroma')
non_aroma = st.checkbox('Non_Aroma')

list_aromas = ["fruit", "nutty", "flower", "salinity", "microbial", "spice", "woddy", "earth_organic", "sulfides",
               "brettanomyces", "caramel", "vegetal", "earth_inorganic"]
aroma_option = st.selectbox('What is your aroma selection ?', list_aromas)
st.write('Your selection for aroma is ', aroma_option)

list_non_aromas = ["body", "acid", "alcohol", "sweetnes", "visual", "complexity", "concentration", "finish"]
nonaroma_option = st.selectbox('What is your non_aroma selection ?',list_non_aromas)
st.write('Your selection for aroma is ', nonaroma_option)

years = ["[2000-2005]", "[2006-2010]", "[2011-2015]", "[2016-2023]"]
year_option = st.selectbox('What is your year selection ?',years)
st.write('Your selection for year is ', year_option)

prices = ["[10-50]", "[51--100]", "[100-500]", "[500-]"]
price_option = st.selectbox('What is your price selection ?', prices)
st.write('Your selection for price is ', price_option)

import os
import numpy as np
import string
from operator import itemgetter
from collections import Counter, OrderedDict

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#Ayarlar

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#2) Extracting Wine Descrpitors
#Descriptors are extracted in a few steps, outlined in the cells below: first, all the wine reviews are conslidated into one large corpus. \
# They are then tokenized into sentences.


reviews_list = list(df['description'])
reviews_list = [str(r) for r in reviews_list]
full_corpus = ' '.join(reviews_list)
sentences_tokenized = sent_tokenize(full_corpus)

print(sentences_tokenized[:5])

#Next, the text in each sentence is normalized (tokenize, remove punctuation and remove stopwords).

stop_words = set(stopwords.words('english'))

punctuation_table = str.maketrans({key: None for key in string.punctuation})
sno = SnowballStemmer('english')


def normalize_text(raw_text):
    try:
        word_list = word_tokenize(raw_text)
        normalized_sentence = []
        for w in word_list:
            try:
                w = str(w)
                lower_case_word = str.lower(w)
                stemmed_word = sno.stem(lower_case_word)
                no_punctuation = stemmed_word.translate(punctuation_table)
                if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                    normalized_sentence.append(no_punctuation)
            except:
                continue
        return normalized_sentence
    except:
        return ''

# sentence_sample = sentences_tokenized[:10]
normalized_sentences = []
for s in sentences_tokenized:
    normalized_text = normalize_text(s)
    normalized_sentences.append(normalized_text)

#Not all of the terms we are interested in are single words. Some of the terms are phrases, consisting of two (or more!) words.
# An example of this might be 'high tannin'.
# We can use gensim's Phrases feature to extract all the most relevant bi- and tri-grams from our corpus.

phrases = Phrases(normalized_sentences)
phrases = Phrases(phrases[normalized_sentences])

ngrams = Phraser(phrases)

phrased_sentences = []
for sent in normalized_sentences:
    phrased_sentence = ngrams[sent]
    phrased_sentences.append(phrased_sentence)

full_list_words = [item for sublist in phrased_sentences for item in sublist]

#Next, we will extract the most common words and rank these by how frequently they appear.

word_counts = Counter(full_list_words)
sorted_counts = OrderedDict(word_counts.most_common(5000))
counter_df = pd.DataFrame.from_dict(sorted_counts, orient='index')
# top_5000_words = counter_df.head(5000)
counter_df.to_csv('top_5000_descriptors.csv')

#Now for the most important part: leveraging existing wine theory, the work of others like Bernard Chen, wine descriptor mappings and the UC Davis wine wheel,
# the top 5000 most frequent wine terms were reviewed to (i) determine whether they are a descriptor that can be derived by blind tasting, and
# (ii) whether they are informative (judgments like 'tasty' and 'great' are not considered to be informative). \
# The roughly 1000 descriptors that remain were then mapped onto a normalized descriptor, a category and a class:

#Next, any terms in the corpus that appear in the overview above are mapped onto the 'level_3' (the most specific) layer in the table


def return_mapped_descriptor(word):
    if word in list(descriptor_mapping.index):
        normalized_word = descriptor_mapping['level_3'][word]
        return normalized_word
    else:
        return word


normalized_sentences = []
for sent in phrased_sentences:
    normalized_sentence = []
    for word in sent:
        normalized_word = return_mapped_descriptor(word)
        normalized_sentence.append(str(normalized_word))
    normalized_sentences.append(normalized_sentence)

#3. From Wine Descriptors to Word Embeddings
#We can now proceed to train a Word2Vec model on our normalized and descriptor-mapped corpus.
# This will allow us to get a word embedding for every term in the corpus.
# In order to reduce the overall size of the corpus, we will exclude any terms that appear fewer than 5 times across all the wine reviews.

wine_word2vec_model = Word2Vec(normalized_sentences, vector_size=300, min_count=5, epochs=15)
print(wine_word2vec_model)
#Word2Vec(vocab=16131, vector_size=300, alpha=0.025)
wine_word2vec_model.save('wine_word2vec_model.bin')

#Let's take a closer look at our trained Word2Vec model - what wine descriptors are most similar to the word 'peach'?
wine_word2vec_model.wv.most_similar(positive='peach', topn=10)
#[('stone_fruit', 0.7395599484443665), ('honeydew', 0.6718518137931824), ('pear', 0.6506419777870178), ('tangerine', 0.6326586008071899), ('apple', 0.6243595480918884), ('nectarine', 0.6206014752388), ('kiwi', 0.6166473627090454), ('cataloupe', 0.6125393509864807),
# ('mango', 0.6002585887908936), ('quince', 0.5944412350654602)]

#4) From Word Embeddings to Wine Review Embeddings
# Now, we need to find a way to create a single embedding for each wine review,
# combining all the relevant word embeddings contained within it.
#First, we will extract a set of mapped & normalized descriptors from each wine review.
# Unlike when we were training our Word2Vec model (when in-between words with semantic meaning could help create more accurate embeddings),
# we will remove any descriptors that do not feature in our list of 1000 curated descriptors

wine_reviews = list(df['description'])

def return_descriptor_from_mapping(word):
    if word in list(descriptor_mapping.index):
        descriptor_to_return = descriptor_mapping['level_3'][word]
        return descriptor_to_return


descriptorized_reviews = []
for review in wine_reviews:
    normalized_review = normalize_text(review)
    phrased_review = ngrams[normalized_review]
    descriptors_only = [return_descriptor_from_mapping(word) for word in phrased_review]
    no_nones = [str(d) for d in descriptors_only if d is not None]
    descriptorized_review = ' '.join(no_nones)
    descriptorized_reviews.append(descriptorized_review)

#Not all descriptors are equally distinctive - 'fruity', for instance, is likely to appear very frequently across all wine reviews,
# whereas 'licorice' is much less common. In computing our 'review' embedding, we want to give a higher weighting to terms that are more infrequent
# in relative terms. We will do this by multiplying each word embedding by a TF-IDF (term frequency inverse document frequency) weighting.
# This takes into consideration both the frequency of each term across all reviews, as well as the number of descriptors in each wine review.

vectorizer = TfidfVectorizer()
X = vectorizer.fit(descriptorized_reviews)

dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))

wine_review_vectors = []
for d in descriptorized_reviews:
    descriptor_count = 0
    weighted_review_terms = []
    terms = d.split(' ')
    for term in terms:
        if term in dict_of_tfidf_weightings.keys():
            tfidf_weighting = dict_of_tfidf_weightings[term]
            word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
            weighted_word_vector = tfidf_weighting * word_vector
            weighted_review_terms.append(weighted_word_vector)
            descriptor_count += 1
        else:
            continue
    try:
        review_vector = sum(weighted_review_terms)/len(weighted_review_terms)
    except:
        review_vector = []
    vector_and_count = [terms, review_vector, descriptor_count]
    wine_review_vectors.append(vector_and_count)


df['normalized_descriptors'] = list(map(itemgetter(0), wine_review_vectors))
df['review_vector'] = list(map(itemgetter(1), wine_review_vectors))
df['descriptor_count'] = list(map(itemgetter(2), wine_review_vectors))

df.reset_index(inplace=True)
df.head()


#To summarize, we generate a single vector for each wine review by going through the following steps:

#Normalize words in wine review (remove stopwords, punctuation, stemming)
#Enhance the set of normalized words with phrases (bi-grams and tri-grams)
#Apply mapping of wine terms to each review, keeping only those mapped phrases & words that have been deemed relevant and mapping these to our curated level 3 descriptors
#Retrieve the Word2Vec word embedding for each mapped term in the review
#Weight each word embedding in the wine review with a TF-IDF weighting
#Sum the word embeddings within each wine review to create a single vector representation of the wine review

#5. Exploring our Wine Review Vectors
#5.2 Wine Recommender

wine_reviews_mincount = df.loc[df['descriptor_count'] > 5]
wine_reviews_mincount.reset_index(inplace=True)
wine_reviews_mincount.head()

input_vectors = list(wine_reviews_mincount['review_vector'])
input_vectors_listed = [a.tolist() for a in input_vectors]
input_vectors_listed = [a[0] for a in input_vectors_listed]

knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric='cosine')
model_knn = knn.fit(input_vectors_listed)

#First, let's see what suggestions are returned when we feed this nearest neighbors algorithm a wine we like.
# To demonstrate this, we have chosen a random wine from our dataset - a Pinot Noir from Santa Barbara, California.
name_test = "Nicosia 2013 Vulkà Bianco  (Etna)"

wine_test_vector = wine_reviews_mincount.loc[wine_reviews_mincount['title'] == name_test]['review_vector'].tolist()[0]
distance, indice = model_knn.kneighbors(wine_test_vector, n_neighbors=3)
distance_list = distance[0].tolist()[1:]
indice_list = indice[0].tolist()[1:]

main_wine = wine_reviews_mincount.loc[wine_reviews_mincount['title'] == name_test]

print('Wine to match:', name_test)
print('The original wine has the following descriptors:', list(main_wine['normalized_descriptors'])[0])
print('_________')

#Wine to match: Nicosia 2013 Vulkà Bianco  (Etna)
#The original wine has the following descriptors:
# ['tropical_fruit', 'fruit', 'dry', 'herb', 'apple', 'citrus', 'dry', 'sage', 'brisk']

n = 1
for d, i in zip(distance_list, indice_list):
    wine_name = wine_reviews_mincount['title'][i]
    wine_descriptors = wine_reviews_mincount['normalized_descriptors'][i]
    print('Suggestion', str(n), ':', wine_name, 'with a cosine distance of', "{:.3f}".format(d))
    print('This wine has the following descriptors:', wine_descriptors)
    print('')
    n+=1


#Finally, we can also use our wine recommender model to return a suggestion for a wine based on a list of input descriptors.
# This could be helpful when looking for similarities between wines, or when searching for a particular flavor profile.

def descriptors_to_best_match_wines(list_of_descriptors, number_of_suggestions=10):
    weighted_review_terms = []
    for term in list_of_descriptors:
        if term not in dict_of_tfidf_weightings:
            if term not in descriptor_mapping.index:
                print('choose a different descriptor from', term)
                continue
            else:
                term = descriptor_mapping['normalized'][term]
        tfidf_weighting = dict_of_tfidf_weightings[term]
        word_vector = wine_word2vec_model.wv.get_vector(term).reshape(1, 300)
        weighted_word_vector = tfidf_weighting * word_vector
        weighted_review_terms.append(weighted_word_vector)
    review_vector = sum(weighted_review_terms)

    distance, indice = model_knn.kneighbors(review_vector, n_neighbors=number_of_suggestions + 1)
    distance_list = distance[0].tolist()[1:]
    indice_list = indice[0].tolist()[1:]

    n = 1
    for d, i in zip(distance_list, indice_list):
        wine_name = wine_reviews_mincount['title'][i]
        wine_descriptors = wine_reviews_mincount['normalized_descriptors'][i]
        print('Suggestion', str(n), ':', wine_name, 'with a cosine distance of', "{:.3f}".format(d))
        print('This wine has the following descriptors:', wine_descriptors)
        print('')
        n += 1


if st.button('Recommend my Wine'):
    descriptors = []
    descriptors.append(country_option)
    descriptors.append(aroma_option)
    descriptors.append(nonaroma_option)
    descriptors.append(year_option)
    descriptors.append(price_option)
    print(descriptors)
    descriptors_to_best_match_wines(list_of_descriptors=descriptors, number_of_suggestions=5)


##Tema Seçimi

base="dark"
primaryColor="#b71111"
backgroundColor="#141313"
secondaryBackgroundColor="#600919"

#Video Ekleme

# video_file = open('/Users/oykucankusbulan/PycharmProjects/pythonProject4/Miuul Wine Proje/summer-wine.mp4', 'rb')
# video_bytes = video_file.read()

# st.video(video_bytes)

#### 2. BÖLÜM (SOL TARAF)??? ####

with st.sidebar.header('Wine K Means'):
    st.write("Wine K Means")



