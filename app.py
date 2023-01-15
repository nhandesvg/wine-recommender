import streamlit as st
import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import string
from operator import itemgetter
from collections import Counter, OrderedDict


import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


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
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from dateutil import parser

import json
from PIL import Image

import os
import re
import string
from operator import itemgetter
from collections import Counter, OrderedDict

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import process


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from dateutil.parser import parse
import difflib

from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()




wine_df=pd.read_csv("datasets/preprocessed_wine_df_10_01_23.csv", index_col="Unnamed: 0")
wine_df.head()

list_country = wine_df["country"].unique()
print(list_country)

country_option = st.selectbox(
    'Please select a country',
    list_country)

st.write('Your selection for country is ', country_option)

years = ["[2000-2005]", "[2006-2010]", "[2011-2015]", "[2016-2023]"]
year_option = st.selectbox(
    'Please select a year',
    years)

st.write('Your selection for year is ', year_option)

wine_df["price"].min() #4
wine_df["price"].max() #3300

#kontrol edilmeli
prices = ["[4-50]", "[51-100]", "[101-500]", "[500-1000]","[1001-4000]"]
price_option = st.selectbox(
    'Please select a price',
    prices)

st.write('Your selection for price is ', price_option)

list_title = wine_df["title"].head(25)
#print(list_title)

title_option = st.selectbox(
    'Please select a wine name, then enjoy your recommended wine ',
    list_title)

st.write('Your selection for title is ', title_option)


stop_words = set(stopwords.words("english"))
stop_words.update(["drink", "now", "wine", "flavor", "flavors"])
#Veri incelendiğinde bu sık tekrar eden kelimelerin anlamlı sonuclar vermediği görüldü, stopwords kümesine dahil edildi.

wine_df["description"] = wine_df["description"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
wine_df["description"].head()

#2.kez kontrol edildi, daha önce hazır hale getirilen dfte stopwordse rastlandıgı için.
#Bu nedenle burada bir daha stopwordü kaldırma işlemi uygulandı.

#MODEL 1:
#Şarabın ismi girildiğinde, descriptiona benzerliğine göre ilk 3 şarap öneren fonk:

#def get_matching_name(input_user, df):
 #   options = df['title'].tolist()
  #  best_match = process.extractOne(input_user, options)[0]
   # return best_match


#def recom_by_title (na, df, number=5):
 #   def intersection(lst1, lst2):
  #      return list(set(lst1) & set(lst2))

   # def find_rec(name):
    #    x = df[df['title'] == name]['description'].values[0]
     #   return [[len(intersection(x, d)), t] for d, t in zip(df['description'], df['title'])]

    #na = get_matching_name(na, df)
    #recommendations = find_rec(na)
    #recommendations.sort(reverse=True)
    #recommendations = recommendations[:number]

    #recommendations.sort(key=lambda x: x[0], reverse=True)
    #recommendations = recommendations[:10]
    #return pd.DataFrame(recommendations)

    #print(f"Recommended top {number} wines with similar tastes are:- ")
    #for _, title in recommendations:
    #    print(title)


#wine_df["title"].head()

#recommend = recom_by_title(title_option, wine_df)
# st.button('Recommend my Wine')

#st.write(recom_by_title(title_option, wine_df))


#if st.button('Recommend my Wine'):
 #   st.write(recom_by_title(title_option, wine_df))





#### kmeans cluster


df_new = wine_df.copy()


replace_by_space = re.compile('[/(){}\[\]\|@,;]')
bad_symb_re = re.compile('[^0-9a-z #+_]')
remove_num = re.compile('[\d+]')
stop_words = set(stopwords.words('english'))
stop_words.update(["drink", "now", "wine", "flavor", "flavors"])


def prepare_text(text):
    text = text.lower()

    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = replace_by_space.sub(' ', text)

    # Remove white space
    text = remove_num.sub('', text)

    #  delete symbols which are in BAD_SYMBOLS_RE from text
    text = bad_symb_re.sub('', text)

    # delete stopwords from text
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Stemming the words
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text


df_new["description"] = df_new["description"].apply(prepare_text)

vectorizer = TfidfVectorizer()



X_train = vectorizer.fit_transform(df_new["description"])


columns=vectorizer.get_feature_names_out()


# X_train=X_train.reshape(-1,1)
from yellowbrick.cluster import KElbowVisualizer
#k_clusters = 18

#score = []
#for i in range(1,k_clusters + 1):
 #   kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=5,random_state=0)
  #  kmeans.fit(X_train)
   # score.append(kmeans.inertia_)
#plt.plot(range(1,k_clusters + 1 ),score)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('Score')
#plt.savefig('elbow.png')
#plt.show()






k_clusters = 13

model = KMeans(n_clusters=k_clusters, init='k-means++', n_init=10, max_iter=600, tol=0.000001, random_state=0)
model.fit(X_train)
cluster_labels = model.predict(X_train)

center = model.cluster_centers_


kmeans_labels = pd.DataFrame(cluster_labels)

df_new.insert((df_new.shape[1]), 'clust',kmeans_labels)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

results_dict = {}

for i in range(k_clusters):
    terms_list = []

    for ind in order_centroids[i, :30]:
        terms_list.append(terms[ind])
    results_dict[i] = terms_list
    #results_dict[f'Cluster {i}'] = terms_list

df_new_clusters = pd.DataFrame.from_dict(results_dict)
#df_new_clusters



df_new["title"] = df_new["title"].apply(prepare_text)






#pred2 = model.predict(vectorizer.transform(new_docs))

# pred2 = model.predict(vectorizer.transform(new_docs2))


user_input = [str(st.text_input("Enter key words for wine i.e fresh, red, black, tannin etc. "))]




#new_docs2 = ['melon vanilla acidic tannin fresh fruit crisp']






pred3 = model.predict(vectorizer.transform(user_input))
ww = df_new_clusters[pred3[0]].tolist()


n = len(df_new["description"])


df=pd.read_csv("datasets/winemag-data-130k-v2.csv")

desc = df.reset_index(drop=True)
seq = difflib.SequenceMatcher()
def ratio(ww,pred,df):    
    title=[]
    n = len(df_new["description"])
    desc["description"] = desc["description"].apply(prepare_text)
    for i in range(n):
        seq.set_seqs(ww,desc["description"][i].split())
        if seq.ratio()*100 > 5:
            title.append(desc["title"][i])
    return pd.DataFrame(title[0:10])
            
#ratio(ww,pred3[0],df_new["description"])


if st.button('Recommend my Wine from key words'):
    st.write(ratio(ww,pred3[0],df_new["description"]))
##Tema Seçimi

base="dark"
primaryColor="#b71111"
backgroundColor="#141313"
secondaryBackgroundColor="#600919"

