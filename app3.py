import streamlit as st
import pandas as pd
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
#from yellowbrick.cluster import KElbowVisualizer
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





st.set_page_config(page_title="Wine Recommender", page_icon="🍷", layout="centered") #mutlaka en başta olmak zorunda

st.title("Wine Recommender ")

#image = Image.open("/Users/havvaserim/Desktop/mywineproject/Streamlit/winelogo.jpeg")
#st.image(image, width=150)

wine_df=pd.read_csv("preprocessed_wine_df_10_01_23.csv.zip", index_col="Unnamed: 0")

tab1, tab2, tab3 = st.tabs(["Unsupervised", "Variety", "Title"])

with tab3:
    st.header("Recommend with Title")
   
    stop_words = set(stopwords.words("english"))
    stop_words.update(["drink", "now", "wine", "flavor", "flavors"])


    wine_df["description"] = wine_df["description"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


    list_title = wine_df["title"]


    title_option = st.selectbox(
    'Please select a wine name, then enjoy your recommended wine ',
    list_title)

#MODEL 1:
#Şarabın ismi girildiğinde, descriptiona benzerliğine göre ilk 3 şarap öneren fonk:

    def get_matching_name(input_user, df):
        options = df['title'].tolist()
        best_match = process.extractOne(input_user, options)[0]
        return best_match


    def recom_by_title (na, df, number=5):
        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        def find_rec(name):
            x = df[df['title'] == name]['description'].values[0]
            return [[len(intersection(x, d)), t] for d, t in zip(df['description'], df['title'])]

        na = get_matching_name(na, df)
        recommendations = find_rec(na)
        recommendations.sort(reverse=True)
        recommendations = recommendations[:number]

        recommendations.sort(key=lambda x: x[0], reverse=True)
        recommendations = recommendations[:10]

        return pd.DataFrame(recommendations)


    if st.button('Recommend my Wine with Title'):
        #print("Recommended with title")
        st.write("Recommended with title",recom_by_title(title_option, wine_df))

with tab2:
    st.header("Recommend with Variety")
    variety_list = wine_df["variety_new"].unique()
    variety_option = st.selectbox('What is your favorite varieties ?', variety_list)
    st.write('Your selection for variety is ', variety_option)
    variety_new = wine_df['variety_new']
    description = wine_df['description']

# Variety ve descriptiondan olusan bir df hazırlandı.
    new_df = pd.DataFrame({"variety_new": variety_new , "description": description})
    

# Üzüm çeşitliliği bazında review sayısı hesaplandı, gözlemlendi.
    review_counts = new_df['variety_new'].value_counts()

# 1'den fazla yorumlanan üzüm çeşitlerinin listesi hazırlandı.
    variety_multi_reviews = review_counts[review_counts > 1].index.tolist()
# Sadece 1 kez yorumlanan üzüm çeşitlerinin listesi hazırlandı.
    variety_single_reviews = review_counts[review_counts == 1].index.tolist()

#TF-IDF Analizi için ön hazırlık:

# CountVectorizer object tanımlandı.
# ngram=range(1,2): unigrams ve bigrams ifadelerin yakalanması için

    variety_description= new_df.set_index("variety_new")
   
    variety_description_2 = pd.DataFrame(columns=["variety_new","description"])

#CountVectorizer object cv olarak tanımlandı.
    cv = CountVectorizer(stop_words="english", ngram_range=(2, 2))

# TfidfTransformer object tanımlandı
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)


    for grape in variety_multi_reviews:
        df = variety_description.loc[[grape]]

    # Belirli bir üzüm çeşidi için yorumda kullanılan kelimeler için word count vektörü hesaplandı.
        word_count_vector = cv.fit_transform(df["description"])

    # IDF değerleri hesaplandı.
        tfidf_transformer.fit(word_count_vector)

    # Yorumlarda kullanılan ilk 100 ortak kelime elde edildi.(başka bir deyişle düşük IDF değerli olanlar)
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(), columns=["idf_weights"])
        df_idf.sort_values(by=["idf_weights"], inplace=True)

    # İlk 100 ortak kelime listeye atıldı.
        common_words = df_idf.iloc[:100].index.tolist()

    # Liste str e çevrildi ve df oluşturuldu.
        common_words_str = ", ".join(elem for elem in common_words)
        new_row = {"variety_new": grape, "description": common_words_str}

    # Variety değişkeni ve ilgili ortak yorumlanan keliemeler yeni bir df e atandı.
        variety_description_2 = variety_description_2.append(new_row, ignore_index=True)

#Checked
    variety_description_2=variety_description_2.set_index("variety_new")
    variety_description_2=variety_description_2.append(variety_description.loc[variety_single_reviews])
    

    tfidf=TfidfVectorizer(stop_words="english", ngram_range=(2,2))
# Descriptiondaki her kelime sayıldı, idf hesaplandı ve idf, tf ile çarpılarak tf-idf matrisi elde edildi.
    tfidf_matrix=tfidf.fit_transform(variety_description_2["description"])
# Matris, satırlarda no. of. description x no.of bigramslardan oluşuyor.

#(707, 61268)
#satırlarda 707 adet üzüm çeşidi, sütunlarda yorumlarda kullanılan 61268 bigrams var.
# Kosinüs benzerliği (cosine similarity) hesaplandı.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# İndeksi üzüm çeşidi (variety) ve elemanı dfteki wine'ın indeksi olan bir seri oluşturuldu.
    variety_description_2=variety_description_2.reset_index()
    indices = pd.Series(variety_description_2.index, index=variety_description_2['variety_new'])

# Varietyi girildiğinde, yorumlarındaki anahtar kelimelerdeki benzerliğine göre 3 tane variety öneren fonk:

    def recommend_by_variety(grape, cosine_sim=cosine_sim):
    # Girilen şarabın indeksini getir.
        idx = indices[grape]

    # Girilen şarap ve diğer şaraplar arasındaki eşleşen benzerlik skorlarını getir.
        sim_scores = list(enumerate(cosine_sim[idx]))

    # Beznerlik skolarına göre şarapları sırala.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # İlk 3 benzerlik skorunu seç.
        sim_scores = sim_scores[1:4]

    # Üzüm çeşidi indekslerini getir.
        wine_idx_list = [i[0] for i in sim_scores]

    # Çıktıyı df e çevir.
        df = pd.DataFrame(columns=["similar wines", "Top 6 common words in wine reviews"])

        for wine_idx in wine_idx_list:

            g_variety = variety_description_2.iloc[wine_idx]["variety_new"]

        # Yorumlardaki ilk 6 ortak kelimeyi getir.
            des = variety_description_2.iloc[wine_idx]["description"]

            if g_variety in variety_multi_reviews:  # 1 den fazla yorumlandıysa
                des_split = des.split(", ")
                key_words_list = des_split[:6]
                key_words_str = ", ".join(key_words_list)

            else:
                key_words_str = des

            new_row = {"similar wines": g_variety, "Top 6 common words in wine reviews": key_words_str}
            df = df.append(new_row, ignore_index=True)

        df.set_index("similar wines")

    # Tüm ortak kelimelerin gösterilmesi için
        pd.set_option('max_colwidth', 500)

        return df

    if st.button('Recommend my Wine with Variety'):
        st.write("Recommended with variety",recommend_by_variety(variety_option))
        

with tab1:
    st.header("Recommend with Unsupervised Learning")

    stemmer = PorterStemmer()

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



    @st.cache(allow_output_mutation=True)
    def get_data():
        return []

    
    form = st.form(key='my_form')
    user_input = form.text_input(label='Enter key words for wine i.e fresh, red, black, tannin etc.')
    submit_button = form.form_submit_button(label='Submit')
    if submit_button:
        get_data().append(user_input)
    
        get_data().append(user_input)
        data = [get_data()[0]]
    
        pred3 = model.predict(vectorizer.transform(data))
        ww = df_new_clusters[pred3[0]].tolist()

        df=pd.read_csv("winemag-data-130k-v2.csv.zip")

        desc = df.reset_index(drop=True)

    

        seq = difflib.SequenceMatcher()


        def ration(ww,df):    
            title=[]
            n = len(desc["description"])
            desc["description"] = desc["description"].apply(prepare_text)
            for i in range(n):
                seq.set_seqs(ww,desc["description"][i].split())
                if seq.ratio()*100 > 10:
                    title.append(desc["title"][i])
            return pd.DataFrame(title[1:11])
        st.write(ration(ww,desc["description"]))
