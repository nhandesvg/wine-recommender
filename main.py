import streamlit as st
import pandas as pd
import numpy as np

#pip install -r requirements.txt

#Ayarlar

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import warnings
warnings.filterwarnings("ignore")



#cd /Users/havvaserim/Desktop/mywineproject/Streamlit
#streamlit run main.py
import json
from PIL import Image

#st.title("Hello World")
st.set_page_config(page_title="Wine Recommender", page_icon="🍷", layout="centered") #mutlaka en başta olmak zorunda

image = Image.open("/Users/havvaserim/Desktop/mywineproject/Streamlit/winelogo.jpeg")
st.image(image, width=150)

st.title("Wine Recommender 🍷")
#dfi okut
#Daha önceden tokenize ve lemmatize edilmiş, noktalama işaretleri kaldırılmıs ve küçük harfe dönüştrülmüş df okutuldu.

wine_df=pd.read_csv("/Users/havvaserim/PycharmProjects/pythonProject/ProjectforMiuul/preprocessed_wine_df_10_01_23.csv", index_col="Unnamed: 0")
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
print(list_title)

title_option = st.selectbox(
    'Please select a wine name, then enjoy your recommended wine ',
    list_title)

st.write('Your selection for title is ', title_option)

#for model 1

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

stop_words = set(stopwords.words("english"))
stop_words.update(["drink", "now", "wine", "flavor", "flavors"])
#Veri incelendiğinde bu sık tekrar eden kelimelerin anlamlı sonuclar vermediği görüldü, stopwords kümesine dahil edildi.

wine_df["description"] = wine_df["description"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
wine_df["description"].head()

#2.kez kontrol edildi, daha önce hazır hale getirilen dfte stopwordse rastlandıgı için.
#Bu nedenle burada bir daha stopwordü kaldırma işlemi uygulandı.

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

    #print(f"Recommended top {number} wines with similar tastes are:- ")
    #for _, title in recommendations:
        #print(title)
    return pd.DataFrame(recommendations)


#wine_df["title"].head()

#recom_by_title("Nicosia 2013 Vulkà Bianco  (Etna)", wine_df)
# st.button('Recommend my Wine')


if st.button('Recommend my Wine'):
    st.write (recom_by_title(title_option, wine_df))








##Tema Seçimi

base="dark"
primaryColor="#b71111"
backgroundColor="#141313"
secondaryBackgroundColor="#600919"
