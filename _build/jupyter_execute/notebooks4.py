#!/usr/bin/env python
# coding: utf-8

# # Topik Modelling Dengan Latent Semantic Indexing (LSI) atau Latent Semantic Analysis (LSA) menggunakan Scikit-Learn
# Dalam pembahasan kali ini, kita akan fokus pada Latent Semantic Indexing (LSI) atau Latent Semantic Analysis (LSA) dan melakukan topik modelling menggunakan Scikit-learn.

# ## **Topik Modelling**
# Topik Modelling ialah teknik tanpa pengawasan untuk menemukan tema dokumen yang diberikan. Ini mengekstrak kumpulan kata kunci yang terjadi bersama. Kata kunci yang muncul bersama ini mewakili sebuah topik. Misalnya, saham, pasar, ekuitas, reksa dana akan mewakili topik 'investasi saham'.

# ## **Latent Semantic Indexing (LSI) atau Latent Semantic Analysis (LSA)**
# Latent Semantic Indexing (LSI) atau Latent Semantic Analysis (LSA)  adalah teknik dalam natural language processing , khususnya  distributional semantics , yang menganalisis hubungan antara satu set dokumen dan istilah yang dikandungnya dengan menghasilkan satu set konsep yang terkait dengan dokumen dan istilah. LSA mengasumsikan bahwa kata-kata yang memiliki makna yang dekat akan muncul dalam potongan teks yang serupa (  distributional hypothesis ). Sebuah matriks yang berisi jumlah kata per dokumen (baris mewakili kata-kata unik dan kolom mewakili setiap dokumen) dibangun dari sepotong besar teks dan teknik matematika yang disebut Singular Value Decomposition (SVD) digunakan untuk mengurangi jumlah baris dengan tetap menjaga kesamaan struktur antar kolom. Dokumen kemudian dibandingkan dengan mengambil kosinus sudut antara dua vektor (atau produk titik antara normalisasi dua vektor) yang dibentuk oleh dua kolom. Nilai yang mendekati 1 menunjukkan dokumen yang sangat mirip sedangkan nilai yang mendekati 0 menunjukkan dokumen yang sangat berbeda.<br>
# <center><img src='https://media.geeksforgeeks.org/wp-content/uploads/20210406165951/Screenshot20210406165933.png'></center><center>Gambar LSA</center> Untuk melakukan LSA dapat dilakukan dengan mengikuti tahapan tahapan berikut.

# ## **Mengambil Dokumen**
# Langkah awal untuk melakukan Topik Modelling ialah dengan mengambil dokumen tersebut dengan mengcrawling data dokumen dengan menggunakan scrapy & crochet seperti berikut.

# In[1]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# In[2]:


import scrapy
from scrapy.crawler import CrawlerRunner
import re
from crochet import setup, wait_for
setup()

class QuotesToCsv(scrapy.Spider):
    name = "MJKQuotesToCsv"
    start_urls = [
        'https://tekno.tempo.co/read/1580340/peran-penting-iptekin-terhadap-kemajuan-sebuah-bangsa'
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            '__main__.ExtractFirstLine': 1
        },
        'FEEDS': {
            'news.csv': {
                'format': 'csv',
                'overwrite': True
            }
        }
    }

    def parse(self, response):
        """parse data from urls"""
        for quote in response.css('#isi > p'):
            yield {'news': quote.extract()}


class ExtractFirstLine(object):
    def process_item(self, item, spider):
        """text processing"""
        lines = dict(item)["news"].splitlines()
        first_line = self.__remove_html_tags__(lines[0])

        return {'news': first_line}

    def __remove_html_tags__(self, text):
        """remove html tags from string"""
        html_tags = re.compile('<.*?>')
        return re.sub(html_tags, '', text)

@wait_for(10)
def run_spider():
    """run spider with MJKQuotesToCsv"""
    crawler = CrawlerRunner()
    d = crawler.crawl(QuotesToCsv)
    return d


# In[3]:


run_spider()


# ## **Meload Dokumen**
# Setelah tahapan mengambil dokumen selesai, selanjutnya meload dokumen yang sudah didapatkan. Untuk dapat meload dokumen kita gunakan library os dan pandas seperti berikut.

# In[4]:


import os
import pandas as pd

# Load Dataset
documents_list = []
with open( os.path.join("news.csv") ,"r") as fin:
    for line in fin.readlines():
        text = line.strip()
        documents_list.append(text)


# ## **Membuat Fitur TF-IDF**
# Setelah berhasil meload dokumen langkah selanjutnya ialah mengenerate fitur TF-IDF pada dokumen. Pada proses ini juga dilakukan operasi prepocessing, yaitu case folding, stopword, dan tokenizing. Untuk melakukan proses ini dengan menggunakan RegexpTokenizer dari library nltk seperti source code berikut.

# In[5]:


from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Vectorize document using TF-IDF
tfidf = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize)

# Fit and Transform the documents
train_data = tfidf.fit_transform(documents_list)  
train_data


# ## **Membuat Matrik SVD**
# Matrik SVD adalah teknik dekomposisi matriks yang memfaktorkan matriks dalam produk matriks. Untuk dapat membuat matrik tersebut kita dapat menggunakan TruncatedSVD dari library sklearn seperti berikut.

# In[6]:


from sklearn.decomposition import TruncatedSVD
# Define the number of topics or components
num_components=12

# Create SVD object
lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

# Fit SVD model on data
lsa.fit_transform(train_data)

# Get Singular values and Components 
Sigma = lsa.singular_values_  
V_transpose = lsa.components_.T
V_transpose


# ## **Ekstrak topik dan istilah**
# Setelah membuar matriks SVD, Selanjutnya kita perlu mengekstrak topik dari matriks komponen SVD dengan source code seperti berikut. 

# In[7]:


# Print the topics with their terms
terms = tfidf.get_feature_names()

for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)


# ## **Kesimpulan**
# Hasil yang didapatkan dari topik modelling dengan Latent Semantic Indexing (LSI) atau Latent Semantic Analysis menggunakan library scikit-learn dengan mengambil 12 topik sebagai berikut.<br>
# Topic 1:  ['negara', 'dan', 'yang', 'di', 'dalam']<br>
# Topic 2:  ['elemen', 'ilmu', 'kunci', 'pengetahuan', 'adalah']<br>
# Topic 3:  ['kemudian', 'juga', 'pendekatan', 'disebut', 'berbagai']<br>
# Topic 4:  ['news', 'akan', 'sistem', 'pendekatan', 'kemudian']<br>
# Topic 5:  ['bahan', 'korea', 'sangat', 'selatan', 'taiwan']<br>
# Topic 6:  ['hal', 'baik', 'ini', 'cendekiawan', 'diskursus']<br>
# Topic 7:  ['seringkali', 'banyak', 'sektor', 'cendekiawan', 'diskursus']<br>
# Topic 8:  ['dan', 'juga', 'nies', 'antaranya', 'kelembagaan']<br>
# Topic 9:  ['dokumen', 'bahan', 'korea', 'sangat', 'selatan']<br>
# Topic 10:  ['ekonomi', 'bagian', 'begitu', 'berupaya', 'catch']<br>
# Topic 12:  ['halnya', 'indonesia', 'langsung', 'menerapkan', 'mengabsorbsi']<br>
# Topic 12:  ['amerika', 'austria', 'bagi', 'bahasa', 'belanda']
