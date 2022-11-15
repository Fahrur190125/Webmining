#!/usr/bin/env python
# coding: utf-8

# # UTS WEB Mining
# 1. Lakukan analisa clustering dengan menggunakan k-mean clustering pada data twitter denga kunci pencarian " tragedi kanjuruhan"<br>
# 2. Lakukan peringkasan dokumen dari berita online ( link berita bebas) menggunakan metode pagerank

# ## **1. Clustering Tragedi Kanjuruhan**
# Klustering data merupakan salah satu teknik dari Web Mining, yang mana clustering digunakan untuk pengelompokkan data berdasarkan kemiripan pada objek data dan sebaliknya meminimalkan kemiripan terhadap kluster yang lain. Untuk dapat melakukan clustering lakukan proses berikut.

# ### **Praprepocessing Text**
# Proses ini merupakan proses awal sebelum melakukan proses prepocessing text, yaitu proses untuk mendapatkan dataset yang akan digunakan untuk proses prepocessing, yang mana dataset yang akan digunakan diambil dari website dengan melakukan crawling pada website.

# #### Crawling Tweeter
# 
# Crawling merupakan suatu proses pengambilan data dengan menggunakan mesin yang dilakukan secara online. Proses ini dilakukan untuk mengimpor data yang ditemukan kedalam file lokal komputer. Kemudian data yang telah di impor tersebut akan dilakukan tahap prepocessing text. Pada proses crawling kali ini dilakukan crawling data pada twitter dengan menggunakan tools Twint.
# 
# 
# 
# 
# 
# 

# #### Installasi Twint
# Twint merupakan sebuah tools yang digunakan untuk dapat melakukan scraping data dari media sosial yaitu twitter dengan menggunakan bahasa pemrograman python. Twint dapat dijalankan tanpa harus menggunakan API twitter itu sendiri, namun kapasitas scrapingnya dibatasi sebanyak 3200 tweet.
# 
# Twint tidak hanya digunakan untuk mengambil data tweet, twint juga bisa digunakan untuk mengambil data user, follower, retweet, dan sejenisnya. Twint memanfaatkan operator pencarian twitter yang digunakan untuk memilih dan memilah informasi yang sensitif, termasuk email dan nomor telepon di dalamnya.
# 
# Proses installasi Twint dapat dilakukan dengan source code berikut.

# In[1]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# In[2]:


get_ipython().system('pip install nest-asyncio')


# In[3]:


get_ipython().system('pip install aiohttp==3.7.0')


# #### Scraping Data Tweeter
# Setelah proses installasi Twint berhasil selanjutnya lakukan scraping data tweeter. Scraping sendiri merupakan proses pengambilan data dari website. Untuk melakukan proses scraping data dari tweeter, tinggal import twint untuk melakukan scraping data tweeter dengan tweet yang mengandung kata "tragedi kanjuruhan" dengan limit 100 menggunakan source code berikut.

# In[4]:


import nest_asyncio
nest_asyncio.apply() #digunakan sekali untuk mengaktifkan tindakan serentak dalam notebook jupyter.
import twint #untuk import twint
c = twint.Config()
c.Search = 'tragedi kanjuruhan'
c.Lang = "in"
c.Pandas = True
c.Limit = 100
twint.run.Search(c)


# #### Ambil Tweet
# Setelah proses crawling didapatkan data tweeter diatas, pada data tersebut terdapat data yang tidak diperlukan. Untuk melakukan prepocessing hanya memerlukan data tweet dari user, maka dari itu buang data yang tidak diperlukan dan ambil data tweet yang akan digunakan dengan source code berikut. 

# In[5]:


Tweets_dfs = twint.storage.panda.Tweets_df
Tweets_dfs["tweet"]


# #### Upload Data Tweet
# Setelah data tweet di dapatkan, simpan data tweet tersebut dalam bentuk csv, kemudian download dan upload ke github untuk nanti digunakan sebagai dataset dari proses prepocessing text.

# In[6]:


Tweets_dfs["tweet"].to_csv("kanjuruhan.csv",index=False)


# ### Prepocessing Text
# 
# Setelah proses crawling, selanjutnya dilakukan prepocessing text, yaitu sebuah proses mesin yang digunakan untuk menyeleksi data teks agar lebih terstruktur dengan melalui beberapa tahapan-tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming. 
# Sebelum melakukan tahapan-tahapan tersebut, terlebih dahulu kita import data crawling yang diupload ke github tadi dengan menggunakan library pandas pada source code berikut.
# 
# 

# In[7]:


import pandas as pd 

tweets = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/kanjuruhan.csv",index_col=False)
tweets


# Setelah data crawling berhasil di import, selanjutnya lakukan tahapan-tahapan prepocessing seperti berikut.

# #### Case Folding
# Setelah berhassil mengambil dataset, selanjutnya ke proses prepocessing ke tahapan case folding yaitu tahapan pertama untuk melakukan prepocessing text dengan mengubah text menjadi huruf kecil semua dengan menghilangkan juga karakter spesial, angka, tanda baca, spasi serta huruf yang tidak penting.
# 
# 

# ##### Merubah Huruf Kecil Semua
# Tahapan case folding yang pertama yaitu merubah semua huruf menjadi huruf kecil semua menggunakan fungsi lower() dengan source code berikut.

# In[8]:


tweets['tweet'] = tweets['tweet'].str.lower()


tweets['tweet']


# ##### Menghapus Karakter Spesial
# Tahapan case folding selanjutnya ialah menghapus karakter spesial dengan menggunakan library nltk, untuk menggunakan librarynya terlebih dahulu install dengan source code berikut.
# 

# In[9]:


#install library nltk
get_ipython().system('pip install nltk')


# Setelah library nltk terinstall kita import librarynya dan buat sebuah function untuk menghapus karakter spesial tersebut.

# In[10]:


import string 
import re #regex library
# import word_tokenize & FreqDist from NLTK

from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist


def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
tweets['tweet'] = tweets['tweet'].apply(remove_special)
tweets['tweet']


# ##### Menghapus Angka
# Selanjutnya melakukan penghapusan angka, penghapusan angka disini fleksibel, jika angka ingin dijadikan fitur maka penghapusan angka tidak perlu dilakukan. Untuk data tweet ini saya tidak ingin menjadikan angka sebagai fitur, untuk itu dilakukan penghapusan angka dengan function berikut
# 

# In[11]:


#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

tweets['tweet'] = tweets['tweet'].apply(remove_number)
tweets['tweet']


# ##### Menghapus Tanda Baca
# Selanjutnya penghapusan tanda baca yang tidak perlu yang dilakukan dengan function punctuation berikut
# 

# In[12]:


#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

tweets['tweet'] = tweets['tweet'].apply(remove_punctuation)
tweets['tweet']


# ##### Menghapus Spasi
# Selanjutnya melakukan penghapusan spasi dengab menggunakan function berikut
# 

# In[13]:


#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

tweets['tweet'] = tweets['tweet'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

tweets['tweet'] = tweets['tweet'].apply(remove_whitespace_multiple)
tweets['tweet']


# ##### Menghapus Huruf
# Selanjutnya melakukan penghapusan huruf yang tidak bermakna dengan function berikut

# In[14]:


# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

tweets['tweet'] = tweets['tweet'].apply(remove_singl_char)
tweets['tweet']


# #### Tokenizing
# Setelah tahapan case folding selesai, selanjutnya masuk ke tahapan tokenizing yang merupakan tahapan prepocessing yang memecah kalimat dari text menjadi kata agar membedakan antara kata pemisah atau bukan. Untuk melakukan tokenizing dapat menggunakan dengan library nltk dan function berikut.
# 
# 

# In[15]:


import nltk
nltk.download('punkt')
# NLTK word Tokenize 


# In[16]:


# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

tweets['tweet'] = tweets['tweet'].apply(word_tokenize_wrapper)
tweets['tweet']


# #### Filtering(Stopword)
# Tahapan prepocessing selanjutnya ialah filtering atau disebut juga stopword yang merupakan lanjutan dari tahapan tokenizing yang digunakan untuk mengambil kata-kata penting dari hasil tokenizing tersebut dengan menghapus kata hubung yang tidak memiliki makna.
# 
# Proses stopword dapat dilakukan dengan mengimport library stopword dan function berikut untuk melakukan stopword.

# In[17]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[18]:


list_stopwords = stopwords.words('indonesian')

# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

tweets['tweet'] = tweets['tweet'].apply(stopwords_removal)

tweets['tweet']


# #### Stemming
# Tahapan terakhir dari proses prepocessing ialah stemming yang merupakan penghapusan suffix maupun prefix pada text sehingga menjadi kata dasar. Proses ini dapat dilakukan dengan menggunakan library sastrawi dan swifter.

# In[19]:


get_ipython().system('pip install Sastrawi')


# In[20]:


get_ipython().system('pip install swifter')


# In[21]:


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in tweets['tweet']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

tweets['tweet'] = tweets['tweet'].swifter.apply(get_stemmed_term)
tweets['tweet']


# In[ ]:


tweets.to_csv('Prepocessing.csv',index=False)


# ### Term Frequncy(TF)
# Term Frequency(TF) merupakan banyaknya jumlah kemunculan term pada suatu dokumen. Untuk menghitung nilai TF terdapat beberapa cara, cara yang paling sederhana ialah dengan menghitung banyaknya jumlah kemunculan kata dalam 1 dokumen.<br>
# Sedangkan untuk menghitung nilai TF dengan menggunakan mesin dapat menggunakan library sklearn dengan source code berikut.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing.csv',index_col=False)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# #### Matrik VSM(Visual Space Model)
# Sebelum menghitung nilai TF, terlebih dahulu buat matrik vsm untuk menentukan bobot nilai term pada dokumen dengan source code berikut.

# In[ ]:


matrik_vsm = bag.toarray()
#print(matrik_vsm)
matrik_vsm.shape


# In[ ]:


matrik_vsm[0]


# Untuk menampilkan nilai TF yang didapat menggunakan source code berikut

# In[ ]:


a=vectorizer.get_feature_names()


# In[ ]:


print(len(matrik_vsm[:,1]))
#dfb =pd.DataFrame(data=matrik_vsm,index=df,columns=[a])
dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF.to_csv('TF.csv',index=False)
dataTF


# ### Klustering Data
# Clustering adalah suatu kegiatan mengelompokkan dokumen berdasarkan pada karakteristik yang terkandung di dalamnya. Proses analisa clustering pada intinya terdapat dua tahapan :<br>
#  yang pertama mentransformasi document ke dalam bentuk quantitative data, dan<br>
#  yang kedua menganalisa dokumen dalam bentuk quantitative data tersebut dengan metode clustering yang ditentukan.<br>
# Untuk proses tahapan kedua ada berbagai jenis metode clustering yang bisa digunakan. Diantara metode-metode tersebut ialah metode K-Means, mixture modelling atau tulisan-tulisan clustering lainnya.<br>
# 
# Yang umumnya menjadi permasalahan dalam pelaksanaan clustering ini adalah bagaimana cara merepresentasikan dokumen ke dalam bentuk data quantitative. Ada beberapa cara yang umum digunakan, salah satunya adalah Vector Space Model(VSM) yang merepresentasikan dokumen ke dalam bentuk vector dari term yang muncul dalam dokumen yang dianalisa. Salah satu bentuk representasinya adalah term-frequency (TF) vector yang bisa dilambangkan dengan :<br>
# $$dtf = (tf_1, tf_2, . . . , tf_m)$$
# dimana<br>
# $tf_i$ : adalah frekuensi dari term ke-i di dalam suatu dokumen.<br>
# Kemudian selanjutnya untuk menganalisa dokumen yang sudah dalam bentuk quantitative dengan menggunakan metode K-Means dijelaskan seperti berikut.
# 

# #### K-Means Clustering
# K-Means clustering adalah algoritma untuk membagi n pengamatan menjadi k kelompok sedemikian hingga tiap pengamatan termasuk ke dalam kelompok dengan rata-rata terdekat (titik tengah kelompok). Algoritma ini memiliki hubungan yang renggang dengan algoritma KNN, algoritma pemelajaran mesin yang cukup terkenal dan sering disalah artikan dengan K-Means karena kemiripan namanya.<br>
# Algoritme pengklasteran k rata-rata adalah sebagai berikut.<br>
# 1. Pilih k buah titik tengah secara acak.<br>
# 2. Kelompokkan data sehingga terbentuk k buah kelompok dengan titik tengah tiap kelompok merupakan titik tengah yang telah dipilih sebelumnya.<br>
# 3. Perbarui nilai titik tengah tiap kelompok.<br>
# 4. Ulangi langkah 2 dan 3 sampai titik tengah semua kelompok tidak lagi berubah.<br>
# 
# Proses pengklasteran data ke dalam suatu kelompok dapat dilakukan dengan cara menghitung jarak terdekat dari suatu data ke sebuah titik tengah. Perhitungan jarak Minkowski dapat digunakan untuk menghitung jarak antara 2 buah data.
# 
# Pembaruan titik tengah dapat dilakukan dengan rumus berikut:<br>
# 
# $${\displaystyle \mu _{k}={\frac {1}{N_{k}}}\sum _{j=1}^{N_{k}}x_{j}}$$
# dengan $µk$ adalah titik tengah kelompok ke-k, $Nk$ adalah banyak data dalam kelompok ke-k, dan $xj$ adalah data ke-j dalam kelompok ke-k.<br>
# Untuk melakukan clustering dengan menggunakan mesin dapat menggunakan library sklearn seperti berikut.

# In[ ]:


from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=3)
Kmeans = Kmeans.fit(dataTF)
pred = Kmeans.predict(dataTF)
centroids = Kmeans.cluster_centers_


# In[ ]:


Kmeans.labels_


# #### Hasil Clustering
# Hasil kluster dengan menggunakan metode K-Means ialah sebagai berikut.

# In[ ]:


dataTF['Cluster_Id'] = Kmeans.labels_
dataTF


# Jumlah dari masing-masing kluster dengan 3 kluster sebagai berikut.

# In[ ]:


import numpy as np
unique, counts = np.unique(Kmeans.labels_, return_counts=True)
dict_data = dict(zip(unique, counts))
dict_data


# ### Kesimpulan
# Berdasar dari proses klustering data dengan menggunakan 3 kluster dari 100 data diperoleh 79 data berkluster dengan id = 0, dan 15 data berkluster dengan id = 1, serta 6 data berkluster dengan id = 2.

# ## **2. Ringkasan Berita dengan PageRank**

# ### Mengambil Dokumen
# Langkah awal untuk melakukan ekstraksi ringkasan dokumen ialah dengan mengambil dokumen tersebut dengan mengcrawling data dokumen dengan menggunakan scrapy & crochet seperti berikut.

# In[ ]:


get_ipython().run_line_magic('cd', '/content')


# In[ ]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# In[ ]:


import scrapy
from scrapy.crawler import CrawlerRunner
import re
from crochet import setup, wait_for
setup()

class QuotesToCsv(scrapy.Spider):
    name = "MJKQuotesToCsv"
    start_urls = [
        'https://tekno.tempo.co/read/1645474/makin-canggih-berikut-4-kelebihan-windows-11'
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


# In[ ]:


run_spider()


# ### Membaca Dokumen
# Setelah tahapan mengambil dokumen selesai, selanjutnya membaca dokumen yang sudah didapatkan. Untuk membaca dokumen terlebih dahulu kita convert file csv kedalam bentuk pdf dengan menggunakan library pdfkit. Untuk itu install library pdfkit terlebih dahulu seperti berikut. 

# In[ ]:


get_ipython().system('pip install pdfkit')

get_ipython().system('wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb')

get_ipython().system('cp wkhtmltox_0.12.6-1.bionic_amd64.deb /usr/bin')

get_ipython().system('sudo apt install /usr/bin/wkhtmltox_0.12.6-1.bionic_amd64.deb')


# #### Convert File CSV ke PDF
# Setelah librari pdfkit berhasil diinstal, maka langsung kita import untuk mengconvert file csv yang di dapat ke dalam format pdf menggunakan source code berikut.

# In[ ]:


import pdfkit
import pandas as pd

path_wkhtmltopdf = "/content/wkhtmltox_0.12.6-1.bionic_amd64.deb"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

data = pd.read_csv('news.csv',index_col=False)
html_string = data.to_html()
pdfkit.from_string(html_string, "Dokumen.pdf")


# Setelah berhasil diconvert selanjutnya baca dokumen yang sudah diconvert tersebut dengan library PyPDF2 dan docx2txt, untuk itu kita install library tersebut terlebih dahulu dengan source code berikut.

# In[ ]:


get_ipython().system('pip install PyPDF2')
get_ipython().system('pip install docx2txt')


# #### Baca Dokumen
# Setelah berhasil diinstal selanjutnya kita import library tersebut untuk membaca dokumen yang sudah convert ke bentuk pdf dengan source code berikut.

# In[ ]:


import numpy as np
import PyPDF2
import docx2txt
import sys


# Setelah diimport kita panggil file dokumen tersebut.

# In[ ]:


name = input('Masukkan nama file: ') 
print('Anda telah memanggil dokument  {}'.format(name))


# Setelah itu baca file dokumen tersebut dengan source code berikut.

# In[ ]:


pdfFileObj = open(name, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# ### Memecah Dokumen
# Setelah berhasil membaca dokumen, selanjutnya pecah dokumen sehingga terdiri dari kalimat dan kata-kata dengan menggunakan library nltk. Maka dari itu terlebih dahulu import librarynya seperti berikut.

# In[ ]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# #### Memecah Kalimat
# Setelah library yang dibutuhkan sudah di import selanjutnya pecah dokumen dalam beberapa kalimat dengan menggunakan function berikut.

# In[ ]:


def tokenize(document):
    # Kita memecahnya menggunakan  PunktSentenceTokenizer
    # 
    doc_tokenizer = PunktSentenceTokenizer()
    
    # metode tokenize() memanggil dokument kita
    # sebagai input dan menghasilkan daftar kalimat dalam dokumen
    
    # sentences_list adalah daftar masing masing kalimat dari dokumen yang ada.
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


# In[ ]:


sentences_list = tokenize(document)
print ("Banyaknya kalimat = ", (len(sentences_list)),'kalimat')


# In[ ]:


n = 1
for i in sentences_list:
  print('----------------------------------------------------------------------------------------------------------------------')
  print('Kalimat',n)
  print('----------------------------------------------------------------------------------------------------------------------')
  print(i)
  n = n+1
print('----------------------------------------------------------------------------------------------------------------------')


# #### Memecah Kata
# Setelah dokumen terpecah menjadi beberapa kalimat, selanjutnya kita pecah lagi menjadi kata dengan library sklearn seperti berikut.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)
print ("Banyaknya kosa kata = ", len((cv.get_feature_names_out())),'kosa kata')


# In[ ]:


print ("kosa kata = ", (cv.get_feature_names_out()))


# ### Membuat Matrik TF-IDF
# Setelah memecah dokumen menjadi beberapa kalimat dan kata, selanjutnya buat sebuah matrik VSM untuk membuat TF-IDF seperti berikut.

# In[ ]:


print(cv_matrix)


# In[ ]:


normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())


# ### Membuat Graph
# Setelah matrik TF-IDF terbentuk, selanjutnya buat graph berdasarkan dari matrik tersebut dengan library networkx seperti berikut.

# In[ ]:


import networkx as nx


# In[ ]:


print(normal_matrix.T.toarray)
res_graph = normal_matrix * normal_matrix.T


# In[ ]:


nx_graph = nx.from_scipy_sparse_matrix(res_graph)


# In[ ]:


nx.draw_circular(nx_graph)


# In[ ]:


print('Banyaknya sisi {}'.format(nx_graph.number_of_edges()))


# In[ ]:


normal_matrix.shape


# ### Menghitung PageRank
# Setelah terbentuk graph, selanjutnya hitung nilai pagerank dari masing-masing kalimat dengan source code berikut.

# In[ ]:


ranks = nx.pagerank(nx_graph)


# In[ ]:


n = 1
rangking = []
for i in ranks:
  m = ranks[i],'Kalimat ke',n
  rangking.append(m)
  print('Kalimat',n,':',ranks[i])
  n = n+1


# Setelah nilai pagerank didapatkan, selanjutnya kita rangking nilai pagerank tersebut dari nilai yang paling tinggi seperti berikut.

# In[ ]:


rangking.sort(reverse=True)
rangking


# ### Memilih Kalimat
# Setelah didapatkan kalimat yang memiliki nilai pagerank tertinggi, selanjutnya pilih kalimat yang memiliki nilai pagerank tertinggi, dari data dapat dilihat bahwa kalimat ke-12,8,23,3,9 dan seterusnya memiliki nilai pagerank dari yang paling tinggi hingga rendah.

# In[ ]:


print(sentences_list[11])
print(sentences_list[7])
print(sentences_list[22])
print(sentences_list[2])
print(sentences_list[8])


# ### Kesimpulan
# Berdasar dari tahapan-tahapan yang dilakukan dapat disimpulkan bahwa ringkasan atau simpulan dokumen yang didapat ialah "Windows 11 akan secara otomatis mendeteksi monitor yang didukung HDR dan meningkatkan warna game yang dibuat di DirectX 11 atau lebih tinggi dengan peningkatan jangkauan dinamis. Apakah Anda memilih untuk tetap menggunakan Windows 10 atau membuat lompatan ke Windows 11, sepertinya
# waktu yang tepat untuk meningkatkan ke NVMe SSD untuk melihat manfaat DirectStorage. Namun akan lebih baik jika Anda memiliki lebih dari persyaratan minimum untuk membuat pengalaman bermain game yang lebih baik. Dilansir dari kingston.com, berikut 4 kelebihan Windows 11: Secara harfiah, DirectStorage adalah pembaruan yang mengubah permainan, teknologi ini memungkinkan NVMe SSD untuk mentransfer data permainan langsung ke kartu grafis, melewati kemacetan CPU dan memberikan kecepatan tinggi untuk rendering, tanpa waktu muat yang lama. Dikombinasikan dengan peningkatan memori dan peningkatan kecepatan dan kapasitas penyimpanan perangkat Anda, peningkatan ini dapat menawarkan peluang untuk meningkatkan kinerja dan dengan demikian pengalaman bermain game Anda secara keseluruhan.." Ringkasan tersebut diperoleh dari 4 data kalimat yang memiliki nilai pagerank tertinggi.
