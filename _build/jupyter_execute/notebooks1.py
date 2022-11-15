#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi & Clustering
# Klasifikasi dan Klustering data merupakan salah satu teknik dari Web Mining, yang mana klasifikasi adalah pemprosesan untuk menemukan sebuah model atau fungsi yang menjelaskan dan mencirikan konsep atau kelas data, untuk kepentingan tertentu. Sedangkan clustering digunakan untuk pengelompokkan data berdasarkan kemiripan pada objek data dan sebaliknya meminimalkan kemiripan terhadap kluster yang lain. Untuk dapat melakukan klasifikasi dan clustering lakukan proses berikut.

# ## **Praprepocessing Text**
# Proses ini merupakan proses awal sebelum melakukan proses prepocessing text, yaitu proses untuk mendapatkan dataset yang akan digunakan untuk proses prepocessing, yang mana dataset yang akan digunakan diambil dari website dengan melakukan crawling pada website.

# ### Crawling Tweeter
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
# Setelah proses installasi Twint berhasil selanjutnya lakukan scraping data tweeter. Scraping sendiri merupakan proses pengambilan data dari website. Untuk melakukan proses scraping data dari tweeter, tinggal import twint untuk melakukan scraping data tweeter dengan tweet yang mengandung kata "#rockygerung" dengan limit 100 menggunakan source code berikut.

# In[4]:


import nest_asyncio
nest_asyncio.apply() #digunakan sekali untuk mengaktifkan tindakan serentak dalam notebook jupyter.
import twint #untuk import twint
c = twint.Config()
c.Search = '#rockygerung'
c.Lang = "in"
c.Pandas = True
c.Limit = 100
twint.run.Search(c)


# #### Ambil Tweet
# Setelah proses crawling didapatkan data tweeter diatas, pada data tersebut terdapat data yang tidak diperlukan. Untuk melakukan prepocessing hanya memerlukan data tweet dari user, maka dari itu buang data yang tidak diperlukan dan ambil data tweet yang akan digunakan dengan source code berikut. 

# In[5]:


Tweets_dfs = twint.storage.panda.Tweets_df
Tweets_dfs["tweet"]


# ### Upload Data Tweet
# Setelah data tweet di dapatkan, simpan data tweet tersebut dalam bentuk csv, kemudian download dan upload ke github untuk nanti digunakan sebagai dataset dari proses prepocessing text.

# In[6]:


Tweets_dfs["tweet"].to_csv("RG.csv",index=False)


# ## **Prepocessing Text** 
# 
# Setelah proses crawling, selanjutnya dilakukan prepocessing text, yaitu sebuah proses mesin yang digunakan untuk menyeleksi data teks agar lebih terstruktur dengan melalui beberapa tahapan-tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming. 
# Sebelum melakukan tahapan-tahapan tersebut, terlebih dahulu kita import data crawling yang diupload ke github tadi dengan menggunakan library pandas pada source code berikut.
# 
# 

# In[7]:


import pandas as pd 

tweets = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/RG.csv",index_col=False)
tweets


# Setelah data crawling berhasil di import, selanjutnya lakukan tahapan-tahapan prepocessing seperti berikut.

# ### Case Folding
# Setelah berhassil mengambil dataset, selanjutnya ke proses prepocessing ke tahapan case folding yaitu tahapan pertama untuk melakukan prepocessing text dengan mengubah text menjadi huruf kecil semua dengan menghilangkan juga karakter spesial, angka, tanda baca, spasi serta huruf yang tidak penting.
# 
# 

# #### Merubah Huruf Kecil Semua
# Tahapan case folding yang pertama yaitu merubah semua huruf menjadi huruf kecil semua menggunakan fungsi lower() dengan source code berikut.

# In[8]:


tweets['tweet'] = tweets['tweet'].str.lower()


tweets['tweet']


# #### Menghapus Karakter Spesial
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


# #### Menghapus Angka
# Selanjutnya melakukan penghapusan angka, penghapusan angka disini fleksibel, jika angka ingin dijadikan fitur maka penghapusan angka tidak perlu dilakukan. Untuk data tweet ini saya tidak ingin menjadikan angka sebagai fitur, untuk itu dilakukan penghapusan angka dengan function berikut
# 

# In[11]:


#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

tweets['tweet'] = tweets['tweet'].apply(remove_number)
tweets['tweet']


# #### Menghapus Tanda Baca
# Selanjutnya penghapusan tanda baca yang tidak perlu yang dilakukan dengan function punctuation berikut
# 

# In[12]:


#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

tweets['tweet'] = tweets['tweet'].apply(remove_punctuation)
tweets['tweet']


# #### Menghapus Spasi
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


# #### Menghapus Huruf
# Selanjutnya melakukan penghapusan huruf yang tidak bermakna dengan function berikut

# In[14]:


# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

tweets['tweet'] = tweets['tweet'].apply(remove_singl_char)
tweets['tweet']


# ### Tokenizing
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


# ### Filtering(Stopword)
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


# ### Stemming
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


# Setelah tahap stemming proses prepocessing sudah selesai, namun pada dataset masih belum memiliki kelas atau label untuk itu akan dilakukan pemberian label atau kelas dengan menggunakan nilai polarity.

# ## **Labelling Dataset**
# Setelah proses prepocesing selesai didapat sebuah dataset yang masih belum memiliki label, untuk itu pada tahapan ini dataset akan diberikan kelas atau label yang sesuai. Akan tetapi tahap pelabelan ini akan memerlukan waktu yang lama jika dilakukan secara manual. Untuk itu pada tahapan ini saya memberikan kelas atau label pada masing-masing data secara otomatis dengan menggunakan nilai polarity.

# ### Nilai Polarity
# Nilai polarity merupakan nilai yang menunjukkan apakah kata tersebut bernilai negatif atau positif ataupun netral. Nilai polarity didapatkan dengan menjumlahkan nilai dari setiap kata dataset yang menunjukkan bahwa kata tersebut bernilai positif atau negatif ataupun netral.<br>
# Didalam satu kalimat atau data,nilai dari kata-kata didalam satu kalimat tersebut akan dijumlah sehingga akan didapatkan nilai atau skor polarity. Nilai atau skor tersebutlah yang akan menentukan kalimat atau data tersebut berkelas positif(pro) atau negatif(kontra) ataupun netral.<br>
# Jika nilai polarity yang didapat lebih dari 0 maka kalimat atau data tersebut diberi label atau kelas pro. Jika nilai polarity yang didapat kurang dari 0 maka kalimat atau data tersebut diberi label atau kelas kontra. Sedangkan jika nilai polarity sama dengan 0 maka kalimat atau data tersebut diberi label netral.

# ### Ambil Nilai Polarity
# Sebelum melakukan pemberian label atau kelas dengan menggunakan nilai polarity, kita ambil nilai polarity dari setiap kata apakah positif atau negatif. Untuk itu saya mengambil nilai polarity dari github yang di dapat dari link github berikut https://github.com/fajri91/InSet
# Nilai lexicon positif dan negatif yang didapat dari github tersebut saya download kemudian saya upload ke github saya dan kemudian saya ambil data lexicon positif dan negatif tersebut dengan source code berikut.

# In[ ]:


positive = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/positive.csv")
positive.to_csv('lexpos.csv',index=False)
negative = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/negative.csv")
negative.to_csv('lexneg.csv',index=False)


# ### Menentukan Kelas/Label dengan Nilai Polarity
# Setelah berhasil mengambil nilai polarity lexicon positif dan negatif selanjutnya kita tentukan kelas dari masing masing data dengan menjumlahkan nilai polarity yang didapat dengan ketentuan jika lebih dari 0 maka memiliki kelas pro, jika kurang dari 0 maka diberi kelas kontra, dan jika sama dengan 0 maka memiliki kelas netral, dengan source code berikut.

# In[ ]:


# Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)
# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('lexpos.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('lexneg.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'pro'
    elif (score < 0):
        polarity = 'kontra'
    else:
        polarity = 'netral'
    return score, polarity


# In[ ]:


# Results from determine sentiment polarity of tweets

results = tweets['tweet'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
tweets['polarity_score'] = results[0]
tweets['label'] = results[1]
print(tweets['label'].value_counts())


# Setelah didapat dataset yang sudah memiliki label selanjutnya kita simpan dengan source code berikut.

# In[ ]:


# Export to csv file
tweets.to_csv('Prepocessing_label.csv',index=False)

tweets


# ## **Term Frequncy(TF)**
# Term Frequency(TF) merupakan banyaknya jumlah kemunculan term pada suatu dokumen. Untuk menghitung nilai TF terdapat beberapa cara, cara yang paling sederhana ialah dengan menghitung banyaknya jumlah kemunculan kata dalam 1 dokumen.<br>
# Sedangkan untuk menghitung nilai TF dengan menggunakan mesin dapat menggunakan library sklearn dengan source code berikut.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing_label.csv',index_col=False)
dataTextPre.drop("polarity_score", axis=1, inplace=True)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# ### Matrik VSM(Visual Space Model)
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


# ### Nilai Term Dokumen
# Setelah didapat nilai matrik vsm, selanjutnya tentukan nilai term pada masing masing dokumen menggunakan source code berikut.

# In[ ]:


datalabel = pd.read_csv('Prepocessing_label.csv',index_col=False)
TF = pd.read_csv('TF.csv',index_col=False)
dataJurnal = pd.concat([TF, datalabel["label"]], axis=1)
dataJurnal


# ### Mengambil Data label
# Setelah didapat nilai term pada masing masing dokumen kita ambil data label pada masing masing dokumen.

# In[ ]:


dataJurnal['label'].unique()


# In[ ]:


dataJurnal.info()


# ### Split Data
# Selanjutnya kita split dataset menjadi data training dan testing dengan source code berikut.

# In[ ]:


### Train test split to avoid overfitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataJurnal.drop(labels=['label'], axis=1),
    dataJurnal['label'],
    test_size=0.15,
    random_state=0)


# #### Data Training

# In[ ]:


X_train


# #### Data Testing

# In[ ]:


X_test


# ## **Information Gain** 
# Metode Information Gain adalah metode yang menggunakan teknik scoring untuk pembobotan sebuah fitur dengan menggunakan maksimal entropy. Fitur yang dipilih adalah fitur dengan nilai Information Gain yang lebih besar atau sama dengan nilai threshold tertentu. Nilai information gain dapat dihitung dengan menggunakan rumus berikut.<br>
# 
# $$Gain(S,A) = Entropy(S) - \sum values(a)\frac{|SV|}{|S|} Entropy(S_{v})$$
# 
# Yang mana :<br>
# Gain(S,A) : nilai Gain dari fitur <br>
# A : fitur<br>
# v : kemungkinan nilai fitur 𝐴<br>
# 𝑉𝑎𝑙𝑢𝑒𝑠(𝐴) : kemungkinan nilai himpunan 𝐴<br>
# 𝑆𝑣 : jumlah contoh nilai dari 𝑣<br>
# 𝑆 : jumlah keseluruhan sampel data<br>
# Entropy(Sv) : 𝐸𝑛𝑡𝑟𝑜𝑝𝑦 contoh nilai v<br>
# 
# Namun sebelum menghitung nilai information gain terlebih dahulu kita harus menghitung nilai entropy dengan rumus berikut.<br>
# 
# $$Entropy(S) = \sum_{i}^{c} -p_{i}\log_{2}p_{i}$$
# 
# Yang mana :<br>
# 𝑐 : akumulasi nilai dari kelas klasfikasi<br>
# 𝑃𝑖 : merupakan akumulasi sampel dari kelas 𝑖.<br>
# Sedangkan untuk menghitung nilai information gain dengan mesin dapat mwnggunakan library mutual information seperti berikut.
# 

# In[ ]:


from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# ### Sorting Information Gain
# Setelah didapat nilai information gainnya, selanjutnya kita dapat mengurutkan nilai information gain dari yang tertinggi hingga yang terendah dengan source code berikut.

# In[ ]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# ### Membuat Grafik Information Gain
# Selanjutnya kita juga dapat membuat grafiknya dengan menggunakan matplotlib seperti berikut.

# In[ ]:


#let's plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(200, 80))


# ### Pilih Fitur Penting
# Selanjutnya kita juga dapat memilih fitur yang penting berdasarkan nilai information gain yang diperoleh, semakin tinggi nilai fitur maka semakin penting fitur tersebut. Disini saya memilih 100 data fitur penting dengan menggunakan library SelectBest.

# In[ ]:


from sklearn.feature_selection import SelectKBest


# In[ ]:


#No we Will select the  top 5 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# ## **Klasifikasi Data**
# Klasifikasi adalah proses penemuan model (atau fungsi) yang
# menggambarkan dan membedakan kelas data atau konsep yang bertujuan agar
# bisa digunakan untuk memprediksi kelas dari objek yang label kelasnya tidak
# diketahui.Klasifikasi data terdiri dari 2 langkah proses. Pertama
# adalah learning (fase training), dimana algoritma klasifikasi dibuat untuk
# menganalisis data training lalu direpresentasikan dalam bentuk rule klasifikasi.
# Proses kedua adalah klasifikasi, dimana data tes digunakan untuk memperkirakan
# akurasi dari rule klasifikasi. Terdapat beberapa metode klasifikasi, diantaranya sebagai berikut.

# ### KNN (K-Nearest Neighbor)
# K-Nearest Neighbor (KNN) merupakan salah satu metode yang digunakan
# dalam menyelesaikan masalah pengklasifikasian. Prinsip KNN yaitu
# mengelompokkan atau mengklasifikasikan suatu data baru yang belum diketahui
# kelasnya berdasarkan jarak data baru itu ke beberapa tetangga (neighbor) terdekat.
# Tetangga terdekat adalah objek latih yang memiliki nilai kemiripan terbesar atau
# ketidakmiripan terkecil dari data lama. Jumlah tetangga terdekat dinyatakan
# dengan k. Nilai k yang terbaik tergantung pada data. 
# Nilai k umumnya ditentukan dalam jumlah ganjil (3, 5, 7) untuk
# menghindari munculnya jumlah jarak yang sama dalam proses pengklasifikasian.
# Apabila terjadi dua atau lebih jumlah kelas yang muncul sama maka nilai k
# menjadi k – 1 (satu tetangga kurang), jika masih ada yang sama lagi maka nilai k
# menjadi k – 2 , begitu seterusnya sampai tidak ditemukan lagi kelas yang sama
# banyak. Banyaknya kelas yang paling banyak dengan jarak terdekat akan menjadi
# kelas dimana data yang dievaluasi berada. Dekat atau jauhnya tetangga (neighbor)
# biasanya dihitung berdasarkan jarak Euclidean (Euclidean Distance). Berikut
# rumus pencarian jarak menggunakan rumus Euclidian :
# 
# $$d_i = \sqrt{\sum_{i=1}^{p}(x_2i-x_1i)^{2}}$$
# 
# dengan:<br>
# $x_1$ = sampel data<br>
# $x_2$ = data uji<br>
# i = variabel data<br>
# $d_i$ = jarak<br>
# p = dimensi data<br>
# 
# Berikut merupakan klasifikasi data dengan metode KNN dengan library scikit learn menggunakan nilai k yang di ubah-ubah.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
K_range = []
K_score = []
for i in range (2,len(y_test)):
  classifier = KNeighborsClassifier(n_neighbors=i) 
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  score = classifier.score(X_test, y_test)
  K_range.append(i)
  K_score.append(score)
  print("Akurasi KNN saat Menggunakan K =",i,":",score)
  #print(classification_report(y_test, y_pred))


# Berikut merupakan grafik nilai akurasi KNN berdasarkan nilai k.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(K_range, K_score)
plt.title('Nilai Accuracy KNN Berdasarkan Nilai K')
plt.xlabel('Nilai K')
plt.ylabel('Nilai Akurasi')
plt.grid(True)
plt.show()


# ### Naive Bayes
# Algoritma Naive Bayes adalah algoritma yang mempelajari probabilitas suatu objek dengan ciri-ciri tertentu yang termasuk dalam kelompok/kelas tertentu. Singkatnya, ini adalah pengklasifikasi probabilistik. Berikut merupakan klasifikasi naive bayes dengan mengunakan library scikit learn.

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
modelnb = GaussianNB()

# Memasukkan data training pada fungsi klasifikasi Naive Bayes
nbtrain = modelnb.fit(X_train, y_train)

# Menentukan hasil prediksi dari x_test
#y_pred = nbtrain.predict(X_test)

print("Akurasi Naive Bayes :",nbtrain.score(X_test, y_test))

#print(classification_report(y_test, y_pred))


# ### SVM(Support Vector Machine)
# Support Vector Machine (SVM) merupakan salah satu metode dalam supervised learning yang biasanya digunakan untuk klasifikasi (seperti Support Vector Classification) dan regresi (Support Vector Regression). Dalam pemodelan klasifikasi, SVM memiliki konsep yang lebih matang dan lebih jelas secara matematis dibandingkan dengan teknik-teknik klasifikasi lainnya. SVM juga dapat mengatasi masalah klasifikasi dan regresi dengan linear maupun non linear. Berikut merupakan klasifikasi SVM dengan mengunakan library scikit learn.

# In[ ]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
svm = svm.SVC() # Linear Kernel

#Train the model using the training sets
svm.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = svm.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Akurasi SVM :",svm.score(X_test, y_test))


# ## **Klustering Data**
# Clustering adalah suatu kegiatan mengelompokkan dokumen berdasarkan pada karakteristik yang terkandung di dalamnya. Proses analisa clustering pada intinya terdapat dua tahapan :<br>
#  yang pertama mentransformasi document ke dalam bentuk quantitative data, dan<br>
#  yang kedua menganalisa dokumen dalam bentuk quantitative data tersebut dengan metode clustering yang ditentukan.<br>
# Untuk proses tahapan kedua ada berbagai jenis metode clustering yang bisa digunakan. Diantara metode-metode tersebut ialah metode K-Means, mixture modelling atau tulisan-tulisan clustering lainnya.<br>
# 
# Yang umumnya menjadi permasalahan dalam pelaksanaan clustering ini adalah bagaimana cara merepresentasikan dokumen ke dalam bentuk data quantitative. Ada beberapa cara yang umum digunakan, salah satunya adalah Vector Space Model(VSM) yang merepresentasikan dokumen ke dalam bentuk vector dari term yang muncul dalam dokumen yang dianalisa. Salah satu bentuk representasinya adalah term-frequency (TF) vector yang bisa dilambangkan dengan :<br>
# 
# $$dtf = (tf_1, tf_2, . . . , tf_m)$$
# 
# dimana<br>
# $tf_i$ : adalah frekuensi dari term ke-i di dalam suatu dokumen.<br>
# Kemudian selanjutnya untuk menganalisa dokumen yang sudah dalam bentuk quantitative dengan menggunakan metode K-Means dijelaskan seperti berikut.
# 

# ### K-Means Clustering
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


# ### Hasil Clustering
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


# ## **Kesimpulan**
# Berdasarkan nilai akurasi yang diperoleh dari 3 metode (KNN, Naive Bayes, dan SVM) yang dilakukan, klasifikasi dengan metode Naive Bayes memiliki nilai akurasi yang lebih baik dibandingkan dengan metode KNN dan SVM. Nilai akurasi yang diperoleh dari metode Naive Bayes sebesar 87%.<br>
# Sedangkan nilai akurasi yang diperoleh dengan menggunakan metode KNN didapat akurasi tertinggi sebesar 66% pada saat nilai k = 7 dan 8 dan akuarasi yang didapat dari metode SVM sebesar 73%. Sehingga dapat disimpulkan bahwa klasifikasi dari dataset yang mengandung kata "#rockygerung" yang diperoleh dari tweeter lebih baik menggunakan metode Naive Bayes dibandingkan dengan metode KNN dan SVM.<br>
# Dengan nilai akurasi yang didapat dari metode naive bayes tersebut, klasifikasi ini sudah bisa dijadikan sebagai acuan untuk menentukan tanggapan user tweeter tentang "#rockygerung" apakah beropini kontra atau pro ataupun netral. Akan tetapi klasifikasi ini masih memerlukan evaluasi atau perbaikan dari tahap prepocessing hingga modelling agar menghasilkan nilai akurasi yang lebih baik.<br>
# Dan dengan proses klustering data dengan menggunakan 3 kluster dari 100 data diperoleh 56 data berkluster dengan id = 0, dan 36 data berkluster dengan id = 1, serta 8 data berkluster dengan id = 2.
