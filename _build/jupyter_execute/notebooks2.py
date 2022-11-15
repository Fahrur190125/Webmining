#!/usr/bin/env python
# coding: utf-8

# # Ekstraksi Ringkasan Dokumen
# Sistem Peringkasan adalah sistem yang digunakan untuk menentukan topik yang sangat penting dari suatu dokumen. Proses peringkasan ini dapat dilakukan dengan melalui tahapan-tahapan berikut.

# ## **Mengambil Dokumen**
# Langkah awal untuk melakukan ekstraksi ringkasan dokumen ialah dengan mengambil dokumen tersebut dengan mengcrawling data dokumen dengan menggunakan scrapy & crochet seperti berikut.

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


# ## **Membaca Dokumen**
# Setelah tahapan mengambil dokumen selesai, selanjutnya membaca dokumen yang sudah didapatkan. Untuk membaca dokumen terlebih dahulu kita convert file csv kedalam bentuk pdf dengan menggunakan library pdfkit. Untuk itu install library pdfkit terlebih dahulu seperti berikut. 

# In[4]:


get_ipython().system('pip install pdfkit')

get_ipython().system('wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb')

get_ipython().system('cp wkhtmltox_0.12.6-1.bionic_amd64.deb /usr/bin')

get_ipython().system('sudo apt install /usr/bin/wkhtmltox_0.12.6-1.bionic_amd64.deb')


# ### Convert File CSV ke PDF
# Setelah librari pdfkit berhasil diinstal, maka langsung kita import untuk mengconvert file csv yang di dapat ke dalam format pdf menggunakan source code berikut.

# In[5]:


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


# ### Baca Dokumen
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


# ## **Memecah Dokumen**
# Setelah berhasil membaca dokumen, selanjutnya pecah dokumen sehingga terdiri dari kalimat dan kata-kata dengan menggunakan library nltk. Maka dari itu terlebih dahulu import librarynya seperti berikut.

# In[ ]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# ### Memecah Kalimat
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


# ### Memecah Kata
# Setelah dokumen terpecah menjadi beberapa kalimat, selanjutnya kita pecah lagi menjadi kata dengan library sklearn seperti berikut.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)
print ("Banyaknya kosa kata = ", len((cv.get_feature_names_out())),'kosa kata')


# In[ ]:


print ("kosa kata = ", (cv.get_feature_names_out()))


# ## **Membuat Matrik TF-IDF**
# Setelah memecah dokumen menjadi beberapa kalimat dan kata, selanjutnya buat sebuah matrik VSM untuk membuat TF-IDF seperti berikut.

# In[ ]:


print(cv_matrix)


# In[ ]:


normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())


# ## **Membuat Graph**
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


# ## **Menghitung PageRank**
# Setelah terbentuk graph, selanjutnya hitung nilai pagerank dari masing-masing kalimat dengan source code di bawah ini. Pengertian PageRank sendiri ialah algoritma otoritas tautan yang dibuat oleh Google. Ini berguna untuk membantu mesin telusur membandingkan halaman yang memenuhi syarat untuk kueri tertentu berdasarkan seberapa sering mereka direferensikan berupa tautan di halaman situs lain.<br>
# <center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/PageRanks-Example.svg/330px-PageRanks-Example.svg.png'></center><center>Gambar PageRank</center><br>PageRank merupakan istilah untuk mengambarkan skor situs berdasarkan kalkulasi dari kuantitas dan kualitas tautan masuk. Ini dilakukan algoritma Google sebagai salah satu faktor penentu peringkat sebuah website.

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


# ## **Memilih Kalimat**
# Setelah didapatkan kalimat yang memiliki nilai pagerank tertinggi, selanjutnya pilih kalimat yang memiliki nilai pagerank tertinggi, dari data dapat dilihat bahwa kalimat ke-14,11,8,16 dan seterusnya memiliki nilai pagerank dari yang paling tinggi hingga rendah.

# In[ ]:


print(sentences_list[13])
print(sentences_list[10])
print(sentences_list[7])
print(sentences_list[15])


# ## **Kesimpulan**
# Berdasar dari tahapan-tahapan yang dilakukan dapat disimpulkan bahwa ringkasan atau simpulan dokumen yang didapat ialah "Meniru secara langsung, mereplikasi, mengadaptasi, mengabsorbsi iptekin dan sistem inovasi dari negara-negara maju atau negara-negara yang sudah berhasil menerapkan hal tersebut menjadi salah satu praktik terbaik yang dilakukan oleh negara-negara sedang berkembang seperti halnya Indonesia. Sejumlah upaya yang dilakukan termasuk di antaranya
# mengembangkan berbagai konsep sistem inovasi sebagai bagian dari pengejaran dan akselerasi pembangunan ekonomi dari negara-negara maju atau NIEs tersebut dengan memperhatikan berbagai aspek mulai ekonomi, sumber daya alam, kebijakan pemerintah, organisasi/kelembagaan, sosial, dan aspek eksternal/lingkungan yang begitu luas
# dan kompleks.Ilmu pengetahuan, teknologi, dan inovasi (Iptekin) adalah salah satu elemen kunci dalam mendorong dan mempercepat pembangunan ekonomi di suatu negara.Hal ini tentu memerlukan intervensi khusus dimana pemerintah menjadi salah satu aktor penting dalam menumbuhkembangkan iptekin nasional baik dengan belajar dari negara-negara maju atau yang sudah berhasil, maupun dengan cara mengembangkan kemampuan berdasar kekuatan dan sumber daya lokal yang dimiliki oleh negara-negara tersebut." Ringkasan tersebut diperoleh dari 4 data kalimat yang memiliki nilai pagerank tertinggi.
