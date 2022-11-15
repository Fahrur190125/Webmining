#!/usr/bin/env python
# coding: utf-8

# # Crawling Website
# Crawling merupakan suatu proses pengambilan data dengan menggunakan mesin yang dilakukan secara online. Proses ini dilakukan untuk mengimpor data yang ditemukan kedalam file lokal komputer. Untuk dapat melakukan crawling dengan menggunakan python kita dapat menggunakan scrapy.

# ## **Installasi Scrapy**
# Scrapy adalah kerangka kerja aplikasi untuk crawling web site dan mengekstraksi data terstruktur yang dapat digunakan untuk berbagai aplikasi yang bermanfaat, seperti data mining, pemrosesan informasi atau arsip sejarah. Meskipun Scrapy awalnya dirancang untuk web scraping, namu scrapy juga dapat digunakan untuk mengekstrak data menggunakan API (seperti Amazon Associates Web Services) atau sebagai web crawl.<br>
# Untuk menggunakan scrapy terlebih dahulu install scrapy dengan source code berikut.

# In[1]:


get_ipython().system('pip install scrapy')


# ## **Crawling Abstrak**
# Setelah berhasil menginstall scrapy, selanjutnya kita dapat melakukan proses crawling data abstrak dari pta.trunojoyo menggunakan library scrapy dengan menggunakan source code berikut.

# In[2]:


import scrapy
from scrapy.crawler import CrawlerProcess


class AbstracToCsv(scrapy.Spider):
  name = 'Abstrac To CSV'
  def start_requests(self):
        x = 100000
        for i in range (0,75):
            x +=1
            urls = [
                'https://pta.trunojoyo.ac.id/welcome/detail/130411'+str(x),
                #'https://pta.trunojoyo.ac.id/welcome/detail/140411'+str(x),
                #'https://pta.trunojoyo.ac.id/welcome/detail/150411'+str(x),
                #'https://pta.trunojoyo.ac.id/welcome/detail/160411'+str(x)
            ]
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)

  custom_settings = {
      'FEED_FORMAT': 'csv',
      'FEED_URI': 'Abstraksi.csv'
  }

  def parse(self, response):
        yield {
            'Abstrak':response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').get()
              }

process = CrawlerProcess()
process.crawl(AbstracToCsv)
process.start()


# ## **Buang Baris Kosong**
# Setelah didapat hasil crawling kita baca file csv tersebut, akan tetapi pada hasil crawling tersebut masih terdapat baris yang kosong yang didapat dari link yang kosong, untuk itu perlu dihilangkan dengan source code berikut.

# In[3]:


import pandas as pd 
abs = pd.read_csv("Abstraksi.csv",index_col=False)


# In[4]:


abs.dropna(inplace=True)
abs.isnull().sum()


# Setelah berhasil dihapus, simpan kembali dalam format csv

# In[5]:


abs.to_csv("hasil_crawling.csv",index=False)


# ## **Baca Hasil Crawling**
# Setelah baris kosong dari hasil crawling sudah dihapus kita tampilkan hasil crawling data absrak dari pta.trunojoyo seperti berikut.

# In[6]:


data = pd.read_csv("hasil_crawling.csv",index_col=False)
data


# ## **Kesimpulan**
# Dari hasil yang diperoleh, didapat data abstrak sebanyak 125 data abstrak yang diperoleh dari pta.trunojoyo.ac.id dengan mengambil data abstrak fakultas teknik mulai dari angkatan 13 hingga 16, dengan urutan NIM dari 0-75. Data abstrak yang diperoleh tersebut disimpan dalam bentuk csv.
