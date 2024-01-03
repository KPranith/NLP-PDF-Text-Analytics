#--------------------------------------------------Code Start---------------------------------------------------------#
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:13:53 2018

@author: Jasmi
Description: NLP code for Bis Data
Here is the data:
  - https://www.bis.org/list/research/index.htm
  - https://www.bis.org/cbhub/index.htm
  - https://www.bis.org/list/wpapers/index.htm
"""
from pypdf import PdfReader
import os, sys

#import textract

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt');
nltk.download('wordnet')
nltk.download('stopwords')


def searchInPDF(filename, searchWords):
    occurrences = 0
    pdfFileObj = open(filename,'rb')
    pdfReader = PdfReader(pdfFileObj)
    num_pages = len(pdfReader.pages)
    count = 0
    text = ""
    while count < num_pages:
        pageObj = pdfReader.pages[count]
        count +=1
        text += pageObj.extract_text()
    if text != "":
       text = text
#   else:
  #     text = textract.process(filename, method='tesseract', language='eng')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [word.lower() for word in tokens]
    print(len(tokens))


    punctuation = ['(',')',';',':','[',']',',']
    stop_words = stopwords.words('english')
    keywords = [word for word in tokens if not word in stop_words and  not word in punctuation]
    for k in keywords:
        if k in searchWords: searchWords[k]+=1
    return searchWords

directory = '/Users/kunalap/Downloads/files'
#pdf_filename = '0330.pdf'

n = len(sys.argv);
print ("Number for words passed in for search");
print("Name of the script executing", sys.argv[0])
print("\nArguments passed:", end = " ")
searchWords = dict()
for i in range(1, n):
    searchWords[sys.argv[i]] = 0;
    print(sys.argv[i], end =' ')


for file in os.listdir(directory):
    if not file.endswith(".pdf"):
        continue
    pdf_filename =  os.path.join(directory,file)
    print(pdf_filename)
    result = searchInPDF(pdf_filename,searchWords)
    print(result)

#--------------------------------------------------Code End---------------------------------------------------------#

