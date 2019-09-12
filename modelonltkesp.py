#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:20:46 2019

@author: rodolfopardo
"""

#Importando librerias 

import pandas as pd
import numpy as np 


#Importando NTLK para trabajar mis datos en textos
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Importamos nuevos modulos necesarios
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Importamos TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



#Funcion para leer el csv 

def leer(url):
    df = pd.read_csv(url)
    return df


#Funcion para limpiar datos 
def clean_data(dataframe):

    # Elimino renglones duplicados 
    dataframe.drop_duplicates(subset='Message', inplace=True)
    dataframe.drop_duplicates(subset='Link Text', inplace=True)
    
    # Me aseguro que todos los dobles espacios me queden en uno 
    dataframe['Message'] = dataframe['Message'].str.replace('  ',' ')
    dataframe['Link Text'] = dataframe['Link Text'].str.replace('  ',' ')

    # Transformar todo el texto en minúscula
    dataframe['Message'] = dataframe['Message'].str.lower()
    dataframe['Link Text'] = dataframe['Link Text'].str.lower()
    
    print("Nueva forma de nuestro dataframe:", dataframe.shape)
    return dataframe.head()



#URL de csv
url = "datalimpia.csv"

#Llamo a la funcion para leer datos 
df = leer(url)

#Se selecciona las columnas que se van a utilizar para el modelo
df3 = df[['Message', 'Link Text', 'Overperforming Score']]

#Llamo a limpiar nuestro dataframe actual
clean_data(df3)

#Se chequea que nuestro dataframe no tenga valores nulos antes de ingresar el modelo 

df3.isnull().sum()


# Se crea la variable a predecir, y
y = df3['category']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df3['Link Text'], y, test_size=0.30, random_state=36)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words= spanish_stopwords)

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

#Se suma una nueva columna con valores sujetos a una condicion de columna 2
df3['Category'] = 0
condition = df['Overperforming Score'] > 10
df3.loc[condition, 'Category'] = 'REAL'
df3.loc[~condition, 'Category'] = 'FAKE'



# Creamos la variable y
y = df3['Category']

# Creamos nuestras variables test y train
X_train, X_test, y_train, y_test = train_test_split(df3['Link Text'], y, test_size=0.30, random_state=30)

# Inicializamos countvectorizer
count_vectorizer = CountVectorizer()

# Transformamos solo train 
count_train = count_vectorizer.fit_transform(X_train)

# Transformamos solo test 
count_test = count_vectorizer.transform(X_test)

# Inicializamos TfidfVectorizer 
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)

# Transformamos train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transformamos test
tfidf_test = tfidf_vectorizer.transform(X_test)

# Inicializamos a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Transformamos train
nb_classifier.fit(count_train, y_train)

# Creamos la predicción
pred = nb_classifier.predict(count_test)

# Calculamos el parametro de nuestro modelo
score = metrics.accuracy_score(y_test, pred)
print(score)

# Imprimimnos nuestra matriz de confusión para evaluar nuestro modelo
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)























    