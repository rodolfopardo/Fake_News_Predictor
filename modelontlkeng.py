#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:10:17 2019

@author: rodolfopardo
"""

#Importando librerias 

import pandas as pd
import numpy as np
#import sys 
#!{sys.executable} -m pip install psaw
#Libreria para scrapear Reddit y redes sociales de forma dinámica y fácil
from psaw import PushshiftAPI
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modelos que vamos a aplicar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


#Funcion para scrapear datos 
def scrape_data(subreddit):
    
    # Inicializacion 
    api = PushshiftAPI()

    # Creo una lista con el scrapping
    scrape_list = list(api.search_submissions(subreddit=subreddit,
                                filter=['title', 'subreddit', 'num_comments', 'author', 'subreddit_subscribers', 'score', 'domain', 'created_utc'],
                                limit=15000))

    #Filtro el subreddit por author y titulos 
    clean_scrape_lst = []
    for i in range(len(scrape_list)):
        scrape_dict = {}
        scrape_dict['subreddit'] = scrape_list[i][5]
        scrape_dict['author'] = scrape_list[i][0]
        scrape_dict['domain'] = scrape_list[i][2]
        scrape_dict['title'] = scrape_list[i][7]
        scrape_dict['num_comments'] = scrape_list[i][3]
        scrape_dict['score'] = scrape_list[i][4]
        scrape_dict['timestamp'] = scrape_list[i][1]
        clean_scrape_lst.append(scrape_dict)

    # Ver numero de suscriptores
    print(subreddit, 'subscribers:',scrape_list[1][6])
    
    # Retorno lista de scrapping
    return clean_scrape_lst

def clean_data(dataframe):

    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
    
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers 
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single 
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
    
    #Formato para imprimir luego de la limpieza
    print("Nuevo formato:", dataframe.shape)
    return dataframe.head()

#Funcion para generar barplots 

def bar_plot(x, y, title, color):    
    
    # Seteamos parametros principales de nuestros barplots 
    plt.figure(figsize=(9,5))
    g=sns.barplot(x, y, color = color)    
    ax=g

    # Agregamos las etiquetas correspondientes
    plt.title(title, fontsize = 15)
    plt.xticks(fontsize = 10)

    
    totals = []

    # Encontrando variables y haciendo append 
    for p in ax.patches:
        totals.append(p.get_width())

    # Seteando el eje con la suma de las etiquetas
    total = sum(totals)

    # Seteando el eje con las etiquetas 
    for p in ax.patches:
        ax.text(p.get_width()+.3, p.get_y()+.38, \
                int(p.get_width()), fontsize=10)

# Llamo la funcion de scrapping y creo el dataframe
df_not_onion = pd.DataFrame(scrape_data('nottheonion'))

# Guardo csv
df_not_onion.to_csv('not_onion.csv')

# Imprimimos el nuevo formato de noticias NO Onion
print(f'df_not_onion shape: {df_not_onion.shape}')

# Mostramos los primeros cinco valores
df_not_onion.head()

# Llamamos a la funcion de scrapear y lo convertimos en dataframe
df_onion = pd.DataFrame(scrape_data('theonion'))

# Lo salvamos en csv
df_onion.to_csv('the_onion.csv')

# Formato de nuestro dataframe ONION
print(f'df_onion shape: {df_onion.shape}')

# Mostramos los primeros cinco valores
df_onion.head()

#Leemos varios dataframes 
df_onion = pd.read_csv('the_onion.csv')
df_not_onion = pd.read_csv('not_onion.csv')

#Limpiamos dataframes
clean_data(df_onion)
clean_data(df_not_onion)

#Verificamos si ambos dataframes tienen valores nulos o no
pd.DataFrame([df_onion.isnull().sum(),df_not_onion.isnull().sum()], index=["TheOnion","notheonion"]).T

# Convertimos nuestros variables de tiempo de forma correcta
df_onion['timestamp'] = pd.to_datetime(df_onion['timestamp'], unit='s')
df_not_onion['timestamp'] = pd.to_datetime(df_not_onion['timestamp'], unit='s')

# Vemos el rango de tiempo que tenemos en cuanto a las noticias traidas de ONION y NO ONION
print("TheOnion start date:", df_onion['timestamp'].min())
print("TheOnion end date:", df_onion['timestamp'].max())
print("nottheonion start date:", df_not_onion['timestamp'].min())
print("nottheonion end date:", df_not_onion['timestamp'].max())

# Mostramos una previa de como queda nuestro dataframe
df.head(10)
       
X = df['title']
y = df['subreddit']

#Divido mi base para entrenar mi modelo
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    stratify=y)

#Iniciamos clasificadory vectorizador
nb = MultinomialNB(alpha = 0.36)
cvec = CountVectorizer(ngram_range= (1, 3))

# Entrenamos el modelo
cvec.fit(X_train)

Xcvec_train = cvec.transform(X_train)
Xcvec_test = cvec.transform(X_test)

# Entrenamos el train
nb.fit(Xcvec_train,y_train)

# Creamos la prediccion
preds = nb.predict(Xcvec_test)

print("El modelo tiene un score de :", nb.score(Xcvec_test, y_test))

# Imprimimos la matriz de resultados
cnf_matrix = metrics.confusion_matrix(y_test, preds)
cnf_matrix
