#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:08:12 2019

@author: rodolfopardo
"""


#Importando las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import time
from scipy import stats
from scipy.stats import ttest_1samp

#Importo funcion que lee dataframe
def leer(url):
    df = pd.read_csv(url)
    return df

#Funcion para graficar visualizaciones de dispersión

def grafica(df, columns):
    for i in columns:
        print('Imprimiendo gráficos de dispersión por medio')
        sns.scatterplot(x = i, y = 'Page Name', data = df)
        plt.xlabel(i)
        plt.ylabel('Medios de comunicación')
        plt.show()
        plt.close()
        time.sleep(2);

#Funcion para visualizar reacciones en Facebook por tiempo 
        
def setear_graficar(*args):
    print('Imprimiendo visualizaciones de tiempo')
    for i in reactions:
        df =df.set_index(x)  #seteamos el index
        df = df.sort_index()  #Ordenamos el index
        df[i].plot()   #Ploteamos 
        plt.show()
        plt.close()
        time.sleep(2)

    
url = "datalimpia.csv"  #Ubicacion del csv
df = leer(url) #Se llama funcion


#Se crea un nuevo dataframe para agrupar datos por porcentaje
df1 = pd.concat([df['Page Name'].value_counts(), 
                df['Page Name'].value_counts(normalize=True).mul(100)],axis=1, keys=('Cantidad','Porcentaje'))
print(df1)  #Lo imprimimos

#Columnas que vamos a graficar por scatter
columns = ['Page Likes at Posting', 'Likes', 'Comments', 'Shares', 'Overperforming Score']
#Llamamos a la funcion de graficos 
grafica(df, columns)


#Visualización de tipos de datos 
sns.countplot(df['Type']); 
plt.xlabel('Tipos de formatos utilizados')
plt.ylabel('Cantidad de personas')
plt.title('Tipos de publicación en Facebook')


#Columna para setear el index de nuestro dataframe 
x = 'Created'
#Columnas para graficar time series
reactions = ['Love', 'Wow', 'Haha', 'Sad', 'Angry']
#Llamado a funcion setear grafica
setear_graficar(df, x, reactions)

#Test de hipotesis 1 sobre likes que genera por día los medios de comunicación
#Community Manager: recibimos comentario de que reciben aproximadamente 1100 likes por día en sus posteos



#Luego pasamos el test de Hipotesis a la columna like
ttest_1samp(df['Likes'], 1100)

#Se rechaza Ho 

ttest_1samp(df['Likes'], 856)

#Se acepta Ho

#Cola izquierda que tiene menores valores, que esta decreciendo y afectando. 

#Test de Hipotesis 2 sobre comentarios que generan 290 comentarios por día los medios de comunicación

ttest_1samp(df['Comments'], 290)

#Cola derecha que tiene mayores valores, que está creciendo 

ttest_1samp(df['Comments'], 330)
