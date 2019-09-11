#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:27:24 2019

@author: rodolfopardo
"""

import pandas as pd   #Se importa pandas 
import numpy as np   #Se importa numpy 


#Se busca una funcion para leer dataframe 
def leer(url): 
    df = pd.read_csv(url)
    print('Dataframe leído')
    return df

#Se rellenan los nulos en columnas imprescindibles
def relleno(df, nombres):
    for i in nombres:
        df[i] = df[i].fillna('sin mensaje')
    print('Columnas rellenadas')
    return df
    
#Se eliminan columnas que no vamos a utilizar
def elimina(df, columnas):
    df = df.drop(columns = columnas, axis = 1)
    print('Columnas eliminadas')
    return df
    

#Se transforman el tipo de datos de dos columnas
def transforma(*args):
    df[col] = df[col].str[0:19]
    df[col] = pd.to_datetime(df[col].astype(str), format = '%Y-%m-%d %H:%M:%S')
    df[col1] = pd.to_numeric(df[col1])
    print('Columnas transformadas')
    return df

#Direccion del df
url = 'diariosarg.csv'
#Leemos el dataframe 
print('Leemos dataframe')
df = leer(url)
#Columnas que usa relleno para rellenar bases 
nombres = ['Message', 'Link Text', 'Description']
#Nombre de columnas que vamos a eliminar
columnas = ['User Name', 'Video Share Status',
            'Final Link', 'Sponsor Id', 'Sponsor Name',
            'URL', 'Link', 'Video Length', 'Thankful']

#Se eliminan las primeras 6 valores por numeros erroneos 
df = df[7:100000]

#Columna para transformar el tiempo 
col = 'Created'
col1= 'Overperforming Score'

#Llamo a columna relleno
print('Vamos a rellenar nans')
df = relleno(df, nombres)
#Llamo a función para eliminar columnas
print('Vamos a eliminar columnas que no hacen falta')
df = elimina(df,columnas)
#Llamo para transformar columna de tiempo 
print('Se transforman datos')
df = transforma(df, col)
#Se guarda el dataframe 
df.to_csv('datalimpia.csv')




