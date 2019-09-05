#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:38:11 2019

@author: rodolfopardo
"""

#Importando librerias 

import pandas as pd #importando pandas
import numpy as np #importando numpy 
import time        #Importando tiempo para visualizar de mejor manera

def carga(url):
    df = pd.read_csv(url)
    return df

def lectura(df):
    print('La data tiene {} columnas y {} filas'.format(df.shape[0], df.shape[1]))  #Se imprime el formato
    print('Mostrando los primeros valores...')
    time.sleep(3)
    print(df.head())   #Primeros valores
    print('Ahora, los últimos valores...')
    time.sleep(3)
    print(df.tail())   #Ultimos valores
    print('Buscando los tipos de datos de nuestras columnas')
    time.sleep(3)
    print(df.dtypes)
    print('Buscando valores nulos por columna')
    time.sleep(3)
    print(df.isnull().sum())
    
url = 'diariosarg.csv'

df = carga(url)
lectura(df)

    
    
