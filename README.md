# Detección de Fake News con NLP 
## Se utiliza base de datos de 190 diarios argentinos vía Crowdtangle para español y se utiliza API para scrapear Reddit
--------
 - [Descripción](#Descripción)
 - [Metodología](#Metodología)
 - [Evaluación de modelo](#Evaluación)
 - [Conclusiones y próximos pasos ](#Próximos)
 
## Descripción
Se realiza una petición mediante CSV a Crowdtangle, herramienta interna de Facebook, para obtener los títulos de los principales medios de Argentina en Facebook. Se obtienen datos concretos para trabajar ETL.
Se prueba el modelo en Inglés, realizando un scraping a dos canales de Reddit propicios a Fake News
## Metodología

| Data Analyst hoja de trabajo       | Descripcion                                                                                                                                                                        |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Adquisión de los datos**            | Se utiliza Crowdtangle para la obtención de diarios argentinos y se usa la api pushshift.io para scrapear datos en Reddit.                                              |
| **Exploración de nuestros datos**   | Se limpian los datos correspondientes, se corrigen errores de tipos de datos. Se visualizan scatters plot                                                           |
| **Pruebas de hipótesis** | Se realiza una entrevista al Community Manager de Grupo Clarín para poder establecer una hipótesis nula y alternativa. Se realiza prueba de cola izquierda.                                                                                                 |
| **Modelo**                    | Se usa Multinomial Naive Bayes classifier, antes se vectoriza. |


 
## Evaluación de modelos 
Se utilizan scores de modelos y matrices de confusión para medir la eficiencia de los mismos.

## Próximos pasos 
1. Lograr modelos que superen el 90% de nuestros datos. 
2. Obtener más datos para entrenar el modelo. 
3. Perfeccionar el modelo en español y obtener nuevas librerías
4. Convertir variables categóricas a numéricas y poder probar nuevos modelos 
5. Me encontré con el conflicto de que este tipo de librerías no tienen un buen desarrollo para trabajar palabras en español por lo que intetaría trabajar datos cuantitativos y mudar modelos a predicciones supervisadas
