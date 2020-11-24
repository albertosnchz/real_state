#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('LArealestate.csv')
df


# In[ ]:


df.info()


# In[ ]:


# Elimino los espacios vacios de los strings del principio
df.address = df.address.apply(lambda x: x.lstrip())


# In[ ]:


# Transformar address a numero o np.nan
def address_to_nan(x):
    if x[:1].isdigit():
        return int(x.split(' ')[0])
    else:
        return np.nan


# In[ ]:


# Aplico funcion de transformación
df.address = df.address.apply(address_to_nan)


# In[ ]:


# Imputo np.nan por su moda
df.beds = df.beds.replace(np.nan,df.beds.mode()[0])
df.baths = df.baths.replace(np.nan,df.baths.mode()[0])
df.address = df.address.replace(np.nan,df.address.mode()[0])


# In[ ]:


# Redondeo los valores de baños y habitaciones
df.beds = df.beds.apply(lambda x: round(x))
df.baths = df.baths.apply(lambda x: round(x))


# In[ ]:


# Grafico para mostrar los valores max y min de mis columnas numericas
for column in ['beds', 'baths']:
    sns.countplot(df[column].values)
    plt.show()
    print('Valor Minimo:', min(df[column]))
    print('Valor Maximo:', max(df[column]))


# In[ ]:


# Funcion para detectar outliers
def outliers(data):
    outliers = list()
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for y in data:
        z_score= (y - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# In[ ]:


for i in ['beds', 'baths']:
    print(outliers(df[i].values))


# In[ ]:


# Elimino las filas con outliers en baños y habitaciones
df.drop(df[df.beds==10].index, inplace = True)
df.drop(df[df.baths==2822].index, inplace = True)

