#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing  import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv", decimal=',')


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

#pré-processamento
countries.columns = new_column_names
try:
    for i in range(0, countries['Region'].shape[0]):
                   countries['Region'][i] = countries['Region'][i].strip()
except:
    pass 

countries.head(5)


# In[5]:


countries.shape


# In[6]:


df_aux = pd.DataFrame({
                'tipos': countries.dtypes,
                'total': countries.shape[0],
                'faltantes':countries.isna().sum(),
                '% faltante': 100*countries.isna().sum() / countries.shape[0]
})

countries['Climate'].fillna(0, inplace=True)

df_aux


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[ ]:





# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[18]:


def q1():
    resp = countries['Region'].unique()
    return list(sorted(resp))
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[8]:


def q2():
    discr = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    x = discr.fit_transform(countries['Pop_density'].values.reshape(-1,1))
    return int((x>=9).sum())
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[9]:


def q3():
    encoding = OneHotEncoder(sparse=False)
    encoded = encoding.fit_transform(countries[['Region', 'Climate']])

    return int(encoded.shape[1])
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[11]:


def q4():
    criando_pipe = Pipeline(steps=[
                                ('imputer',SimpleImputer(strategy="median")),
                                ('standardscaler', StandardScaler())
                                ])
    lista = []
    for i in countries.columns:
        if countries[i].dtypes == 'float64' or countries[i].dtypes == 'int64':
            lista.append(i)
        
    implementador = criando_pipe.fit(countries[lista])

    novo_df = pd.DataFrame([test_country], columns=countries.columns)
    resp = implementador.transform(novo_df[lista])[0][lista.index('Arable')].round(3)
    return float(resp)
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[12]:


def q5():
    countries['Net_migration'].dropna()
    q1,q3 = countries['Net_migration'].quantile([0.25, 0.75])
    iqr = q3 - q1
    out_baixo, out_alto = int((countries['Net_migration'] < q1 - 1.5*iqr).sum()), int((countries['Net_migration'] > q3 + 1.5*iqr).sum())
    aceitacao = bool (out_baixo/countries['Net_migration'].shape[0] < 0.05 or out_alto/countries['Net_migration'].shape[0] < 0.05)
    return out_baixo, out_alto, aceitacao
q5()


# In[13]:


#sns.boxplot(countries['Net_migration'], orient='vertical')
#sns.distplot(countries.Net_migration)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[14]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[15]:


def q6():
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(newsgroup.data)
    dx = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    return int (dx['phone'].sum())
q6()  


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[16]:


def q7():
    tf = TfidfVectorizer()
    news_tf = tf.fit_transform(newsgroup.data)
    dx = pd.DataFrame(news_tf.toarray(), columns=tf.get_feature_names())
    return float(dx['phone'].sum().round(3))
q7()

