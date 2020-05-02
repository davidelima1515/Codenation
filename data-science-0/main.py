#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[84]:


import pandas as pd
import numpy as np


# In[85]:


black_friday = pd.read_csv('black_friday.csv')


# In[86]:


black_friday.shape


# ## Inicie sua análise a partir daqui

# # Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[87]:


def q1():
    res = black_friday.shape 
    return res
    


# 
# # Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[88]:


def q2():
    res = (black_friday[black_friday['Gender']=='F']['Age']=='26-35').sum()
    return int(res)


# # Questão 3 
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[89]:


def q3():
    res = black_friday['User_ID'].nunique()
    return res


# # Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[90]:


def q4():
    res = black_friday.dtypes.nunique()
    return res
    


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[91]:


def q5():
    res = (1 - len(black_friday.dropna())/ len(black_friday))
    return res
    


# # Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[92]:


def q6():
    res = black_friday.isna().sum().max()
    return int(res)
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[93]:


def q7():
    res = black_friday['Product_Category_3'].mode()
    return int(res)
    


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[94]:


def q8():
    # Retorne aqui o resultado da questão 8.
    df_aux = black_friday['Purchase']
    black_friday_norm = (df_aux - df_aux.min())/(df_aux.max() - df_aux.min())
    res = black_friday_norm.mean()
    return float(res)
    


# # Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[82]:


def q9():
    # Retorne aqui o resultado da questão 9.
    df_aux = black_friday['Purchase']
    black_friday_padr= (df_aux - df_aux.mean())/df_aux.std()
    res=black_friday_padr.between(-1,1).sum()
    return int(res)
    


# # Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[81]:


def q10():
    res = any(black_friday['Product_Category_2'].isna() & black_friday['Product_Category_3'].notna())
    if res == False:
        return True
    else:
        return False
        
    

