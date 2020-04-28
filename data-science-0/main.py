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

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv('black_friday.csv')


# In[3]:


black_friday.shape


# ## Inicie sua análise a partir daqui

# # Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    


# 
# # Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[29]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return (black_friday[black_friday['Gender']=='F']['Age']=='26-35').sum()

    


# In[27]:


#return len(black_friday.query('Gender == "F" &  Age == "26-35"')
    


# # Questão 3 
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[41]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# # Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[42]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()
    


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[43]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return 1 - len(black_friday.dropna())/ len(black_friday)
    


# # Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[50]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())
    


# In[49]:





# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[1]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday['Product_Category_3'].mode())
    


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:





# In[334]:


def q8():
    # Retorne aqui o resultado da questão 8.
    df_aux = black_friday['Purchase']
    black_friday_norm = (df_aux - df_aux.min())/(df_aux.max() - df_aux.min())
    return float(black_friday_norm.mean())
    


# # Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[335]:


def q9():
    # Retorne aqui o resultado da questão 9.
    df_aux = black_friday['Purchase']
    black_friday_padr= (df_aux - df_aux.mean())/df_aux.std()
    return int(black_friday_padr.between(-1,1).sum())
    


# # Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[348]:


def q10():
    # Retorne aqui o resultado da questão 10.
    if any(black_friday['Product_Category_2'].isna() & black_friday['Product_Category_3'].notna()) == False:
        return True
    else:
        return False
        
    


# In[ ]:




