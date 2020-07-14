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


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


df = black_friday.copy()


# In[8]:


df.sample(10)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[118]:


def q1():   
    return df.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[117]:


def q2():
    return df[ (df['Age'] == '26-35') & (df['Gender'] == 'F')]['User_ID'].count()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[19]:


def q3():
    return df['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[27]:


def q4():
    return df.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[46]:


def q5():
    df['count_null'] = df.isna().sum(axis=1)
    return df[df['count_null'] > 0]['User_ID'].count()/df.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[20]:


def q6():
    
    return df.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[116]:


def q7():
    aux = df['Product_Category_3'].value_counts().head(1)
    return aux.index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[65]:


def q8():
    df['purchase_normalized'] = (df['Purchase'] - df['Purchase'].min()) /                                 ( df['Purchase'].max() - df['Purchase'].min())
    
    return df['purchase_normalized'].mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[75]:


def q9():
    df['purchase_satandadized'] = (df['Purchase'] - df['Purchase'].mean()) / df['Purchase'].std()
    return df[(df['purchase_satandadized'] >= -1) & (df['purchase_satandadized'] <= 1)]['User_ID'].count()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[147]:


def q10():
    aux = df[df['Product_Category_2'].isnull()]
    
    return aux['Product_Category_2'].isnull().sum() == aux['Product_Category_3'].isnull().sum()

