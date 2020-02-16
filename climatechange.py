#!/usr/bin/env python
# coding: utf-8

# Análise e Mudanças Climáticas do Azure Notebook

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()


# Import data

# In[8]:


yearsBase, meanBase = np.loadtxt('5-year-mean-1951-1980.csv', delimiter=',', usecols=(0, 1), unpack=True)
years, mean = np.loadtxt('5-year-mean-1882-2014.csv', delimiter=',', usecols=(0, 1), unpack=True)


# Criar um gráfico de dispersão

# In[9]:


plt.scatter(yearsBase, meanBase)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=20)
plt.ylabel('mean temp difference', fontsize=20)
plt.show()


# Gráfico de dispersão produzido pelo Matplotlib

# Executar regressão linear

# In[10]:


# cria uma regressão linear a partir dos pontos de dados
m, b = np.polyfit(yearsBase, meanBase, 1)

# função linear y = mx + b
def f(x):
    return m*x + b

# Gera o mesmo gráfico de dispersão, mas adiciona linhas usando a função acima
plt.scatter(yearsBase, meanBase)
plt.plot(yearsBase, f(yearsBase))
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()

# exibe na tela os valores computados de m e b
print('y = {0} * x + {1}'.format(m, b))
plt.show()


#  A maior parte do trabalho computacional necessário para gerar a linha de regressão foi feita pela função polyfit do NumPy, que calculou os valores m e b na equação y = mx + b.

# Executar regressão linear com o Scikit-learn

# In[11]:


# Seleciona o modelo de regressão linear e instancia
model = LinearRegression(fit_intercept=True)

# Constrói o modelo
model.fit(yearsBase[:, np.newaxis], meanBase)
mean_predicted = model.predict(yearsBase[:, np.newaxis])

# Gera um gráfico
plt.scatter(yearsBase, meanBase)
plt.plot(yearsBase, mean_predicted)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()

print(' y = {0} * x + {1}'.format(model.coef_[0], model.intercept_))


# A saída é quase idêntica à saída do exercício anterior. A diferença é que o Scikit-learn fez a maior parte do trabalho para você. Especificamente, você não precisou codificar uma função de linha como fez com o NumPy; a função LinearRegression do Scikit-learn fez isso para você. O Scikit-learn é compatível com muitos diferentes tipos regressão, que são úteis na criação de modelos de machine learning sofisticados.

# Executar regressão linear com o Seaborn

# In[12]:


plt.scatter(years, mean)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
sns.regplot(yearsBase, meanBase)
plt.show()


# Comparação dos valores reais e previstos gerados com o Seaborn

# Observe como os pontos de dados para os primeiros 100 anos estão em perfeita conformidade com os valores previstos, ao contrário dos pontos de dados de aproximadamente 1980 em diante. São modelos como esses que levam os cientistas a acreditar que as mudanças climáticas estão sendo aceleradas.
