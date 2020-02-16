import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()


# Import data
yearsBase, meanBase = np.loadtxt('5-year-mean-1951-1980.csv', delimiter=',', usecols=(0, 1), unpack=True)
years, mean = np.loadtxt('5-year-mean-1882-2014.csv', delimiter=',', usecols=(0, 1), unpack=True)


# Criar um gráfico de dispersão usando Matplotlib
plt.scatter(yearsBase, meanBase)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=20)
plt.ylabel('mean temp difference', fontsize=20)
plt.show()

# Executar regressão linear
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

###################################################################################

# Executar regressão linear com o Scikit-learn
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


#####################################################################################

# Executar regressão linear com o Seaborn

plt.scatter(years, mean)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
sns.regplot(yearsBase, meanBase)
plt.show()

# Comparação dos valores reais e previstos gerados com o Seaborn
