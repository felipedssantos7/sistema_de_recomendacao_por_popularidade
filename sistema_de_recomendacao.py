# modulos basicos
import pandas as pd
import numpy as np

# modulos de visualizacao de dados
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# lendo dados do arquivo CSV
df = pd.read_csv('ratings_Grocery_and_Gourmet_Food.csv', header=None, names=['user_id', 'prod_id', 'rating', 'timestamp'])
# visualiza 5 primeiras linhas
print(df.head(5))
# apaga coluna timestamp
df.drop('timestamp', axis=1, inplace=True)
# segunda visualizacao (sem timestamp)
print(df.head(5))

# informacoes da tabela de dados
print(df.shape)
# estatisticas
print(df.describe(include='all'))
# procura celulas vazias
print(df.isnull().sum())

# grafico de frequencia
plt.figure(figsize=(6,3)) # dimensoes da telinha
sns.countplot('rating', data=df, alpha=0.85)  # mostra grafico de frequencia das avaliacoes
plt.xlabel('Ratings', size=6)
plt.show() # mostra telinha

# informacoes importantes
num_ratings = len(df); # numero de avaliacoes
num_users = len(df.user_id.unique()) # numero de usuarios
avg_ratings_users = round(float(num_ratings) / float(num_users), 2) # media de avalicoes por usuarios
# exibe informacoes
print('Numero total de avaliacoes: ' + str(num_ratings))
print('Numero total de usuarios: ' + str(num_users))
print('Media de avaliacoes por usuario: ' + str(avg_ratings_users))

# usuarios que mais avaliaram produtos
more = df.groupby('user_id').rating.count().sort_values(ascending=False)
# exibe telinha com usuarios que mais avaliaram produtos
plt.figure(figsize=(10,5))
more.head(10).plot(kind='bar', alpha=0.9, width=0.85)
plt.title('Usuarios que mais avaliaram produtos', size=15)
plt.show()

# criacao de um conjunto de quantis de 0 a 100% com etapa de 1%
quantiles = more.quantile(np.arange(0, 1.01, 0.01), interpolation='higher')
# escala de registro de quantis
quantiles_log = np.log(quantiles)
# grafico dos quantis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,6))
ax1.plot(quantiles_log, c='red')
ax2.plot(quantiles, linewidth=2)
fig.suptitle('Avalicoes de usuarios - Grafico de registro de quantis | Grafico de quantis', size=18)
plt.xlabel('Quantis')
plt.ylabel('Numero de avalicoes')
plt.xticks(np.arange(0, 1.01, 0.1))
plt.show()

# quantidade de usuarios que avaliaram mais de 50 produtos
print('Quantidade de usuarios que avaliaram 50 ou mais produtos: ' + str(sum(more>=50)))
print('Quantidade de usuarios que avaliaram mais de 50 produtos: ' + str(sum(more>50)))

# quadro de dados com produtos que tem mais de 50 avaliacoes
new_df=df.groupby("prod_id").filter(lambda x:x['rating'].count() >=50)
print(new_df.head())
# criando um novo quadro de dados com a media
ratings_df = pd.DataFrame(new_df.groupby('prod_id').rating.mean())
# adicionando coluna com o numero de avaliacoes por produto
ratings_df['rating_counts'] = new_df.groupby('prod_id').rating.count()
# ordena quadro de dados comencando pelos produtos mais vezes avaliados
print(ratings_df.sort_values(by='rating_counts', ascending=False).head(5))

# grafico de quantidade de avaliacoes de produtos
ratings_df.rating_counts.hist(bins=50)
plt.title('Grafico de frequencia de produtos por numero de avaliacoes')
plt.show()

# grafico de dispersao das medidas de avaliacoes vs quantidades de avaliacoes
sns.jointplot(x='rating', y='rating_counts', data=ratings_df, alpha=0.4)
plt.show()

# media global das avaliacoes
C = ratings_df['rating'].mean()
print(round(C, 2))

# use o numero minimo de votos necessarios para estar no recomendador de popularidade
m = ratings_df.rating_counts.min()
print(m)

# funcao que calcula a pontuacao de cada produto
def weighted_rating(x, m=m, C=C):
	v = x['rating_counts']
	R = x['rating']
	# calculo baseado na formula IMDB
	return (v/(v+m)*R) + (m/(m+v)*C)

# adiciona a coluna "score" (pontuacao) calculada pela funcao weighted_rating() no quadro de dados
ratings_df['score'] = ratings_df.apply(weighted_rating, axis=1)

# resultado final com o top 15 dos produtos mais populares
print(ratings_df.sort_values(by='score', ascending=False).head(15))
