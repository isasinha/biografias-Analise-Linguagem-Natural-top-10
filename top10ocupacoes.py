#packages
#!pip install unidecode
#!pip install fraction

# Load EDA Pkgs
from operator import contains
import pandas as pd

# Load ML Pkgs
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('rslp')

#charts & others stuff
from unidecode import unidecode
import string

df_bios = pd.read_json("biografias.json")
df_ocupacao = pd.DataFrame({"conteudo_original": df_bios['ocupacao']})

# remoção dos registros sem informação de ocupação
df_ocupacao = df_ocupacao.drop(df_ocupacao[df_ocupacao.conteudo_original == "não informado"].index)

stop = stopwords.words('portuguese')
stop2 = list()
for word in stop:
  stop2.append(unidecode(word))
  
# remoção de palavras de parada
df_ocupacao['conteudo_tratado'] = df_ocupacao['conteudo_original'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# transforma em minuscula e remoção de acentos 
df_ocupacao['conteudo_tratado2'] = df_ocupacao['conteudo_tratado'].str.lower().apply(lambda x: unidecode(x))

# preparando lista de nacionalidades
nacionalidades = pd.read_csv("nacionalidades.csv")
nacionalidades['nacionalidades'] = nacionalidades['nacionalidades'].str.lower().apply(lambda x: unidecode(x))
nacionalidadeList = list()
for item in nacionalidades['nacionalidades']:
    nacionalidadeList.append(item)

# remoção de nacionalidades
df_ocupacao['conteudo_tratado3'] = df_ocupacao['conteudo_tratado2'].apply(lambda x: ' '.join([word for word in x.split() if word not in nacionalidadeList]))

# remoção de pontuaçao
df_ocupacao['conteudo_tratado4'] = df_ocupacao['conteudo_tratado3'].str.replace('[{}]'.format(string.punctuation), ' ')

# remoção de numeros
df_ocupacao['conteudo_tratado4'] = df_ocupacao['conteudo_tratado4'].str.replace('[{}]'.format(string.digits), '')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = CountVectorizer()

Xocupacoes = vectorizer.fit_transform(df_ocupacao['conteudo_tratado4'])
vocabulary_ocupacoes = vectorizer.get_feature_names()
pdXocupacoes = pd.DataFrame(data=Xocupacoes.toarray(), columns=vocabulary_ocupacoes) #.iloc[:,0::2]

# Transpoe a matriz para que as palavras fiquem como linhas, e cada documento como uma linha
df_Xocupacoes = pdXocupacoes.T
df_Xocupacoes['total_ocupacoes'] = df_Xocupacoes.sum(axis=1) 

#Remove as colunas individuais de cada documento para gerar a tabela
df_Xocupacoes.drop(df_Xocupacoes.columns[0:len(df_ocupacao['conteudo_tratado4'])], axis=1, inplace=True)

#Ordena o resultado final
df_Xocupacoes = df_Xocupacoes.sort_values(by='total_ocupacoes', ascending=False)

#Gera a tabela top 10 ocupações
df_Xocupacoes.head(10)
print(df_Xocupacoes.head(10))
df_Xocupacoes.head(10).to_csv("Top_10_Ocupações.csv")
df_Xocupacoes.head(10).to_json("Top_10_Ocupações.json")


#/#/#/ EXTRA: TOP 10 nacionalidades

# remoção do hífem para que palavras como norte-americano não sejam interpretadas como 2 palavras separadas
nacionalidades['nacionalidades'] = nacionalidades['nacionalidades'].str.replace('-', '')
df_ocupacao['conteudo_tratado5'] = df_ocupacao['conteudo_tratado2'].str.replace('-', '')

# transformando dataframe em lista
nacionalidadeList2 = list()
for item in nacionalidades['nacionalidades']:
    nacionalidadeList2.append(item)

# permitindo só nacionalidades
df_ocupacao['nacionalidades'] = df_ocupacao['conteudo_tratado5'].apply(lambda x: ' '.join([word for word in x.split() if word in nacionalidadeList2]))

Xnacionalidades = vectorizer.fit_transform(df_ocupacao['nacionalidades'])
vocabulary_nacionalidades = vectorizer.get_feature_names()
pdXnacionalidades = pd.DataFrame(data=Xnacionalidades.toarray(), columns=vocabulary_nacionalidades) #.iloc[:,0::2]

# Transpoe a matriz para que as palavras fiquem como linhas, e cada documento como uma linha
df_Xnacionalidades = pdXnacionalidades.T
df_Xnacionalidades['total_nacionalidades'] = df_Xnacionalidades.sum(axis=1) 

#Remove as colunas individuais de cada documento para gerar a tabela
df_Xnacionalidades.drop(df_Xnacionalidades.columns[0:len(df_ocupacao['nacionalidades'])], axis=1, inplace=True)

#Ordena o resultado final
df_Xnacionalidades = df_Xnacionalidades.sort_values(by='total_nacionalidades', ascending=False)

#Gera a tabela top 10 nacionalidades
df_Xnacionalidades.head(10)
print(df_Xnacionalidades.head(10))
df_Xnacionalidades.head(10).to_csv("Top_10_Nacionalidades.csv")
df_Xnacionalidades.head(10).to_json("Top_10_Nacionalidades.json")