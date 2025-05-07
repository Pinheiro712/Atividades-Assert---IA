import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cabecalho = ['comprimento_sepala', 'largura_sepala', 'comprimento_petala', 'largura_petala', 'especie']
df = pd.read_csv(data, names=cabecalho)

x = df.drop('especie', axis= 1)
y = df['especie']


x_treinar, x_teste, y_treinar, y_teste = train_test_split(x,y, test_size = 0.2)

modelo = DecisionTreeClassifier()
modelo.fit(x_treinar, y_treinar)
y_pred = modelo.predict(x_teste)

precisão = accuracy_score(y_teste, y_pred)
print("Precisão:", precisão)