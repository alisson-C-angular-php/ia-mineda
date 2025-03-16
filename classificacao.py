import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

# Caminho do arquivo
arquivo = r"https://raw.githubusercontent.com/alisson-C-angular-php/ia-mineda/refs/heads/main/2024.csv"

# Leitura do CSV
dataframe = pd.read_csv(arquivo, low_memory=False, encoding="latin1")
dataframe.rename(columns={'ï»¿Country name': 'Country Name'}, inplace=True)

# Separa as colunas numéricas (excluindo 'Country Name')
numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
#numerical_columns.remove('Country name')

# Criação do scaler
scaler = MinMaxScaler()

# Aplica o MinMaxScaler em todas as colunas numéricas, exceto 'Country Name'
dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])

# Exibe as 5 primeiras linhas
dataframe.head()

# Substitui valores nulos por 0
cols_to_fill = [
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
    "Economy (GDP per Capita)\t"
]
dataframe[cols_to_fill] = dataframe[cols_to_fill].fillna(0.1)


# Altera o tipo de 'Age' para int

dataframe['Happiness Rank'] = dataframe['Happiness Rank'].astype(int)


# Exibe as 5 primeiras linhas
dataframe.head()

# Utilizando apenas alguns atributos
X = dataframe[['Happiness score' ]]
# Queremos fazer os diagnósticos

y = dataframe['Social support']
print(y)


# Retira 20% para testar

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



print(y_train)



# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Converter valores contínuos em categorias binárias (acima ou abaixo da mediana)
threshold = np.median(y_train)
y_train = (y_train >= threshold).astype(int)
y_test = (y_test >= threshold).astype(int)

# Verifica se a conversão foi bem-sucedida
print("Valores únicos em y_train:", np.unique(y_train))

# Criar e treinar o modelo
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
model.fit(X_train, y_train)

# Predição
y_pred = model.predict(X_test)
print("Predições:", y_pred)



# Predição
y_pred = model.predict(X_test)
print('Matriz de Confusão')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.unique())
disp.plot()


# Acurácia
acc = accuracy_score(y_test, y_pred)
print("Acurácia: {:.2f}".format(acc))


# Precisão para Maligno
pre = precision_score(y_true = y_test, y_pred = y_pred, pos_label = 1)
print("Precisão (M): {:.2f}".format(pre))


# Precisão média ponderada
prem = precision_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Precisão média ponderada: {:.2f}".format(prem))



# Recall médio ponderado
recm = recall_score(y_true = y_test, y_pred = y_pred, average = "weighted")
print("Recall médio ponderado: {:.2f}".format(recm))
