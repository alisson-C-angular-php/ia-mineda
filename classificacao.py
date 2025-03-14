import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Caminho do arquivo
arquivo = r"C:/Users/aliss/OneDrive/Área de Trabalho/classificação/2024.csv"

# Leitura do CSV
dataframe = pd.read_csv(arquivo, sep=",", low_memory=False, encoding="latin1")

# Exibe as 5 primeiras linhas
print(dataframe.head(5))

# Verifica a quantidade de valores nulos antes da substituição
print("Valores nulos antes da substituição:")
print(dataframe.isnull().sum())

# Limpeza do nome das colunas (removendo espaços e tabulações)
dataframe.columns = dataframe.columns.str.strip()

# Substitui valores nulos por 0
cols_to_fill = [
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
    "Economy (GDP per Capita)"
]
dataframe[cols_to_fill] = dataframe[cols_to_fill].fillna(0)

# Verifica novamente a quantidade de valores nulos
print("Valores nulos após a substituição:")
print(dataframe.isnull().sum())

# Forçando os tipos de dados
DTYPES = {'Happiness Rank': int,'Generosity':float,'Happiness score':float}
dataframe = dataframe.astype(DTYPES)

print(dataframe.dtypes)

# Padronização dos dados numéricos
numerical_attributes = ["Happiness Rank"]
numerical_columns = dataframe[numerical_attributes]

# Instancia e treina MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(numerical_columns)

# Normaliza colunas numéricas
norm_data = pd.DataFrame(scaler.transform(numerical_columns), columns=scaler.feature_names_in_)

# Exclui as colunas originais e junta as normalizadas
numeric_train = dataframe.drop(numerical_attributes, axis=1)
numeric_train = pd.concat([numeric_train, norm_data], axis=1)

print(numeric_train.head())


# APLICANDO DISCRETIZAÇÃO
atribut = 'Happiness score'
category = ['Infeliz', 'Pouco Feliz', 'Feliz', 'Muito Feliz']

# Definindo os intervalos para a discretização
bins = [0, 3.25, 5.5, 7.75, 10]

# Aplicando a discretização
bin = numeric_train.copy()
bin[atribut] = pd.cut(x=bin[atribut], bins=bins, labels=category)

# Exibindo o resultado
print(bin)


