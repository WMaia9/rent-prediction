import os
import settings
import pandas as pd
import numpy as np
import geofast
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Ler os arquivos CSV dos imóveis
def read():
    df = pd.read_csv(os.path.join(settings.dir_processos, 'geo.csv'))
    return df

# Ajustar os dados de acordo com as coordenadas da cidade de São Paulo
def ajust(df):
    df = df.loc[df['latitude'] <= -23.26]
    df = df.loc[df['longitude'] <= -46.4]
    df = df.loc[df['latitude'] >= -23.9]
    df = df.loc[df['longitude'] >= -46.8]
    df = df.reset_index(drop=True)
    return df

# Funções de distância
def distance(df):
    #Calcula a distância mais próxima
    data = ['dist_subway.csv', 'dist_bus.csv', 'dist_school.csv']
    rows = df.index
    for i in data:
        stops = pd.read_csv(os.path.join(settings.dir_sp, i))
        df[i] = np.nan
        for n in rows:
            h = df.loc[n, ["latitude", "longitude"]]
            df.loc[n, i] = geofast.distance(h.latitude, h.longitude, stops.latitude, stops.longitude).min()

    # Calcula os pontos em uma localidade de 1km
    data = ['culture.csv', 'crime.csv', 'restaurant.csv']
    for i in data:
        region = pd.read_csv(os.path.join(settings.dir_sp, i))
        df[i] = np.nan
        for n in rows:
            h = df.loc[n, ["longitude", "latitude"]]
            h500 = geofast.distance(region.latitude, region.longitude, h.latitude, h.longitude)
            df[i][n] = sum(h500 < 700)
    df['crime.csv'] = df['crime.csv'] * 100 / 12377
    return df

# Renomeia as variáveis
def rename(df):

    df = df.rename(columns={'dist_bus.csv': 'dist_bus', 'dist_subway.csv': 'dist_subway',
                            'dist_school.csv': 'dist_school', 'culture.csv': 'ncult',
                            'crime.csv': 'crime%', 'restaurant.csv': 'food'})

    df = df[['Id', 'Price', 'Type', 'Region', 'Total Area', 'Bathrooms',
             'Bedrooms', 'Vacancies', 'dist_subway', 'dist_bus', 'dist_school',
             'ncult', 'food', 'crime%', 'latitude', 'longitude', 'sala', 'suíte',
             'mobiliado', 'piscina', 'churrasqueira', 'salão', 'varanda', 'academia',
             'duplex', 'quintal', 'armário', 'reformado', 'sobrado', 'condicionado',
             'condominio', 'escritorio']]
    return df

# Engenharia de Recursos

def resources(df):

    df['sala'] = (df["Description"].str.lower().str.contains("sala", regex=False, na=False).astype(int))
    df['suíte'] = (df["Description"].str.lower().str.contains("suíte", regex=False, na=False).astype(int)) | (df["Description"].str.lower()
                                .str.contains("suite", regex=False, na=False).astype(int))
    df['mobiliado'] = (df["Description"].str.lower().str.contains("mobiliado", regex=False, na=False).astype(int))
    df['piscina'] = (df["Description"].str.lower().str.contains("piscina", regex=False, na=False).astype(int))
    df['churrasqueira'] = (df["Description"].str.lower().str.contains("churrasqueira", regex=False, na=False).astype(int))
    df['salão'] = (df["Description"].str.lower().str.contains("salão", regex=False, na=False).astype(int))
    df['varanda'] = (df["Description"].str.lower().str.contains("varanda", regex=False, na=False).astype(int))
    df['academia'] = (df["Description"].str.lower().str.contains("academia", regex=False, na=False).astype(int))
    df['duplex'] = (df["Description"].str.lower()
                                .str.contains("duplex", regex=False, na=False).astype(int))
    df['quintal'] = (df["Description"].str.lower().str.contains("quintal", regex=False, na=False).astype(int))
    df['armário'] = (df["Description"].str.lower().str.contains("armário", regex=False, na=False).astype(int)) | (df["Description"].str.lower()
                                .str.contains("armario", regex=False, na=False).astype(int))
    df['reformado'] = (df["Description"].str.lower().str.contains("reforma", regex=False, na=False).astype(int)) | (df["Description"].str.lower()
                                .str.contains("reformado", regex=False, na=False).astype(int))
    df['sobrado'] = (df["Description"].str.lower().str.contains("sobrado", regex=False, na=False).astype(int))
    df['condicionado'] = (df["Description"].str.lower().str.contains("condicionado", regex=False, na=False).astype(int))
    df['condominio'] = (df["Description"].str.lower().str.contains("condomínio", regex=False, na=False).astype(int)) | (df["Description"].str.lower()
                                .str.contains("condominio", regex=False, na=False).astype(int))
    df['escritorio'] = (df["Description"].str.lower().str.contains("escritório", regex=False, na=False).astype(int)) | (df["Description"].str.lower()
                                .str.contains("escritorio", regex=False, na=False).astype(int))
    df['Type'] = df['Type'].replace({'apartamentos': 'apartament', 'casas': 'house'})
    return df

# Escreve os dados
def split(df):
    df_train, df_test = train_test_split(df, test_size=0.3)
    df_train.to_csv(os.path.join(settings.dir_processos, "train.csv"), index=False)
    df_test.to_csv(os.path.join(settings.dir_processos, "test.csv"), index=False)

if __name__ == "__main__":
    df = read()
    df = ajust(df)
    df = distance(df)
    df = resources(df)
    df = rename(df)
    split(df)
