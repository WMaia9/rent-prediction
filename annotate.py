import os
import settings
import pandas as pd
from geopy.geocoders import Nominatim

# ler o arquivo csv
def read():
    acquisition = pd.read_csv(os.path.join(settings.dir_processos, 'sp.csv'))
    return acquisition


# Fazer uma limpeza nos dados
def cleaning(df):
    df = df.drop_duplicates(subset=['Link'])
    df = df.reset_index(drop=True)
    df1 = df.copy()
    df1['1'] = df1['Zone'].str.split(pat="+").str.get(-1).str.replace('-', ' ')
    df1['2'] = df1['Zone'].str.split(pat="+").str.get(1).str.replace('-', ' ')
    df1['test'] = df1['Link'].str.split(pat="-").str.get(-1)
    df['Id'] = df1['test'].str.replace('/', '')
    df['Region'] = df1['1']
    df['Address'] = df1['Address'].str.cat(df1['2'], sep=' ')
    return df


# Aplicar condições para eliminar outlier
def out(df):
    df1 = df.copy()
    df1 = df1[df1['Price'] <= 15000]
    df1 = df1[df1['Vacancies'] <= 10]
    df1 = df1[df1['Total Area'] <= 50000]
    df1 = df1.reset_index(drop=True)
    return df1


# definir a posição do imovel utilizando a biblioteca geopy
def position(df):
    lat = []
    long = []
    geolocator = Nominatim(user_agent="são paulo")
    df['gcode'] = df['Address'].apply(geolocator.geocode, timeout=10)

    for row in df['gcode']:
        addr = geolocator.geocode(row, timeout=10)
        print(addr)
        if addr is None:
            lat.append(None)
            long.append(None)
        else:
            latitude = addr.latitude
            longitude = addr.longitude

        lat.append(latitude)
        long.append(longitude)

    try:
        df['latitude'] = lat
        df['longitude'] = long
    except:
        print('Error')
    df = df.drop(['gcode', 'Zone'], axis=1)

    return df


# salvar os dados em um novo arquivo csv
def write(acquisition):
    acquisition.to_csv(os.path.join(settings.dir_processos, "geo.csv"), index=False)


if __name__ == "__main__":
    acquisition = read()
    limpar = cleaning(acquisition)
    outliers = out(limpar)
    pos = position(outliers)
    write(pos)
