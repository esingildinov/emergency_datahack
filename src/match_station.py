import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic

def insert_space(string):
    return string[0:3] + ' ' + string[3:]


def insert_dot(string):
    return string[0:2] + '.' + string[2:]

# функция, склеивающая колонки с координатами в кортеж


def coord_merge(df: pd.DataFrame) -> pd.DataFrame:
    df['lat_long'] = df[['lat', 'lon']].apply(tuple, axis=1)
    return df

# аналогичная функция для датасета, содержащего координаты, соответствующие километрам трассы (данные первого геокодера)
def coord_km(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'lat_geoc': "Широта"})
    df = df.rename(columns={'lon_geoc': "Долгота"})
    df['Широта'] = round(df['Широта'], 2)
    df['Долгота'] = round(df['Долгота'], 2)
    df['lat_long'] = df[['Широта', 'Долгота']].apply(tuple, axis=1)
    return df

# данные Глонасса
def coord_km_glonass(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'lat_glonass': "Широта"})
    df = df.rename(columns={'lon_glonass': "Долгота"})
    df['Широта'] = round(df['Широта'], 2)
    df['Долгота'] = round(df['Долгота'], 2)
    df['lat_long'] = df[['Широта', 'Долгота']].apply(tuple, axis=1)
    return df

# получаем список метеостанций с координатами из поданного датасета Росгидромета
def stations_list(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['station', 'lat_long']]
    df = df.drop_duplicates()
    df = df.set_axis(range(0, len(df)))
    return df

# находим ближайшую метеостанцию, сопоставляя координаты километра трассы с координатами метеостанций из списка
def stat_km(point, stat_list):
    stations_list = stat_list
    lst = []
    if pd.isnull(point) == True:
        lst.append(np.nan)
    else:
        for i in stations_list['lat_long']:
            x = geodesic(point, i).km
            lst.append(x)
            stations_list['dist'] = pd.DataFrame(lst)
            y = stations_list['station'][stations_list['dist']
                                         == stations_list['dist'].min()]
        y = y.to_string()
        y = re.sub("[0-9]", "", y)
        y = re.sub(" ", "", y)
        return y
