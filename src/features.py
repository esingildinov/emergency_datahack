import pandas as pd
import numpy as np
import functools

from typing import Optional, List
from match_station import coord_km, coord_merge, stations_list, stat_km


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    df['segment_id'] = df[['road_id', 'road_km']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df

def compile_train(train: pd.DataFrame,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  test_segment_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Function for compiling training set with set of segments and each hour between start and end date

    Args:
        train (pd.DataFrame): [description]
        segment_ids (Optional[List[str]], optional): [description]. Defaults to None.
        start_date (Optional[str], optional): [description]. Defaults to None.
        end_date (Optional[str], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """

    if 'segment_id' not in train.columns:
        train = add_segment_id(train)

    segment_ids = train['segment_id'].unique()

    if start_date is None:
        start_date = train.datetime.min().strftime('%Y-%m-%d %H:%M:00')
    if end_date is None:
        end_date = train.datetime.max().strftime('%Y-%m-%d %H:%M:00')

    tr = pd.DataFrame({'datetime': pd.date_range(start_date, end_date, freq="1h")})

    for sid in segment_ids:
        tr[str(sid)] = 0
        events = train.loc[train['segment_id'] == sid]
        datetimes = events.datetime.dt.floor('H')
        accident_classes = events.target 
        dates = datetimes.astype(str).unique()
        for date in dates:
            target_value = train[(train.datetime == date) & (train.segment_id == sid)]['target'].values[0]
            tr.loc[tr.datetime == date, sid] = target_value
        
    result = pd.DataFrame({
        'datetime x segment_id': np.concatenate([[" x ".join([str(dt), str(sid)]) for sid in segment_ids] 
                                                 for dt in tr['datetime']]),
        'datetime': np.concatenate([[str(dt) for sid in segment_ids] for dt in tr['datetime']]),
        'segment_id': np.concatenate([[str(sid) for sid in segment_ids] for dt in tr['datetime']]),
        'target': tr[segment_ids].values.flatten()
    })

    if test_segment_ids is not None:
        result = result.loc[result['segment_id'].isin(test_segment_ids)]

    result['road_id'] = result['segment_id'].str.split('_').str[0].astype(int)
    result['road_km'] = result['segment_id'].str.split('_').str[1].astype(int)

    return result

def add_long_lat(df: pd.DataFrame,
                 geo_data: pd.DataFrame) -> pd.DataFrame:

    geo_data = geo_data.dropna(subset=['lat_geoc', 'lon_geoc'], how='all')
    geo_data = coord_km(geo_data)
    geo_data = geo_data[['road_id', 'road_km', 'lat_long']]

    df = pd.merge(df, geo_data, on=["road_id", "road_km"], how='left')
    df['segment_lat'], df['segment_long'] = df['lat_long'].str

    return df

def add_meteo_stations(train: pd.DataFrame,
                       meteo: pd.DataFrame):
    meteostations_list = stations_list(meteo)
    train['station'] = train['lat_long'].map(functools.partial(stat_km, stat_list=meteostations_list))
    train['station'] = train['station'].where(pd.notnull(train['station']), np.nan)

    return train

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date.astype(str)
    df['s_hour'] = round(np.sin(2 * np.pi * df['hour'] / 24), 3)
    df['c_hour'] = round(np.cos(2 * np.pi * df['hour'] / 24), 3)
    df['s_month'] = round(np.sin(2 * np.pi * df['month'] / 12), 3)
    df['c_month'] = round(np.cos(2 * np.pi * df['month'] / 12), 3)
    return df

def length_on_segment(t):
    if np.floor(t[0]) == np.floor(t[1]):
        res = t[1] - t[0]
    elif np.floor(t[0]) == t[2]:
        res = t[2] + 1 - t[0]
    else:
        res = t[1] - t[2]

    if res > 1: res = 1
    if res == 0: res = t[3]/1000

    return res

def preprocess_meteo(df: pd.DataFrame) -> pd.DataFrame:
    # вручную добавляем координаты для станций 'MOSKBAL' и 'MONCHEG', взятые из других датасетов Росгидромета
    df.loc[df['station']=='MOSKBAL', 'lat'] = 55.8 
    df.loc[df['station']=='MOSKBAL', 'lon'] = 37.5
    df.loc[df['station']=='MONCHEG', 'lat'] = 67.9 
    df.loc[df['station']=='MONCHEG', 'lon'] = 32.9
    df = coord_merge(df)
    return df

def preprocess_crash_parts(df: pd.DataFrame) -> pd.DataFrame:
    df['crash_type'] = df['length']
    df.loc[df['crash_type'] > 3, 'crash_type'] = 0
    df.loc[df['length'] <= 3, 'length'] = 0

    df['length_on_segment'] = df[['avuch_start', 'avuch_end', 'road_km', 'length']].apply(lambda t: length_on_segment(t), axis=1)
    df.rename(columns={'length': 'remuch_length',
                       'length_on_segment': 'remuch_length_on_segment'}, inplace=True)
    
    return df

def preprocess_repair(df: pd.DataFrame) -> pd.DataFrame:
    df['length_on_segment'] = df[['remuch_start', 'remuch_end', 'road_km', 'length']].apply(lambda t: length_on_segment(t), axis=1)
    
    df.loc[df['year_span'].str.contains('-'), 'year_span'] = df['year_span'].str.split('-')
    df = df.explode('year_span')
    
    df['datetime'] = pd.to_datetime(df['year_span'] + df['datetime'].dt.strftime('%m%d'), format='%Y-%m-%d')

    df.rename(columns={'length': 'avuch_length',
                       'length_on_segment': 'avuch_length_on_segment'}, inplace=True)
    return df
