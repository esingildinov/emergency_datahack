import pandas as pd
import numpy as np
import functools

from tqdm.auto import tqdm
from typing import Optional, List
from match_station import coord_km, coord_merge, stations_list, stat_km


def add_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    df['segment_id'] = df[['road_id', 'road_km']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df

def meteo_stations_coords(df: pd.DataFrame) -> pd.DataFrame:

    df.loc[df['station']=='MOSKBAL', 'lat'] = 55.8 
    df.loc[df['station']=='MOSKBAL', 'lon'] = 37.5
    df.loc[df['station']=='MONCHEG', 'lat'] = 67.9 
    df.loc[df['station']=='MONCHEG', 'lon'] = 32.9
    df = coord_merge(df)

    return df

def add_meteo_stations(geo: pd.DataFrame,
                       meteo: pd.DataFrame):
    meteostations_list = stations_list(meteo)
    geo['meteo_station'] = geo['lat_long'].map(functools.partial(stat_km, stat_list=meteostations_list))
    geo['meteo_station'] = geo['meteo_station'].where(pd.notnull(geo['meteo_station']), np.nan)

    return geo

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

    train = train[(train.datetime >= start_date) & (train.datetime <= end_date)]

    tr = pd.DataFrame({'datetime': pd.date_range(start_date, end_date, freq="1h")})

    for sid in segment_ids:
        tr[str(sid)] = 0
        events = train.loc[train['segment_id'] == sid]
        datetimes = events.datetime.dt.floor('H')
        dates = datetimes.astype(str).unique()
        for date in dates:
            target_value = train[(train.datetime == date) & (train.segment_id == sid)]['target'].values[0]
            tr.loc[tr.datetime == date, sid] = target_value
    
    train_chunks = []
    for dt in tqdm(tr['datetime']):
        chunk = pd.DataFrame({
            'datetime': [str(dt) for sid in segment_ids],
            'segment_id': [str(sid) for sid in segment_ids],
            'target': tr.loc[tr.datetime==dt, segment_ids].values.flatten()
        })
        train_chunks.append(chunk)
    
    result = pd.concat(train_chunks)

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


def get_traffic_stations(traffic: pd.DataFrame) -> pd.DataFrame:
    traffic_stations = traffic[['station_id', 'road_id', 'road_km', 'latitude', 'longitude']].drop_duplicates()
    traffic_stations = traffic_stations.sort_values(['road_id', 'road_km']).reset_index(drop=True)
    return traffic_stations

def preprocess_geo(geo: pd.DataFrame,
                   traffic: pd.DataFrame,
                   meteo: pd.DataFrame) -> pd.DataFrame:

    geo['lat_geoc'] = geo['lat_geoc'].fillna(geo['lat_glonass'])
    geo['lon_geoc'] = geo['lon_geoc'].fillna(geo['lon_glonass'])

    geo['lat_glonass'] = geo['lat_glonass'].fillna(geo['lat_geoc'])
    geo['lon_glonass'] = geo['lon_glonass'].fillna(geo['lon_geoc'])

    traffic_stations = get_traffic_stations(traffic)
    geo = geo.merge(traffic_stations[['station_id', 'road_id', 'road_km']], on=['road_id', 'road_km'], how='left')

    geo.loc[geo.road_id == 9, 'station_id_match'] = geo.loc[geo.road_id == 9, 'station_id'].interpolate(method='nearest').ffill().bfill()
    geo.loc[geo.road_id == 14, 'station_id_match'] = geo.loc[geo.road_id == 14, 'station_id'].interpolate(method='nearest').ffill().bfill()
    
    geo = geo.merge(traffic_stations[['station_id', 'latitude', 'longitude']], 
                              left_on='station_id_match', right_on='station_id',
                              how='left')

    geo['dist_nearest_traffic_station'] = np.sqrt((geo['latitude'] - geo['lat_geoc'])**2 +
                                                  (geo['longitude'] - geo['lon_geoc'])**2)

    geo.drop(columns=['km_name', 'station_id_x', 'station_id_y', 'latitude', 'longitude'], inplace=True)
    geo.rename(columns={'station_id_match': 'station_id'}, inplace=True)

    geo = coord_km(geo)

    geo['lat'], geo['lon'] = geo['lat_long'].str
    geo = geo[~(geo.lat.isnull() | geo.lon.isnull())].reset_index(drop=True)

    geo.drop(columns=['Широта', 'Долгота', 'lat_glonass', 'lon_glonass'], inplace=True)

    geo = add_meteo_stations(geo,
                             meteo_stations_coords(meteo))

    return geo

def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['road_id'] != 5]
    df = df[df['target'] != 3]
    
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = add_segment_id(df)

    df = df[['datetime', 'road_id', 'road_km', 'segment_id', 'target']]

    return df

def preprocess_meteo(df: pd.DataFrame) -> pd.DataFrame:
    # вручную добавляем координаты для станций 'MOSKBAL' и 'MONCHEG', взятые из других датасетов Росгидромета
   
    df['weather_range'] = pd.factorize(df['weather_range'], sort=True)[0]
    df['weather_on_measure'] = pd.factorize(df['weather_on_measure'], sort=True)[0]
    df['vsp_mean'] = df[['vsp_1', 'vsp_2', 'vsp_3']].mean(axis=1)
    df['precip'].fillna(0, inplace=True)

    q_cols = [c for c in df.columns if '_q' in c]
    q_values_bin = {'Значение элемента отсутствует': 0,
                    'Значение элемента отсутствует.': 0,
                    'Значение элемента достоверно': 1, 
                    'Значение элемента достоверно и восстановлено автоматически': 1,
                    'Значение элемента достоверно и восстановлено вручную': 1,         
                    'Значение элемента забраковано на станции': 2}

    df[q_cols] = df[q_cols].replace(q_values_bin)

    meteo_stations = []

    na_linear_cols = ['vsp_mean', 'visib', 'avg_wind', 'temp_on_measure', 'precip', 'humidity', 'pressure']
    na_nearest_cols = [c for c in df.columns if c not in na_linear_cols + ['measure_dt']]
    
    for station_name, station_df in df.groupby('station'):
       
        station_df_resampled = station_df.sort_values('measure_dt')
        station_df_resampled.set_index('measure_dt', inplace=True)
        station_df_resampled = station_df_resampled.asfreq(freq='1H')
        station_df_resampled['station'] = station_name
        
        station_df_resampled[na_nearest_cols] = station_df_resampled[na_nearest_cols].interpolate(method='nearest')
        station_df_resampled[na_linear_cols] = station_df_resampled[na_linear_cols].interpolate(method='linear')

        for col in na_linear_cols:
            station_df_resampled[col+'_diff'] = station_df_resampled[col].diff()
        for col in ['weather_range', 'weather_on_measure']:
            station_df_resampled[col+'_diff'] = station_df_resampled[col].diff(3)

        station_df_resampled['weather_range_diff'] = (station_df_resampled['weather_range_diff'] != 0).astype(int)
        station_df_resampled['weather_on_measure_diff'] = (station_df_resampled['weather_range_diff'] != 0).astype(int)

        meteo_stations.append(station_df_resampled)

    result = pd.concat(meteo_stations).reset_index()
    result.drop(columns=['lat_long', 'road_id'], inplace=True)

    return result

def preprocess_crash_parts(df: pd.DataFrame) -> pd.DataFrame:
    df['crash_type'] = df['length']
    df.loc[df['crash_type'] > 3, 'crash_type'] = 0
    df.loc[df['length'] <= 3, 'length'] = 0

    df['length_on_segment'] = df[['avuch_start', 'avuch_end', 'road_km', 'length']].apply(lambda t: length_on_segment(t), axis=1)
    df.rename(columns={'length': 'remuch_length',
                       'length_on_segment': 'remuch_length_on_segment'}, inplace=True)
    
    return df

def preprocess_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = df['datetime'].dt.round('H')

    groupby_cols = ['datetime', 'road_id', 'road_km', 'station_id', 'direction']
    df = df.groupby(groupby_cols).agg({'volume': 'sum',
                                        'speed': 'mean',
                                        'occupancy': 'mean',
                                        'lane_count': 'max'}).reset_index()
    
    result = pd.pivot_table(df, values=['volume', 'speed', 'occupancy'], 
                            index=['datetime', 'station_id', 'lane_count'], 
                            columns=['direction']).reset_index()

    result.columns = result.columns.map('_'.join).str.strip('_')

    result.set_index('datetime', inplace=True)

    cols_nan = ['occupancy_backward', 'occupancy_forward', 'speed_backward', 'speed_forward', 'volume_backward', 'volume_forward'] 
    for col in cols_nan:
        result[col] = result[col].fillna(result.groupby(['station_id', result.index.weekday, result.index.hour])[col].transform('mean'))
        result[col] = result[col].fillna(result.groupby(['station_id', result.index.hour])[col].transform('mean'))

    result.reset_index(inplace=True)
    result.dropna(inplace=True)

    return result

def preprocess_repair(df: pd.DataFrame) -> pd.DataFrame:
    df['length_on_segment'] = df[['remuch_start', 'remuch_end', 'road_km', 'length']].apply(lambda t: length_on_segment(t), axis=1)
    
    df.loc[df['repair_period'].str.contains('-'), 'repair_period'] = df['repair_period'].str.split('-')
    df = df.explode('repair_period')
    
    df['datetime'] = pd.to_datetime(df['repair_period'] + df['datetime'].dt.strftime('%m%d'), format='%Y-%m-%d')
    df['price_per_km']

    df.rename(columns={'length': 'avuch_length',
                       'length_on_segment': 'avuch_length_on_segment'}, inplace=True)
    return df
