import pandas as pd
import numpy as np

from typing import Optional, List

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

    result['road_id'] = result['segment_id'].str.split('_').str[0]
    result['road_km'] = result['segment_id'].str.split('_').str[1]

    return result

def preprocess_crash_parts(df: pd.DataFrame) -> pd.DataFrame:
    df['crash_type'] = df['length']
    df.loc[df['crash_type'] > 3, 'crash_type'] = 0
    df.loc[df['length'] <= 3, 'length'] = 0

    
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

    df['length_on_segment'] = df[['avuch_start', 'avuch_end', 'road_km', 'length']].apply(lambda t: length_on_segment(t), axis=1)
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
