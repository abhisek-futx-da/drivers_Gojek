import pandas as pd
from haversine import haversine

from src.utils.time import robust_hour_of_iso_date




def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
   
   
    df = df.copy()
    df = df.sort_values(['driver_id', 'event_timestamp']).reset_index(drop=True)
    
   
    if 'participant_status' in df.columns:
        df['_is_completed'] = (df['participant_status'] == 'ACCEPTED').astype(int)
    elif 'is_completed' in df.columns:
        df['_is_completed'] = df['is_completed'].astype(int)
    else:
        df['driver_historical_completed_bookings'] = 0
        return df
    

    df['driver_historical_completed_bookings'] = (
        df.groupby('driver_id')['_is_completed']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    
    
    df['driver_historical_completed_bookings'] = df['driver_historical_completed_bookings'].fillna(0).astype(int)
    
   
    df = df.drop(columns=['_is_completed'], errors='ignore')
    
    return df


