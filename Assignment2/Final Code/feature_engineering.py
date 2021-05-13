import pandas as pd
import numpy as np
from tqdm import tqdm
import holidays
import pickle
import calendar
from collections import defaultdict
from datetime import datetime

DATA_PATH_DOMINIC = 'C:/Users/doist/OneDrive/Documenten/Business Analytics/Master/Year 1/Data Mining Techniques/Assignment 2/Data/'
DATA_PATH_TITUS = '../2nd-assignment-dmt-2021/'

USE_DOMINIC = False
DATA_PATH = DATA_PATH_DOMINIC if USE_DOMINIC else DATA_PATH_TITUS


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    drop_cols = {'visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score', 'gross_bookings_usd',
                 'position'}
    for i in range(1, 9):
        drop_cols |= {f'comp{i}_rate', f'comp{i}_inv', f'comp{i}_rate_percent_diff'}

    df.drop(columns=list(cols & drop_cols), inplace=True)

    print(df.columns)

    return df


def load_data(filename: str) -> pd.DataFrame:

    df = pd.read_csv(DATA_PATH+filename, parse_dates=['date_time'])

    return df


def impute_prop_review_score(df: pd.DataFrame) -> pd.DataFrame:
    df['prop_review_score'] = df['prop_review_score'].fillna(0)
    return df


def impute_orig_dest_distance(df: pd.DataFrame, is_train_set=False) -> pd.DataFrame:
    if is_train_set:
        df_country_dist = df[['visitor_location_country_id', 'prop_country_id', 'orig_destination_distance']]
        df_country_dist = df_country_dist.set_index(['visitor_location_country_id', 'prop_country_id']).sort_index().dropna()

        combin_dist = df_country_dist.groupby(['visitor_location_country_id', 'prop_country_id'])['orig_destination_distance'].apply(np.array)

        # concat (a, b) and (b, a) distances:
        c2c_arrays = dict()
        for i, (c1, c2) in enumerate(combin_dist.index):
            vDist = combin_dist.values[i]

            if (c2, c1) in c2c_arrays:
                c2c_arrays[(c2, c1)] = np.append(c2c_arrays[(c2, c1)], vDist)
                continue

            c2c_arrays[c1, c2] = vDist

        c2c_median = [dict(c1=c1, c2=c2, median_dist=np.median(vDist)) for (c1, c2), vDist in c2c_arrays.items()]
        df_c2c_median = pd.DataFrame(c2c_median)

        # concat df with swapped c1, c2 to accomodate simple merge
        df_c2c_median2 = pd.concat([df_c2c_median, df_c2c_median.rename(columns={'c1': 'c2', 'c2': 'c1'})],
                                   axis=0).drop_duplicates()
        df_c2c_median2.to_csv('orig_destination_distance_impute.csv', index=False)

    try:
        df_c2c_median2 = pd.read_csv('orig_destination_distance_impute.csv')
    except FileNotFoundError:
        raise FileNotFoundError("No imputation data found, please run on train set with is_train_set=True to store needed data.")

    df_joint_dist = pd.merge(df, df_c2c_median2, left_on=['visitor_location_country_id', 'prop_country_id'],
                             right_on=['c1', 'c2'], how='left')

    nan_mask = df_joint_dist['orig_destination_distance'].isnull()
    df_joint_dist.loc[nan_mask, 'orig_destination_distance'] = df_joint_dist.loc[nan_mask, 'median_dist']

    missing = df_joint_dist['orig_destination_distance'].isnull().sum() / df_joint_dist.shape[0]
    print(missing)

    median_of_median = df_c2c_median2['median_dist'].median()
    df_joint_dist['orig_destination_distance'].fillna(median_of_median, inplace=True)

    df = df_joint_dist.drop(columns=df_c2c_median2.columns)

    return df


def impute_prop_location_score2(df: pd.DataFrame, is_train_set: bool = False) -> pd.DataFrame:
    if is_train_set:
        # grouped by 'srch_destination_id' (city) would be more precise, but still has some NaN's
        country_score2 = df.groupby('prop_country_id')['prop_location_score2'].quantile(0.25)

        country_score2.to_csv('prop_location_score2_impute.csv')

    try:
        country_score2 = pd.read_csv('prop_location_score2_impute.csv')
    except FileNotFoundError:
        raise FileNotFoundError("No imputation data found, please run on train set with is_train_set=True to store needed data.")

    score2_country_join = pd.merge(df, country_score2, on='prop_country_id', how='left')['prop_location_score2_y']
    print(f"Still {score2_country_join.isnull().sum()} unsolved NaN's")

    # Fill left-overs
    score2_country_join.fillna(country_score2['prop_location_score2'].mean(), inplace=True)
    print(f"Still {score2_country_join.isnull().sum()} unsolved NaN's")

    nan_mask = df['prop_location_score2'].isnull()
    df.loc[nan_mask, 'prop_location_score2'] = score2_country_join[nan_mask]

    return df


def windsor_price_usd(df: pd.DataFrame, alpha: float = 0.01, is_train_set: bool = False) -> pd.DataFrame:
    if is_train_set:
        store_dict = dict(lower_quantile=df['price_usd'].quantile(alpha),
                          upper_quantile=df['price_usd'].quantile(1 - alpha))
        with open('price_usd_windsor.pickle', 'wb') as handle:
            pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        with open('price_usd_windsor.pickle', 'rb') as handle:
            store_dict = pickle.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            "No imputation data found, please run on train set with is_train_set=True to store needed data.")

    price_usd_winsorized = df['price_usd'].clip(lower=store_dict['lower_quantile'],
                                                upper=store_dict['upper_quantile'])

    outliers = df.loc[df['price_usd'] != price_usd_winsorized, 'price_usd']
    df['price_usd'] = price_usd_winsorized

    return df


def mine_date_features(dt_series: pd.Series, prefix: str = '') -> pd.DataFrame:
    df = pd.DataFrame()
    df['weekday'] = dt_series.dt.weekday
    df['monthday'] = dt_series.dt.day
    df['month'] = dt_series.dt.month
    df['week'] = dt_series.dt.week
    df['year'] = dt_series.dt.year

    df.columns = list(map(lambda s: prefix + s, df.columns.tolist()))
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'], unit='D')  # speedup with vectorize?,
    df['checkout_date'] = df['checkin_date'] + pd.to_timedelta(df['srch_length_of_stay'], unit='D')

    df_srchdate_features = mine_date_features(df['date_time'], prefix='srchdate_')
    df_srchdate_features['srchdate_hour'] = df['date_time'].dt.hour

    df_checkin_features = mine_date_features(df['checkin_date'], prefix='checkin_')
    df_checkout_features = mine_date_features(df['checkout_date'], prefix='checkout_')

    df = pd.concat([df, df_srchdate_features, df_checkin_features, df_checkout_features], axis=1)

    return df


def dayofweek_count(dt_start: datetime, dt_end: datetime) -> pd.Series:
    dt_range = pd.date_range(dt_start, dt_end, freq='d')

    cnt = {day: 0 for day in calendar.day_name}
    for date in dt_range:
        cnt[date.day_name()] += 1
    return pd.Series(cnt)


def booking_contains_holidays(dt_start: datetime, dt_end: datetime, holidays) -> pd.Series:
    dt_range = [dt.date() for dt in pd.date_range(dt_start, dt_end, freq='D')]

    contained = defaultdict(int)
    for dt_holiday, name_holiday in holidays.items():
        if contained[name_holiday]: # prevents override of holiday from another year
            continue
        contained[name_holiday] = int(dt_holiday in dt_range)

    return pd.Series(contained)


def add_srch_features(df: pd.DataFrame) -> pd.DataFrame:
    srch_grouped = df.groupby('srch_id').first()

    srch_window = pd.DataFrame(srch_grouped[['checkin_date', 'checkout_date']])
    srch_daycounts = srch_window.apply(lambda df: dayofweek_count(df['checkin_date'], df['checkout_date']), axis=1)
    df = pd.merge(df, srch_daycounts, on='srch_id', how='left')

    us_holidays = holidays.UnitedStates(years=[2012, 2013, 2014, 2015, 2016])  # easter is missing but fuck that
    srch_holidays = srch_window.apply(lambda df: booking_contains_holidays(df['checkin_date'], df['checkout_date'], us_holidays), axis=1)

    df = pd.merge(df, srch_holidays, on='srch_id', how='left')
    return df


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    return percentile_


def add_pricing_features(df: pd.DataFrame, is_train_set: bool = False) -> pd.DataFrame:
    df_prop = df[['prop_id', 'srch_id', 'date_time', 'price_usd']].sort_values(['prop_id', 'date_time'])

    if is_train_set:
        df_prop.to_csv('price_usd_history.csv', index=False)
    else:
        try:
            df_prop_hist = pd.read_csv('price_usd_history.csv', parse_dates=['date_time'])
        except FileNotFoundError:
            raise FileNotFoundError("No imputation data found, please run on train set with is_train_set=True to store needed data.")

        df_prop = pd.concat([df_prop_hist, df_prop], axis=0).sort_values(['prop_id', 'date_time'])

    df_prop['last_price'] = df_prop.groupby('prop_id')['price_usd'].shift().fillna(df_prop['price_usd'])  # use same price
    df_prop['diff_last_price'] = (df_prop['price_usd'] - df_prop['last_price']) / df_prop['last_price']
    df_prop['diff_last_price'].fillna(0, inplace=True)

    aggregates = {f'{fn}_price': fn
                  for fn in ['min', 'max', 'median', 'std', 'mean', 'count']}
    aggregates.update({'price_percentile_25%': percentile(25), 'price_percentile_75%': percentile(75)})

    # TODO: Better fill na !!
    df_prop[list(aggregates.keys())] = df_prop.groupby('prop_id')['price_usd'].expanding().agg(list(aggregates.values())).fillna(0).values

    diff_ignore = ['std', 'count']
    for col_name, fn in aggregates.items():
        if fn in diff_ignore:  # doesn't make sense
            continue

        df_prop[f'diff_{col_name}'] = (df_prop['price_usd'] - df_prop[col_name]) / df_prop[col_name]
        df_prop[f'diff_{col_name}'].fillna(0, inplace=True)

    df_prop.replace(np.inf, 0, inplace=True)
    # drop price_usd to avoid duplicates
    df_prop.drop(columns=['price_usd'], inplace=True)

    df = pd.merge(df, df_prop, on=['srch_id', 'prop_id', 'date_time'], how='left')
    return df


def create_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df['label'] = 0
    book_indices = df['booking_bool'] == 1
    click_indices = df['click_bool'] == 1

    df['label'].loc[click_indices] = 1
    df['label'].loc[book_indices] = 5

    return df


def feature_engineer(file_name: str, is_train_set: bool) -> pd.DataFrame:
    df = load_data(file_name)
    df = drop_columns(df)

    df = impute_prop_review_score(df)
    df = impute_orig_dest_distance(df, is_train_set=is_train_set)
    df = impute_prop_location_score2(df, is_train_set=is_train_set)
    df = windsor_price_usd(df, is_train_set=is_train_set)
    df = add_date_features(df)
    df = add_srch_features(df)
    df = add_pricing_features(df, is_train_set=is_train_set)

    if is_train_set:
        df = create_label_column(df)
        df.drop(columns=['click_bool', 'booking_bool'], inplace=True)
    df.drop(columns=['date_time', 'checkin_date', 'checkout_date'], inplace=True)

    print(df.columns)
    print(df.describe())

    describ = df.describe()

    dataset_name = file_name.split('.')[0]
    df.to_pickle(f'{dataset_name}.pkl')

    return df

#
# import gc
# import time

# df = pd.read_csv('../2nd-assignment-dmt-2021/training_set_VU_DM.csv')
# df = create_label_column(df)
#
# label = df['label'].to_numpy()
# del df
#
# gc.collect()
# time.sleep(5)
#
# df2 = pd.read_pickle('training_set_VU_DM2.pkl')
# df2['label'] = label
#
# df2.to_pickle('training_set_VU_DM.pkl')


if __name__ == '__main__':
    feature_engineer('training_set_VU_DM.csv', is_train_set=True)
    feature_engineer('test_set_VU_DM.csv', is_train_set=False)


# TODO: ADD DOCSTRINGS
# TODO: Add type hints




