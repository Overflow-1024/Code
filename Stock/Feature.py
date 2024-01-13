import pandas as pd
import numpy as np
import os

from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count


# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
sns.set_context("talk")
style.use('seaborn-colorblind')

pd.set_option('max_columns', 300)
pd.set_option('display.max_columns', None)

input_path = '../input/optiver-realized-volatility-prediction'
model_path = '/kaggle/working'
data_path = '/kaggle/working'


def load_book(stock_id, data_type):
    """
    load parquet book data for given stock_id
    """
    df_book = pd.read_parquet(os.path.join(input_path, f'book_{data_type}.parquet/stock_id={stock_id}'))
    df_book['stock_id'] = stock_id
    df_book['stock_id'] = df_book['stock_id'].astype(np.int8)

    # sort
    df_book = df_book.sort_values(by=['stock_id', 'time_id'])

    return df_book


def load_trade(stock_id, data_type):
    """
    load parquet trade data for given stock_id
    """
    df_trade = pd.read_parquet(os.path.join(input_path, f'trade_{data_type}.parquet/stock_id={stock_id}'))
    df_trade['stock_id'] = stock_id
    df_trade['stock_id'] = df_trade['stock_id'].astype(np.int8)

    # sort
    df_trade = df_trade.sort_values(by=['stock_id', 'time_id'])

    return df_trade


# 用3σ准则去除异常值
def remove_anomaly(df_bucket_all):

    stock_list = df_bucket_all['stock_id'].unique()
    df_bucket_all_remove = pd.DataFrame()

    for stock_id in stock_list:
        df_bucket = df_bucket_all[df_bucket_all['stock_id'] == stock_id]
        mean = df_bucket['target'].mean()
        std = df_bucket['target'].std()

        low_bound = mean - 4 * std
        high_bound = mean + 4 * std

        df_bucket.drop(df_bucket[(df_bucket['target'] > high_bound) | (df_bucket['target'] < low_bound)].index, inplace=True)
        df_bucket_all_remove = pd.concat([df_bucket_all_remove, df_bucket], axis=0)

    per = round((df_bucket_all.shape[0] - df_bucket_all_remove.shape[0]) / df_bucket_all.shape[0] * 100, 3)
    print(f"去除异常值之前样本数：{df_bucket_all.shape[0]}  去除异常值之后样本数：{df_bucket_all_remove.shape[0]}  异常值比例：{per}%")

    return df_bucket_all_remove


# 计算log_return
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


# 计算volatility
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


# 计算log_return绝对值之和
def return_sum(series_log_return):
    return np.sum(np.abs(series_log_return))


# 计算四分位差
def interquartile(series_price):
    return np.percentile(series_price, 75) - np.percentile(series_price, 25)


# 提取粒度为second的特征
def fe_second(book, trade):
    book['wap1'] = (book['bid_price1'] * book['ask_size1'] + book['ask_price1'] * book['bid_size1']) / (
            book['bid_size1'] + book['ask_size1'])

    book['wap2'] = (book['bid_price2'] * book['ask_size2'] + book['ask_price2'] * book['bid_size2']) / (
            book['bid_size2'] + book['ask_size2'])

    book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / (book['ask_price1'] + book['bid_price1'])

    book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])

    book['log_return1'] = book.groupby(['time_id'])['wap1'].apply(log_return)

    book['log_return2'] = book.groupby(['time_id'])['wap2'].apply(log_return)

    trade['log_return'] = trade.groupby(['time_id'])['price'].apply(log_return)

    # 计算log_return的时候由于用的是diff()，第一行没有值，这里要处理一下NaN
    book.fillna(method='backfill', inplace=True)
    trade.fillna(method='backfill', inplace=True)

    return book, trade


# 提取粒度为bucket的特征
def fe_bucket(book, trade, stock_id):
    # features
    book_features = ['wap1', 'wap2', 'price_spread', 'total_volume', 'log_return1', 'log_return2']
    trade_features = ['price', 'size', 'log_return']
    id_col = ['stock_id', 'time_id']

    book_agg_features = {
        'wap1': ['mean', 'std', interquartile],
        'price_spread': ['sum', 'mean'],
        'total_volume': ['sum'],
        'seconds_in_bucket': ['count'],
        'log_return1': ['std', return_sum, realized_volatility],
        'log_return2': ['std', return_sum, realized_volatility]
    }
    trade_agg_features = {
        'price': ['mean', 'std', interquartile],
        'size': ['sum'],
        'seconds_in_bucket': ['count'],
        'log_return': ['std', return_sum, realized_volatility]
    }

    def fe_window(second, suffix):

        # book agg features
        book_bucket_window = book[book['seconds_in_bucket'] >= second].groupby(['time_id']).agg(
            book_agg_features).reset_index()

        book_bucket_window.columns = ['_'.join(col) for col in book_bucket_window.columns]
        book_bucket_window.rename(columns={'time_id_': 'time_id',
                                           'seconds_in_bucket_count': f'book_bucket_seconds'}, inplace=True)

        if suffix:
            book_bucket_window = book_bucket_window.add_suffix('_' + str(second))
            book_bucket_window.rename(columns={f'time_id_{second}': 'time_id'}, inplace=True)

        # trade agg features
        trade_bucket_window = trade[trade['seconds_in_bucket'] >= second].groupby(['time_id']).agg(
            trade_agg_features).reset_index()

        trade_bucket_window.columns = ['_'.join(col) for col in trade_bucket_window.columns]
        trade_bucket_window.rename(columns={'time_id_': 'time_id',
                                            'seconds_in_bucket_count': f'trade_bucket_seconds'}, inplace=True)

        if suffix:
            trade_bucket_window = trade_bucket_window.add_suffix('_' + str(second))
            trade_bucket_window.rename(columns={f'time_id_{second}': 'time_id'}, inplace=True)

        return book_bucket_window, trade_bucket_window

    # Get the stats for different windows
    book_bucket_window, trade_bucket_window = fe_window(second=0, suffix=False)
    book_bucket_window150, trade_bucket_window150 = fe_window(second=150, suffix=True)
    book_bucket_window300, trade_bucket_window300 = fe_window(second=300, suffix=True)
    book_bucket_window450, trade_bucket_window450 = fe_window(second=450, suffix=True)

    book_bucket = pd.merge(book_bucket_window, book_bucket_window150, how='left', on=['time_id'])
    book_bucket = pd.merge(book_bucket, book_bucket_window300, how='left', on=['time_id'])
    book_bucket = pd.merge(book_bucket, book_bucket_window450, how='left', on=['time_id'])

    trade_bucket = pd.merge(trade_bucket_window, trade_bucket_window150, how='left', on=['time_id'])
    trade_bucket = pd.merge(trade_bucket, trade_bucket_window300, how='left', on=['time_id'])
    trade_bucket = pd.merge(trade_bucket, trade_bucket_window450, how='left', on=['time_id'])

    book_bucket['stock_id'] = stock_id
    book_bucket['stock_id'] = book_bucket['stock_id'].astype(np.int8)
    trade_bucket['stock_id'] = stock_id
    trade_bucket['stock_id'] = trade_bucket['stock_id'].astype(np.int8)

    #     # book agg features
    #     book_bucket = book.groupby(['stock_id', 'time_id']).agg(book_agg_features).reset_index()
    #     book_bucket.columns = ['_'.join(col) for col in book_bucket.columns]
    #     book_bucket.rename(columns={'stock_id_': 'stock_id',
    #                                 'time_id_': 'time_id',
    #                                 'seconds_in_bucket_count': 'book_bucket_seconds'}, inplace=True)

    #     # trade agg features
    #     trade_bucket = trade.groupby(['stock_id', 'time_id']).agg(trade_agg_features).reset_index()
    #     trade_bucket.columns = ['_'.join(col) for col in trade_bucket.columns]
    #     trade_bucket.rename(columns={'stock_id_': 'stock_id',
    #                                  'time_id_': 'time_id',
    #                                  'seconds_in_bucket_count': 'trade_bucket_seconds'}, inplace=True)

    return book_bucket, trade_bucket


# 每只股票的特征提取_train
def fe_per_stock_train(stock_id=0):
    """
    load orderbook and trade data for the given stock_id and merge

    """

    df_book = load_book(stock_id, 'train')
    df_trade = load_trade(stock_id, 'train')

    # sort by time
    df_book = df_book.sort_values(by=['time_id', 'seconds_in_bucket'])
    df_trade = df_trade.sort_values(by=['time_id', 'seconds_in_bucket'])

    df_book_second, df_trade_second = fe_second(df_book, df_trade)

    df_book_bucket, df_trade_bucket = fe_bucket(df_book_second, df_trade_second, stock_id)

    df_bucket = pd.merge(df_book_bucket, df_trade_bucket, how='outer', on=['stock_id', 'time_id'])
    df_bucket.fillna(method='ffill', inplace=True)

    return df_bucket


# 每只股票的特征提取_test
def fe_per_stock_test(stock_id=0):
    """
    load orderbook and trade data for the given stock_id and merge

    """

    df_book = load_book(stock_id, 'test')
    df_trade = load_trade(stock_id, 'test')

    # sort by time
    df_book = df_book.sort_values(by=['time_id', 'seconds_in_bucket'])
    df_trade = df_trade.sort_values(by=['time_id', 'seconds_in_bucket'])

    df_book_second, df_trade_second = fe_second(df_book, df_trade)

    df_book_bucket, df_trade_bucket = fe_bucket(df_book_second, df_trade_second, stock_id)

    df_bucket = pd.merge(df_book_bucket, df_trade_bucket, how='outer', on=['stock_id', 'time_id'])
    df_bucket.fillna(method='ffill', inplace=True)

    return df_bucket


# 提取粒度为stock和time的特征
def fe_stock_and_time(df_bucket_all, data_type):
    # feature engineering agg by stock_id
    vol_features = [f for f in df_bucket_all.columns if ('volatility' in f)]

    if data_type == 'train':
        # train模式下 计算stock特征并保存为文件
        df_stock = df_bucket_all.groupby('stock_id')[vol_features].agg(['mean']).reset_index()
        # fix column names
        df_stock.columns = ['_'.join(i) for i in df_stock.columns]
        df_stock.rename(columns={'stock_id_': 'stock_id'}, inplace=True)
        df_stock.columns = ['stock_id'] + [f'{f}_stock' for f in df_stock.columns[1:]]

        df_stock.to_parquet(f'{data_path}/df_stock_feature.parquet')
    else:
        # test模式下 从文件读取stock特征
        df_stock = pd.read_parquet(f'{data_path}/df_stock_feature.parquet')

    # feature engineering agg by time_id
    df_time = df_bucket_all.groupby('time_id')[vol_features].agg(['mean']).reset_index()
    # fix column names
    df_time.columns = ['_'.join(i) for i in df_time.columns]
    df_time.rename(columns={'time_id_': 'time_id'}, inplace=True)
    df_time.columns = ['time_id'] + [f'{f}_time' for f in df_time.columns[1:]]

    return df_stock, df_time


# 所有股票的特征提取
def fe_all_stock(data_type):
    # all book data feature engineering
    stocks_dir_list = os.listdir(os.path.join(input_path, f'book_{data_type}.parquet'))
    stock_list = [int(i.split('=')[-1]) for i in stocks_dir_list]

    #     if data_type == 'train':
    #         stock_list = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29]
    #     else:
    #         stock_list = [0]

    # feature engineering agg by stock_id x time_id
    with Pool(cpu_count()) as p:
        if data_type == 'train':
            df_bucket_list = list(tqdm(p.imap(fe_per_stock_train, stock_list), total=len(stock_list)))
        elif data_type == 'test':
            df_bucket_list = list(tqdm(p.imap(fe_per_stock_test, stock_list), total=len(stock_list)))

    df_bucket_all = pd.concat(df_bucket_list)

    # 合并 label， bucket特征 2个DataFrame，然后去除异常值（训练集）
    target_path = os.path.join(input_path, f'{data_type}.csv')
    target = pd.read_csv(target_path)

    df_bucket_all = pd.merge(target, df_bucket_all, how='left', on=['stock_id', 'time_id'])

    if data_type == 'train':
        df_bucket_all = remove_anomaly(df_bucket_all)

    df_stock, df_time = fe_stock_and_time(df_bucket_all, data_type)

    # 合并bucket特征，stock特征，time特征 3个DataFrame

    df_feature = pd.merge(df_bucket_all, df_stock, how='left', on='stock_id')
    df_feature = pd.merge(df_feature, df_time, how='left', on='time_id')
    df_feature = df_feature.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    # make row_id
    df_feature['row_id'] = df_feature['stock_id'].astype(str) + '-' + df_feature['time_id'].astype(str)

    return df_bucket_all, df_stock, df_time, df_feature


df_bucket, df_stock, df_time, df_feature = fe_all_stock('train')
print('train data feature engineering finish')

print(f'df_bucket shape: {df_bucket.shape}')
print(f'df_time shape: {df_time.shape}')
print(f'df_feature shape: {df_feature.shape}')


df_bucket.to_parquet(f'{data_path}/df_bucket_feature.parquet')
df_time.to_parquet(f'{data_path}/df_time_feature.parquet')
df_feature.to_parquet(f'{data_path}/df_feature.parquet')
