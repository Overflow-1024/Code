DEBUG = False
MODE = 'TRAIN'
# MODE = 'INFERENCE'
MODEL_DIR = '../input/optiver-lgb-and-te-baseline'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import pathlib
from tqdm.auto import tqdm
import json
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob
import os
from sklearn import model_selection
import joblib
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

import warnings
warnings.simplefilter('ignore')

pd.get_option("display.max_columns")

class CFG:
    INPUT_DIR = '../input/optiver-realized-volatility-prediction'
    OUTPUT_DIR = './'
    N_SPLITS = 10
    SEED = 46

# Logging is always nice for your experiment:)
def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = init_logger(log_file=f'{CFG.OUTPUT_DIR}/baseline.log')
logger.info(f'Start Logging...')

train = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'train.csv'))
logger.info('Train data: {}'.format(train.shape))

test = pd.read_csv(os.path.join(CFG.INPUT_DIR, 'test.csv'))
logger.info('Test data: {}'.format(test.shape))

train_book_stocks = os.listdir(os.path.join(CFG.INPUT_DIR, 'book_train.parquet'))

if DEBUG:
    logger.info('Debug mode: using 3 stocks only')
    train_book_stocks = train_book_stocks[:3]

logger.info('{:,} train book stocks: {}'.format(len(train_book_stocks), train_book_stocks))


# load stock_id=0
def load_book(stock_id=0, data_type='train'):
    """
    load parquest book data for given stock_id
    """
    book_df = pd.read_parquet(os.path.join(CFG.INPUT_DIR, f'book_{data_type}.parquet/stock_id={stock_id}'))
    book_df['stock_id'] = stock_id
    book_df['stock_id'] = book_df['stock_id'].astype(np.int8)

    return book_df


def load_trade(stock_id=0, data_type='train'):
    """
    load parquest trade data for given stock_id
    """
    trade_df = pd.read_parquet(os.path.join(CFG.INPUT_DIR, f'trade_{data_type}.parquet/stock_id={stock_id}'))
    trade_df['stock_id'] = stock_id
    trade_df['stock_id'] = trade_df['stock_id'].astype(np.int8)

    return trade_df

def fix_jsonerr(df):
    """
    fix json column error for lightgbm
    """
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

def realized_volatility(series_log_return):
    series_log_return = log_return(series_log_return)
    return np.sqrt(np.sum(series_log_return ** 2))


def fe_row(book):
    """
    Feature engineering (just volatility for now) for each row
    """

    # volatility
    for i in [1, 2, ]:
        # wap
        book[f'book_wap{i}'] = (book[f'bid_price{i}'] * book[f'ask_size{i}'] +
                                book[f'ask_price{i}'] * book[f'bid_size{i}']) / (
                                       book[f'bid_size{i}'] + book[f'ask_size{i}'])

    # mean wap
    book['book_wap_mean'] = (book['book_wap1'] + book['book_wap2']) / 2

    # wap diff
    book['book_wap_diff'] = book['book_wap1'] - book['book_wap2']

    # other orderbook features
    book['book_price_spread'] = (book['ask_price1'] - book['bid_price1']) / (book['ask_price1'] + book['bid_price1'])
    book['book_bid_spread'] = book['bid_price1'] - book['bid_price2']
    book['book_ask_spread'] = book['ask_price1'] - book['ask_price2']
    book['book_total_volume'] = book['ask_size1'] + book['ask_size2'] + book['bid_size1'] + book['bid_size2']
    book['book_volume_imbalance'] = (book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2'])

    return book


def fe_agg(book_df):
    """
    feature engineering (aggregation by stock_id x time_id)
    """

    # features
    book_feats = book_df.columns[book_df.columns.str.startswith('book_')].values.tolist()
    trade_feats = ['price', 'size', 'order_count', 'seconds_in_bucket']

    # agg trade features
    trade_df = book_df.groupby(['time_id', 'stock_id'])[trade_feats].agg([
        'sum', 'mean', 'std', 'max', 'min'
    ]).reset_index()

    # agg volatility features
    fe_df = book_df.groupby(['time_id', 'stock_id'])[book_feats].agg([
        realized_volatility
    ]).reset_index()
    fe_df.columns = [" ".join(col).strip() for col in fe_df.columns.values]

    # merge
    fe_df = fe_df.merge(
        trade_df
        , how='left'
        , on=['time_id', 'stock_id']
    )

    return fe_df


def fe_all(book_df):
    """
    perform feature engineerings
    """

    # row-wise feature engineering
    book_df = fe_row(book_df)

    # feature engineering agg by stock_id x time_id
    fe_df = fe_agg(book_df)

    return fe_df


def book_fe_by_stock(stock_id=0):
    """
    load orderbook and trade data for the given stock_id and merge

    """
    # load data
    book_df = load_book(stock_id, 'train')
    trade_df = load_trade(stock_id, 'train')
    book_feats = book_df.columns.values.tolist()

    # merge
    book_df = book_df.merge(
        trade_df
        , how='outer'
        , on=['time_id', 'seconds_in_bucket', 'stock_id']
    )

    # sort by time
    book_df = book_df.sort_values(by=['time_id', 'seconds_in_bucket'])

    # fillna for book_df
    book_df[book_feats] = book_df[book_feats].fillna(method='ffill')

    # feature engineering
    fe_df = fe_all(book_df)
    return fe_df


def book_fe_by_stock_test(stock_id=0):
    """
    same function but for the test

    """
    # load data
    book_df = load_book(stock_id, 'test')
    trade_df = load_trade(stock_id, 'test')
    book_feats = book_df.columns.values.tolist()

    # merge
    book_df = book_df.merge(
        trade_df
        , how='outer'
        , on=['time_id', 'seconds_in_bucket', 'stock_id']
    )

    # sort by time
    book_df = book_df.sort_values(by=['time_id', 'seconds_in_bucket'])

    # fillna for book_df
    book_df[book_feats] = book_df[book_feats].fillna(method='ffill')

    # feature engineering
    fe_df = fe_all(book_df)
    return fe_df


def book_fe_all(stock_ids, data_type='train'):
    """
    Feature engineering with multithread processing
    """
    # feature engineering agg by stock_id x time_id
    with Pool(cpu_count()) as p:
        if data_type == 'train':
            feature_dfs = list(tqdm(p.imap(book_fe_by_stock, stock_ids), total=len(stock_ids)))
        elif data_type == 'test':
            feature_dfs = list(tqdm(p.imap(book_fe_by_stock_test, stock_ids), total=len(stock_ids)))

    fe_df = pd.concat(feature_dfs)

    # feature engineering agg by stock_id
    vol_feats = [f for f in fe_df.columns if ('realized' in f) & ('wap' in f)]
    if data_type == 'train':
        # agg
        stock_df = fe_df.groupby('stock_id')[vol_feats].agg(['mean', 'std', 'max', 'min', ]).reset_index()

        # fix column names
        stock_df.columns = ['stock_id'] + [f'{f}_stock' for f in stock_df.columns.values.tolist()[1:]]
        stock_df = fix_jsonerr(stock_df)

    # feature engineering agg by time_id
    time_df = fe_df.groupby('time_id')[vol_feats].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    time_df.columns = ['time_id'] + [f'{f}_time' for f in time_df.columns.values.tolist()[1:]]

    # merge
    fe_df = fe_df.merge(
        time_df
        , how='left'
        , on='time_id'
    )

    # make sure to fix json error for lighgbm
    fe_df = fix_jsonerr(fe_df)

    # out
    if data_type == 'train':
        return fe_df, stock_df
    elif data_type == 'test':
        return fe_df


if MODE == 'TRAIN':
    # all book data feature engineering
    stock_ids = [int(i.split('=')[-1]) for i in train_book_stocks]
    book_df, stock_df = book_fe_all(stock_ids, data_type='train')

    assert book_df['stock_id'].nunique() > 2
    assert book_df['time_id'].nunique() > 2

    # save stock_df for the test
    stock_df.to_pickle('train_stock_df.pkl')
    logger.info('train stock df saved!')

    # merge
    book_df = book_df.merge(
        stock_df
        , how='left'
        , on='stock_id'
    ).merge(
        train
        , how='left'
        , on=['stock_id', 'time_id']
    ).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    # make row_id
    book_df['row_id'] = book_df['stock_id'].astype(str) + '-' + book_df['time_id'].astype(str)
    book_df.to_pickle('book_df.pkl')

    print(book_df.shape)

# test
test_book_stocks = os.listdir(os.path.join(CFG.INPUT_DIR, 'book_test.parquet'))

logger.info('{:,} test book stocks: {}'.format(len(test_book_stocks), test_book_stocks))

# all book data feature engineering
test_stock_ids = [int(i.split('=')[-1]) for i in test_book_stocks]
test_book_df = book_fe_all(test_stock_ids, data_type='test')

# load stock_df, if inference
if MODE == 'INFERENCE':
    book_df = pd.read_pickle(f'{MODEL_DIR}/book_df.pkl')
    stock_df = pd.read_pickle(f'{MODEL_DIR}/train_stock_df.pkl')

# merge
test_book_df = test.merge(
    stock_df
    , how='left'
    , on='stock_id'
).merge(
    test_book_df
    , how='left'
    , on=['stock_id', 'time_id']
).replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

# make row_id
test_book_df['row_id'] = test_book_df['stock_id'].astype(str) + '-' + test_book_df['time_id'].astype(str)

print(test_book_df.shape)


target = 'target'
drops = [target, 'row_id', 'time_id']
features = [f for f in test_book_df.columns.values.tolist() if (f not in drops) & (test_book_df[f].isna().sum() == 0) & (book_df[f].isna().sum() == 0)]
cats = ['stock_id', ]

logger.info('{:,} features ({:,} categorical): {}'.format(len(features), len(cats), features))


# evaluation metric
def RMSPEMetric(XGBoost=False):

    def RMSPE(yhat, dtrain, XGBoost=XGBoost):

        y = dtrain.get_label()
        elements = ((y - yhat) / y) ** 2
        if XGBoost:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y))), False

    return RMSPE

# LightGBM parameters
params = {
    'n_estimators': 10000,
    'objective': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.01,
    'subsample': 0.72,
    'subsample_freq': 4,
    'feature_fraction': 0.9,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'seed': 46,
    'early_stopping_rounds': 100,
    'verbose': -1
}


def fit_model(params, X_train, y_train, X_test, features=features, cats=[], era='stock_id', fold_type='kfold', n_fold=5,
              seed=42):
    """
    fit model with cross validation
    """

    models = []
    oof_df = X_train[['time_id', 'stock_id', target]].copy()
    oof_df['pred'] = np.nan
    y_preds = np.zeros((len(X_test),))

    if fold_type == 'stratifiedshuffle':
        cv = model_selection.StratifiedShuffleSplit(n_splits=n_fold, random_state=seed)
        kf = cv.split(X_train, X_train[era])
    elif fold_type == 'kfold':
        cv = model_selection.KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        kf = cv.split(X_train, y_train)
    elif fold_type == 'gkf':
        cv = model_selection.GroupKFold(n_splits=n_fold)
        kf = cv.split(X_train, y_train, X_train[era])

    fi_df = pd.DataFrame()
    fi_df['features'] = features
    fi_df['importance'] = 0

    for fold_id, (train_index, valid_index) in tqdm(enumerate(kf)):
        # split
        X_tr = X_train.loc[train_index, features]
        X_val = X_train.loc[valid_index, features]
        y_tr = y_train.loc[train_index]
        y_val = y_train.loc[valid_index]

        # model (note inverse weighting)
        train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=cats, weight=1 / np.power(y_tr, 2))
        val_set = lgb.Dataset(X_val, y_val, categorical_feature=cats, weight=1 / np.power(y_val, 2))
        model = lgb.train(
            params
            , train_set
            , valid_sets=[train_set, val_set]
            , feval=RMSPEMetric()
            , verbose_eval=250
        )

        # feature importance
        fi_df[f'importance_fold{fold_id}'] = model.feature_importance(importance_type="gain")
        fi_df['importance'] += fi_df[f'importance_fold{fold_id}'].values

        # save model
        joblib.dump(model, f'model_fold{fold_id}.pkl')
        logger.debug('model saved!')

        # predict
        oof_df['pred'].iloc[valid_index] = model.predict(X_val)
        y_pred = model.predict(X_test[features])
        y_preds += y_pred / n_fold
        models.append(model)

    return oof_df, y_preds, models, fi_df


if MODE == 'TRAIN':
    oof_df, y_preds, models, fi_df = fit_model(params,
                                          book_df,
                                          book_df[target],
                                          test_book_df,
                                          features=features,
                                          cats=cats,
                                          era='time_id',
                                          fold_type='gkf',
                                          n_fold=CFG.N_SPLITS,
                                          seed=CFG.SEED
                                              )

from sklearn.metrics import r2_score


def rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


if MODE == 'TRAIN':
    oof_df.dropna(inplace=True)
    y_true = oof_df[target].values
    y_pred = oof_df['pred'].values

    oof_df[target].hist(bins=100)
    oof_df['pred'].hist(bins=100)

    R2 = round(r2_score(y_true, y_pred), 3)
    RMSPE = round(rmspe(y_true, y_pred), 3)
    logger.info(f'Performance of the naive prediction: R2 score: {R2}, RMSPE: {RMSPE}')

if MODE == 'TRAIN':
    fi_df = fi_df.sort_values(by='importance', ascending=False)
    fi_df.to_csv('feature_importance.csv', index=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 40))
    sns.barplot(x='importance', y='features', data=fi_df.iloc[:30], ax=ax)
    logger.info(fi_df[['features', 'importance']].iloc[:50].to_markdown())

if MODE == 'INFERENCE':
    """
    used for inference kernel only
    """
    y_preds = np.zeros(len(test_book_df))
    files = glob.glob(f'{MODEL_DIR}/*model*.pkl')
    assert len(files) > 0
    for i, f in enumerate(files):
        model = joblib.load(f)
        y_preds += model.predict(test_book_df[features])
    y_preds /= (i + 1)

test_book_df[target] = y_preds

# test
test_book_df[target] = y_preds

# save the submit file
sub = test_book_df[['row_id', target]]
sub.to_csv('submission.csv', index=False)

logger.info('submitted!')
sub.head()