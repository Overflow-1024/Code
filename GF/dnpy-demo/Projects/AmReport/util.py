import pandas as pd
import datetime as dt
from pymongo import DESCENDING


def query_table(db_connect, cltName, sort_k, datetime):
    if cltName == u'NCD':
        res = db_connect.query(db='DNPY_AM_report', collection=cltName, flt={sort_k: {'$lte': datetime}}, sort_key='')
        res.index = res[sort_k]
        res.index.name = 'date'
    else:
        res = db_connect.query(db='DNPY_AM_report', collection=cltName, flt={sort_k: {'$lte': datetime}}, sort_key=sort_k,
                               sort_direction=DESCENDING)
        res.set_index(sort_k, inplace=True)

    return res


# -----获取日期
def getNmonths(date, n, unit='m'):
    if unit == 'm':
        res = pd.date_range(end=date, periods=2, freq='%dMS' % n)[0]
    else:
        res = pd.date_range(end=date, periods=2, freq='%dAS' % n)[0]
    return res


# 解析描述变化的词
def parse_word(x, words, rev=False):
    if rev:
        res = words[1] if x['sn'] == 1 else words[0]
    else:
        res = words[0] if x['sn'] == 1 else words[1]
    return res


def weekdata(d, mk_up=u'增加', mk_dw=u'减少'):
    d = d.dropna()
    res = {'tw_dt': d.index[0].date(), 'tw': d.values[0], 'lw': d.values[1],
           'mk': mk_up if d.values[0] >= d.values[1] else mk_dw, 'dif': abs(d.values[0] - d.values[1]),
           'dif_r': abs((d.values[0] / d.values[1] - 1) * 100), 'sn': 1 if d.values[0] >= d.values[1] else -1}
    return res


def dailydata(d, mk_up=u'增加', mk_dw=u'减少'):
    d = d.resample('w-fri', closed='right', label='right').last().sort_index(ascending=False)
    res = {'tw_dt': d.index[0].date(), 'tw': d.values[0], 'lw': d.values[1],
           'mk': mk_up if d.values[0] >= d.values[1] else mk_dw, 'dif': abs(d.values[0] - d.values[1]),
           'dif_r': abs((d.values[0] / d.values[1] - 1) * 100), 'sn': 1 if d.values[0] >= d.values[1] else -1}
    return res


# -----获取日期
def getToday_Lastweek(date_time, ix):
    def _p(d):
        while True:
            if d in ix:
                break
            else:
                d = d - dt.timedelta(1)
        return d

    today = _p(date_time)
    lastWeek = _p(today - dt.timedelta(7))

    return today, lastWeek