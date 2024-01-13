"""xbond复权引擎"""
import logging
import os
import pandas as pd
import datetime
from dndata.API.cons import BARMCODE_DB_NAME
from dndata.bond.engine.bondpool import get_bond_list
from dndata.tools.datetimetools import BusinessDay
from dndata.database import mongo

bd = BusinessDay('exchange')
logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)


class NotChangedError(Exception):
    def __init__(self, msg):
        self.msg = msg
    pass


def get_main_trading_bond_change_date(pre, suc, consecutive_days=3):

    """获取债券主力切换日期及价差，需提供相邻的两个券名称"""
    from dndata.bond.query import get_bond_tick_xbond

    raw = get_bond_tick_xbond(','.join((pre, suc)), start='1990-1-1', main=False, chinese=False)
    # 转一下日期的类型，便于后面进行比较和加减
    raw['DATE'] = pd.to_datetime(raw['DATE'])
    raw['TRANSACTTIME'] = pd.to_datetime(raw['TRANSACTTIME'])

    counted = raw.groupby(by=['DATE', 'SECURITYID']).YTM.agg(['count']).reset_index()

    pre_data = counted[counted['SECURITYID'] == pre].rename(columns={"SECURITYID": "predecessor", 'count': 'cnt_pre'})
    suc_data = counted[counted['SECURITYID'] == suc].rename(columns={"SECURITYID": "successor", 'count': 'cnt_suc'})

    # pre_data['DATE'] = pd.to_datetime(pre_data['DATE'])
    # suc_data['DATE'] = pd.to_datetime(suc_data['DATE'])

    # 获取旧券和新券的日期范围，即发行日期至今天
    date_start = counted.min()['DATE']
    date_start = date_start.date()

    date_end = counted.max()['DATE']
    date_end = date_end.date()

    # 从calendar_table中取出对应时间区间的工作日列表
    calendar_table = bd.calendar.calendar_table.copy()
    calendar_table = calendar_table.loc[(calendar_table.index >= date_start) & (calendar_table.index <= date_end)]
    calendar_table = calendar_table.reset_index()
    calendar_table.drop(index=calendar_table[calendar_table['DayoffFlag'] == 1].index, inplace=True)
    calendar_table.drop(columns='DayoffFlag', inplace=True)

    # 改一下列名和数据类型，让后面可以merge
    calendar_table.rename(columns={'date': 'DATE'}, inplace=True)
    calendar_table['DATE'] = pd.to_datetime(calendar_table['DATE'])

    pre_data = pd.merge(left=calendar_table, right=pre_data, how='left', on='DATE')
    suc_data = pd.merge(left=calendar_table, right=suc_data, how='left', on='DATE')

    pre_data['predecessor'].fillna(pre, inplace=True)
    pre_data['cnt_pre'].fillna(0, inplace=True)

    suc_data['successor'].fillna(pre, inplace=True)
    suc_data['cnt_suc'].fillna(0, inplace=True)

    merged = pd.merge(left=pre_data, right=suc_data, how='inner', on='DATE')

    merged['surpass'] = merged['cnt_suc'] > merged['cnt_pre']
    merged['surpass_'] = merged['surpass'].shift()

    merged['diff'] = merged.apply(lambda x: 1 if x.surpass != x.surpass_ else 0, axis=1)
    merged['label'] = merged['diff'].cumsum()

    # merged['label1'] = (merged['surpass'] != merged['surpass'].shift()).cumsum()

    for _, grp in merged[merged['surpass'] == True].groupby('label'):

        if len(grp) >= consecutive_days:
            change_date = list(grp.iloc[:consecutive_days]['DATE'])

            target_date = grp.iloc[consecutive_days-1]['DATE'] + bd
            target = raw[raw['DATE'] == target_date].sort_values(by=['TRANSACTTIME'])
            # 日期往前回溯，直到新券和旧券皆有交易
            while target[target['SECURITYID'] == pre].empty or target[target['SECURITYID'] == suc].empty:
                target_date = target_date - datetime.timedelta(days=1)
                target = raw[raw['DATE'] == target_date].sort_values(by=['TRANSACTTIME'])

            # change_date = change_date.strftime('%Y-%m-%d')
            # target = raw[raw['DATE'] == change_date].sort_values(by=['TRANSACTTIME'])

            pre_data = target[target['SECURITYID'] == pre]
            suc_data = target[target['SECURITYID'] == suc]

            # 找新券和旧券交易时间较为接近的两笔交易
            if pre_data['TRANSACTTIME'].iloc[0] < suc_data['TRANSACTTIME'].iloc[0]:
                # 以suc_data.iloc[0]为准
                pre_iloc = 0
                suc_iloc = 0
                while pre_iloc < len(pre_data) and pre_data['TRANSACTTIME'].iloc[pre_iloc] < suc_data['TRANSACTTIME'].iloc[0]:
                    pre_iloc = pre_iloc + 1
                if pre_iloc < len(pre_data):
                    pre_diff = suc_data['TRANSACTTIME'].iloc[0] - pre_data['TRANSACTTIME'].iloc[pre_iloc - 1]
                    post_diff = pre_data['TRANSACTTIME'].iloc[pre_iloc] - suc_data['TRANSACTTIME'].iloc[0]
                    if post_diff < pre_diff:
                        pre_iloc = pre_iloc
                    else:
                        pre_iloc = pre_iloc - 1
                else:
                    pre_iloc = pre_iloc - 1
            else:
                # 以pre_data.iloc[0]为准
                pre_iloc = 0
                suc_iloc = 0
                while suc_iloc < len(suc_data) and suc_data['TRANSACTTIME'].iloc[suc_iloc] < pre_data['TRANSACTTIME'].iloc[0]:
                    suc_iloc = suc_iloc + 1
                if suc_iloc < len(suc_data):
                    pre_diff = pre_data['TRANSACTTIME'].iloc[0] - suc_data['TRANSACTTIME'].iloc[suc_iloc - 1]
                    post_diff = suc_data['TRANSACTTIME'].iloc[suc_iloc] - pre_data['TRANSACTTIME'].iloc[0]
                    if post_diff < pre_diff:
                        suc_iloc = suc_iloc
                    else:
                        suc_iloc = suc_iloc - 1
                else:
                    suc_iloc = suc_iloc - 1

            pre_price = pre_data['PRICE'].iloc[pre_iloc]
            suc_price = suc_data['PRICE'].iloc[suc_iloc]
            pre_ytm = pre_data['YTM'].iloc[pre_iloc]
            suc_ytm = suc_data['YTM'].iloc[suc_iloc]

            change_date = list(map(lambda x: x.strftime("%Y-%m-%d"), change_date))

            return change_date, target_date, pre_price, suc_price, pre_ytm, suc_ytm
    else:
        raise NotChangedError(f'债券{pre}, {suc}之间没有检测到主力切换！')


def bond_main_code_to_mongo(bond_list, code_name):
    """
    根据bond_list按顺序逐一作为主力，
    查询前后两个券的切换日期及开盘价差（包括price/ytm），插入mongo
    规则为：当新券成交笔数连续3天超越旧券，第4个交易日切换主力至新券，且不往回切

    bond_list: list, 按发行顺序提供的债券列表，意即主力切换顺序也应如此
    code_name: str, 入库时的品种名称，如CDB10Y即国开10年期
    """
    for pre, suc in zip(bond_list, bond_list[1:]):
        change_date, target_date, pre_price, suc_price, pre_ytm, suc_ytm = get_main_trading_bond_change_date(pre, suc)
        print(f'新券{suc}在{change_date}3天成交笔数超越旧券{pre}')
        print(f'{code_name}在{target_date}切换主力，前主力{pre}开盘价{pre_price}，主力{suc}开盘价{suc_price}，价差{suc_price - pre_price}')
        flt = {
            'code': code_name,
            'date': change_date,
        }
        doc = {
            'today_code': suc,
            'yesterday_code': pre,
            'main_code_price': suc_price,
            'yesterday_code_price': pre_price,
            'price_diff': suc_price - pre_price,
            'main_code_ytm': pre_ytm,
            'yesterday_code_ytm': suc_ytm,
            'ytm_diff': suc_ytm - pre_ytm,
        }
        doc.update(flt)

        # mongo.replace_all(BARMCODE_DB_NAME, 'BOND_MAIN_CODE', flt, [doc])
        print(f'已替换数据库数据，flt={flt}')


# 统计给定时间区间内某个证券tick的缺失情况
def check_missingdata(code, date_start, date_end, type):

    # 从calendar_table中取出对应时间区间的工作日列表
    calendar_table = bd.calendar.calendar_table.copy()
    calendar_table = calendar_table.loc[(calendar_table.index >= date_start) & (calendar_table.index <= date_end)]

    calendar_table.index.name = 'DATE'
    calendar_table.index = pd.to_datetime(calendar_table.index)

    from dndata.bond.query import get_bond_tick_xbond
    from dndata.bond.query import get_bond_trade_qeubee
    if type == 'xbond':
        raw = get_bond_tick_xbond(code=code, start=date_start, end=date_end, main=False, chinese=False)
    elif type == 'qeubee':
        raw = get_bond_trade_qeubee(code=code, start=date_start, end=date_end, main=False, chinese=False)
    else:
        raw = None

    if raw.empty:
        return None

    raw['DATE'] = pd.to_datetime(raw['DATE'])

    t = pd.merge(left=calendar_table, right=raw, on='DATE', how='left')

    t.drop(index=t[t['DayoffFlag']==1].index, inplace=True)

    # t1统计记录条数，因为没有交易的也会占1条记录（NAN），另用t2统计NAN的日子
    t1 = t['DATE'].value_counts()
    t['flag'] = 0
    t.loc[t['SECURITYID'].isnull(), 'flag'] = 1
    t2 = t.groupby('DATE').agg({'flag': 'sum'})

    t2.index = pd.to_datetime(t2.index)
    t = pd.merge(left=t1, right=t2, left_index=True, right_index=True, how='inner')
    t = t.sort_index()

    t.rename(columns={'DATE': 'count'}, inplace=True)
    t['count'] = t['count'] - t['flag']

    return t['count']


if __name__ == '__main__':

    # 生成主力复权表
    bond_type = {
        'CB5Y': {'bond_type': '00', 'period': '5年'},
        'CB10Y': {'bond_type': '00', 'period': '10年'},
        'CDB5Y': {'bond_type': '02', 'period': '5年'},
        'CDB10Y': {'bond_type': '02', 'period': '10年'},
        'ABC5Y': {'bond_type': '04', 'period': '5年'},
        'ABC10Y': {'bond_type': '04', 'period': '10年'},
    }

    years = ['20', '21', '22']

    for type_name in bond_type:
        bond_list = []

        for year in years:
            bond_list.extend(get_bond_list(year=year, **bond_type[type_name]))

        print("\n")
        print(f'正在处理{years[0]}-{years[-1]}年{type_name}')
        print(f'{type_name}的主力列表为{bond_list}')
        try:
            bond_main_code_to_mongo(
                bond_list=bond_list,
                code_name=type_name,
            )
        except NotChangedError as e:
            print(e.msg)
            # print(f'{type_name}没有发生主力切换')
            continue
        print(f'{type_name}处理完成')



# if __name__ == '__main__':
#
#     # 生成主力复权表
#     bond_type = {
#         'CB5Y': {'bond_type': '00', 'period': '5年'},
#         'CB10Y': {'bond_type': '00', 'period': '10年'},
#         'CDB5Y': {'bond_type': '02', 'period': '5年'},
#         'CDB10Y': {'bond_type': '02', 'period': '10年'},
#         'ABC5Y': {'bond_type': '04', 'period': '5年'},
#         'ABC10Y': {'bond_type': '04', 'period': '10年'},
#     }
#
#     years = ['20', '21', '22']
#
#     for year in years:
#         for type_name in bond_type:
#             print(f'正在处理{year}年{type_name}')
#             bond_list = get_bond_list(year=year, **bond_type[type_name])
#             print(f'{type_name}的主力列表为{bond_list}')
#
#             for bond_code in bond_list:
#                 print(bond_code)
#                 # 获取债券发行时间
#
#                 projection = {'_id': 0, 'SECURITYID': 1, 'ISSUE_DATE': 1}
#                 res = mongo.query(
#                     db='RDI', collection='RDI_CFETS_BOND',
#                     flt={'SECURITYID': bond_code[2:]},
#                     projection=projection, return_as_df=False
#                 )
#
#                 start = datetime.datetime.strptime(res[0]['ISSUE_DATE'], '%Y-%m-%d')
#                 start = start.date()
#                 end = datetime.date(2022, 8, 25)
#                 record_xbond = check_missingdata(bond_code, start, end, type='xbond')
#                 record_qeubee = check_missingdata(bond_code, start, end, type='qeubee')
#
#                 if record_xbond is not None and record_qeubee is not None:
#
#                     record_xbond.rename('count_xbond', inplace=True)
#                     record_qeubee.rename('count_qeubee', inplace=True)
#                     record = pd.merge(record_xbond, record_qeubee, left_index=True, right_index=True, how='inner')
#                 elif record_xbond is not None:
#                     record = record_xbond
#
#                 elif record_qeubee is not None:
#                     record = record_qeubee
#
#                 else:
#                     record = None
#                     print("xbond和qeubee在所选时间段内均没有交易记录")
#                     continue
#
#                 # filename = year + '_' + type_name + '_' + bond_code + '.csv'
#                 # record.to_csv(os.path.join("D:/record-exchange", filename))
