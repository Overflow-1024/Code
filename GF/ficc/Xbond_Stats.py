import dndata as d
import pandas as pd
import numpy as np

import datetime
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 1000)

QUANTITY_UNIT = 10000000
QUANTITY_THRESHOLD = 5 * QUANTITY_UNIT
RATIO = 0.8


# 统计倒挂交易时点（单只债券）
def get_transaction_point(data, quantity_threshold):

    def calc_price_diff(df):
        if df['买2净价'] != '' and df['卖2净价'] != '':
            return df['买2净价'] - df['卖2净价']
        else:
            return np.nan

    def is_transaction_point(df):
        if df['倒挂价差'] is not np.nan and (abs(df['倒挂价差']) < 0.0000001 or df['倒挂价差'] > 0.0000001) and min(df['买2量'], df['卖2量']) >= quantity_threshold:
            return 1
        else:
            return 0

    df = data.copy()

    df['倒挂价差'] = df.apply(calc_price_diff, axis=1)
    df['是否倒挂'] = df.apply(is_transaction_point, axis=1)
    res = df[df['是否倒挂'] == 1]

    return res


# 统计倒挂交易区间（单只债券）
def get_transaction_interval(data, quantity_threshold):

    def calc_price_diff(df):
        if df['买2净价'] != '' and df['卖2净价'] != '':
            return df['买2净价'] - df['卖2净价']
        else:
            return np.nan

    def is_transaction_point(df):
        if df['倒挂价差'] is not np.nan and (abs(df['倒挂价差']) < 0.0000001 or df['倒挂价差'] > 0.0000001):
            return 1
        else:
            return 0

    df = data.copy()

    res = []

    # 按日期处理数据
    for title, group in df.groupby(['日期', '上下午']):

        # 处理收盘时间边界
        group.sort_values('时间', ascending=True, inplace=True)

        # 新增一行
        group = group.append(group.iloc[-1])
        group = group.reset_index(drop=True)

        # 赋值
        if title[1] == 'AM':
            group.loc[group.index.max(), '时间'] = datetime.time(12, 0, 0)
        elif title[1] == 'PM':
            group.loc[group.index.max(), '时间'] = datetime.time(17, 0, 0)

        group.loc[group.index.max(), '系统时间'] = datetime.datetime.combine(group.loc[group.index.max(), '日期'], group.loc[group.index.max(), '时间'])

        group['倒挂价差'] = group.apply(calc_price_diff, axis=1)
        group['是否倒挂'] = group.apply(is_transaction_point, axis=1)

        flag = False    # 表示当前是否在一个倒挂区间内
        buy_price = -1
        sell_price = -1
        for index, row in group.iterrows():
            # 表示当前正在一个倒挂区间内
            if flag:
                # 结束当前倒挂区间
                if row['买2净价'] != buy_price or row['卖2净价'] != sell_price:
                    row['开始'] = 'end'
                    res.append(row.copy())
                    flag = False
                    # 开启一个新的倒挂区间
                    if row['是否倒挂'] == 1 and min(row['买2量'], row['卖2量']) >= quantity_threshold:
                        buy_price = row['买2净价']
                        sell_price = row['卖2净价']
                        row['开始'] = 'start'
                        res.append(row.copy())
                        flag = True
                # 收盘了，强制结束倒挂区间
                if index == group.index.max():
                    row['开始'] = 'end'
                    res.append(row.copy())
                    flag = False
            # 表示当前不在倒挂区间内
            else:
                # 开启一个新的倒挂区间
                if row['是否倒挂'] == 1 and min(row['买2量'], row['卖2量']) >= quantity_threshold:
                    buy_price = row['买2净价']
                    sell_price = row['卖2净价']
                    row['开始'] = 'start'
                    res.append(row.copy())
                    flag = True

    return res


def Stats_transaction_summary(start_date, end_date, code='*'):

    df = d.get_bond_deep_xbond(code, start=start_date, end=end_date)
    df = df[['代码', '买2净价', '买2量', '卖2净价', '卖2量', '系统时间']]

    df['系统时间'] = pd.to_datetime(df['系统时间'])
    df['日期'] = pd.to_datetime(df['系统时间'], format='%Y-%m-%d %H:%M:%S').dt.date
    df['时间'] = pd.to_datetime(df['系统时间'], format='%Y-%m-%d %H:%M:%S').dt.time

    start_time_am = datetime.time(9, 0, 0)
    end_time_am = datetime.time(12, 0, 0)
    start_time_pm = datetime.time(13, 30, 0)
    end_time_pm = datetime.time(17, 0, 0)

    df = df.loc[(df['时间'] >= start_time_am) & (df['时间'] <= end_time_am) | (df['时间'] >= start_time_pm) & (df['时间'] <= end_time_pm)]

    def get_apm(df):
        if start_time_am <= df['时间'] <= end_time_am:
            return 'AM'
        else:
            return 'PM'

    df['上下午'] = df.apply(get_apm, axis=1)

    df.sort_values('系统时间', ascending=True, inplace=True)

    summary = pd.DataFrame(columns=['债券代码', '总倒挂次数', '有效倒挂次数', '有效可成交量（亿）'])

    groups = df.groupby('代码')

    for code, group in groups:

        print(code)

        result = get_transaction_interval(group, quantity_threshold=50000000)

        if len(result) > 0:

            result = pd.DataFrame(result)

            result['时间间隔'] = result['系统时间'] - result['系统时间'].shift(1)
            result['时间间隔'] = result['时间间隔'].dt.total_seconds()

            result['成功与否'] = np.nan
            result.loc[(result['开始'] == 'end') & (result['时间间隔'] >= 2), '成功与否'] = 1
            result.loc[(result['开始'] == 'end') & (result['时间间隔'] < 2), '成功与否'] = 0

            result.loc[result['开始'] == 'start', '时间间隔'] = ''

            def get_quantity(df):
                if df['买2量'] != '' and df['卖2量'] != '':
                    return min(df['买2量'], df['卖2量']) * RATIO // QUANTITY_UNIT * QUANTITY_UNIT / 100000000
                else:
                    return ''

            result['有效可成交量（亿）'] = result.apply(get_quantity, axis=1)

            result.loc[result['开始'] == 'end', '有效可成交量（亿）'] = ''

            result['成功与否'].fillna(method='backfill', inplace=True)

            # result.to_excel('./output/{}_detail.xlsx'.format(code), encoding='utf-8-sig')

            count_total = result[result['开始'] == 'end'].shape[0]
            count_success = result[(result['开始'] == 'end') & (result['成功与否'] == 1)].shape[0]
            quantity = result.loc[(result['开始'] == 'start') & (result['成功与否'] == 1), '有效可成交量（亿）'].sum() * 2

            summary.loc[len(summary.index)] = {'债券代码': code, '总倒挂次数': count_total, '有效倒挂次数': count_success, '有效可成交量（亿）': quantity}

    summary.sort_values(by=['有效倒挂次数', '总倒挂次数', '有效可成交量（亿）'], ascending=False, inplace=True)
    # summary.to_excel('./output/summary.xlsx', encoding='utf-8-sig')


def Stats_threshold(start_date, end_date, code='*'):

    df = d.get_bond_deep_xbond(code, start=start_date, end=end_date)
    df = df[['代码', '买2净价', '买2量', '卖2净价', '卖2量', '系统时间']]

    df['系统时间'] = pd.to_datetime(df['系统时间'])
    df['日期'] = pd.to_datetime(df['系统时间'], format='%Y-%m-%d %H:%M:%S').dt.date
    df['时间'] = pd.to_datetime(df['系统时间'], format='%Y-%m-%d %H:%M:%S').dt.time

    start_time_am = datetime.time(9, 0, 0)
    end_time_am = datetime.time(12, 0, 0)
    start_time_pm = datetime.time(13, 30, 0)
    end_time_pm = datetime.time(17, 0, 0)

    df = df.loc[(df['时间'] >= start_time_am) & (df['时间'] <= end_time_am) | (df['时间'] >= start_time_pm) & (
                df['时间'] <= end_time_pm)]

    def get_apm(df):
        if start_time_am <= df['时间'] <= end_time_am:
            return 'AM'
        else:
            return 'PM'

    df['上下午'] = df.apply(get_apm, axis=1)

    df.sort_values('系统时间', ascending=True, inplace=True)

    x = []
    y = []

    quantity_threshold_list = [0, 1 * QUANTITY_UNIT, 2 * QUANTITY_UNIT, 3 * QUANTITY_UNIT, 4 * QUANTITY_UNIT,
                               5 * QUANTITY_UNIT, 6 * QUANTITY_UNIT, 7 * QUANTITY_UNIT,
                               8 * QUANTITY_UNIT, 9 * QUANTITY_UNIT, 10 * QUANTITY_UNIT]

    groups = df.groupby('代码')

    for th in quantity_threshold_list:

        print("threshold: {}".format(th))

        count = 0

        for code, group in groups:

            result = get_transaction_interval(group, th)
            count += len(result) / 2

        x.append(th / QUANTITY_UNIT)
        y.append(count)

    # 画图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.bar(x, y, width=0.5)
    plt.xlabel("盘口量阈值（千万）")
    plt.ylabel("总倒挂次数")
    plt.show()


start = '2022-08-01'
end = '2022-09-01'
code = 'IB220210'

# Stats_transaction_summary(start, end)
Stats_threshold(start, end, code)
