import dndata as d
import pandas as pd
import numpy as np
import datetime
import os

pd.set_option('display.max_columns', 1000)

period = [('9:30', '12:00'), ('13:30', '17:00')]

'''
FR007_1Y*2Y 买价 = FR007_1Y*5Y 买价 - FR007_2Y*5Y 卖价
FR007_1Y*2Y 卖价 = FR007_1Y*5Y 卖价 - FR007_2Y*5Y 买价
FR007_1Y*5Y 买价 = FR007_1Y*2Y 买价 + FR007_2Y*5Y 买价
FR007_1Y*5Y 卖价 = FR007_1Y*2Y 卖价 + FR007_2Y*5Y 卖价
FR007_2Y*5Y 买价 = FR007_1Y*5Y 买价 - FR007_1Y*2Y 卖价
FR007_2Y*5Y 卖价 = FR007_1Y*5Y 卖价 - FR007_1Y*2Y 买价

FR007_6M*9M 买价 = FR007_6M*1Y 买价 - FR007_9M*1Y 卖价
FR007_6M*9M 卖价 = FR007_6M*1Y 卖价 - FR007_9M*1Y 买价
FR007_6M*1Y 买价 = FR007_9M*1Y 买价 + FR007_6M*9M 买价
FR007_6M*1Y 卖价 = FR007_9M*1Y 卖价 + FR007_6M*9M 卖价
FR007_9M*1Y 买价 = FR007_6M*1Y 买价 - FR007_6M*9M 卖价
FR007_9M*1Y 卖价 = FR007_6M*1Y 卖价 - FR007_6M*9M 买价
'''

expression_list = [
    {'dst': 'FR007_1Y*2Y', 'dst_field': '买1价', 'op': 0,
     'src1': 'FR007_1Y*5Y', 'src1_field': '买1价', 'src2': 'FR007_2Y*5Y', 'src2_field': '卖1价'},

    {'dst': 'FR007_1Y*2Y', 'dst_field': '卖1价', 'op': 0,
     'src1': 'FR007_1Y*5Y', 'src1_field': '卖1价', 'src2': 'FR007_2Y*5Y', 'src2_field': '买1价'},

    {'dst': 'FR007_1Y*5Y', 'dst_field': '买1价', 'op': 1,
     'src1': 'FR007_1Y*2Y', 'src1_field': '买1价', 'src2': 'FR007_2Y*5Y', 'src2_field': '买1价'},

    {'dst': 'FR007_1Y*5Y', 'dst_field': '卖1价', 'op': 1,
     'src1': 'FR007_1Y*2Y', 'src1_field': '卖1价', 'src2': 'FR007_2Y*5Y', 'src2_field': '卖1价'},

    {'dst': 'FR007_2Y*5Y', 'dst_field': '买1价', 'op': 0,
     'src1': 'FR007_1Y*5Y', 'src1_field': '买1价', 'src2': 'FR007_1Y*2Y', 'src2_field': '卖1价'},

    {'dst': 'FR007_2Y*5Y', 'dst_field': '卖1价', 'op': 0,
     'src1': 'FR007_1Y*5Y', 'src1_field': '卖1价', 'src2': 'FR007_1Y*2Y', 'src2_field': '买1价'},



    {'dst': 'FR007_6M*9M', 'dst_field': '买1价', 'op': 0,
     'src1': 'FR007_6M*1Y', 'src1_field': '买1价', 'src2': 'FR007_9M*1Y', 'src2_field': '卖1价'},

    {'dst': 'FR007_6M*9M', 'dst_field': '卖1价', 'op': 0,
     'src1': 'FR007_6M*1Y', 'src1_field': '卖1价', 'src2': 'FR007_9M*1Y', 'src2_field': '买1价'},

    {'dst': 'FR007_6M*1Y', 'dst_field': '买1价', 'op': 1,
     'src1': 'FR007_9M*1Y', 'src1_field': '买1价', 'src2': 'FR007_6M*9M', 'src2_field': '买1价'},

    {'dst': 'FR007_6M*1Y', 'dst_field': '卖1价', 'op': 1,
     'src1': 'FR007_9M*1Y', 'src1_field': '卖1价', 'src2': 'FR007_6M*9M', 'src2_field': '卖1价'},

    {'dst': 'FR007_9M*1Y', 'dst_field': '买1价', 'op': 0,
     'src1': 'FR007_6M*1Y', 'src1_field': '买1价', 'src2': 'FR007_6M*9M', 'src2_field': '卖1价'},

    {'dst': 'FR007_9M*1Y', 'dst_field': '卖1价', 'op': 0,
     'src1': 'FR007_6M*1Y', 'src1_field': '卖1价', 'src2': 'FR007_6M*9M', 'src2_field': '买1价'},
]

contract_comb = {'FR007_1Y*2Y': 1, 'FR007_1Y*5Y': 1, 'FR007_2Y*5Y': 1}
trade_unit = {'FR007_1Y*2Y': 50, 'FR007_1Y*5Y': 20, 'FR007_2Y*5Y': 20}

# contract_comb = {'FR007_6M*9M': 2, 'FR007_6M*1Y': 3, 'FR007_9M*1Y': 1}
# trade_unit = {'FR007_6M*9M': 100, 'FR007_6M*1Y': 50, 'FR007_9M*1Y': 150}

expression = expression_list[2]


# 计算价格
def calc_price(exp, data_src1, data_src2):

    vfield_src1 = exp['src1_field'][0] + '可成交量'
    vfield_src2 = exp['src2_field'][0] + '可成交量'

    if (data_src1[vfield_src1] >= trade_unit[exp['src1']] * contract_comb[exp['src1']]
            and data_src2[vfield_src2] >= trade_unit[exp['src2']] * contract_comb[exp['src2']]):
        volume_flag = 1
    else:
        volume_flag = 0

    if exp['op'] == 1:
        price = data_src1[exp['src1_field']] + data_src2[exp['src2_field']]
    elif exp['op'] == 0:
        price = data_src1[exp['src1_field']] - data_src2[exp['src2_field']]
    else:
        price = 0
        print("error!")

    return price, volume_flag


def Stats_price_diff(exp, start, end):

    code_list = list(contract_comb.keys())
    df = d.get_irs_deep(code=code_list, start=start, end=end, source='not_real')

    # 预处理
    df = df[['交易品种', '更新时间', '买1价', '买可成交量', '卖1价', '卖可成交量']]

    df['更新时间'] = pd.to_datetime(df['更新时间'])
    df['日期'] = pd.to_datetime(df['更新时间'], format='%Y-%m-%d %H:%M:%S').dt.date
    df['时间'] = pd.to_datetime(df['更新时间'], format='%Y-%m-%d %H:%M:%S').dt.time

    df = df.fillna('')

    start_time_am = datetime.time(9, 30, 0)
    end_time_am = datetime.time(12, 0, 0)
    start_time_pm = datetime.time(13, 30, 0)
    end_time_pm = datetime.time(17, 0, 0)

    df.sort_values(by='更新时间', ascending=True, inplace=True)

    df.reset_index(inplace=True, drop=True)

    name_th_price = exp['dst_field'][0] + '理论价'
    name_price_diff = exp['dst_field'][0] + '价差'
    name_rd_dst = '盘口_' + exp['dst']
    name_rd_src1 = '盘口_' + exp['src1']
    name_rd_src2 = '盘口_' + exp['src2']

    # 定义要计算统计的新列
    df['目标合约'] = exp['dst']
    df[name_th_price] = ''
    df[name_price_diff] = ''
    df['可交易'] = ''
    df[name_rd_dst] = ''
    df[name_rd_src1] = ''
    df[name_rd_src2] = ''

    res = []
    # 用于遍历的指针
    cur = {exp['dst']: -1, exp['src1']: -1, exp['src2']: -1}

    # 按日期处理数据
    for date, grp in df.groupby('日期'):

        grp.reset_index(inplace=True, drop=True)

        # 倒序遍历
        for ind in range(len(grp)-1, -1, -1):

            time = grp.loc[ind, '更新时间']

            # 匹配3个产品的盘口记录
            cur_start = ind
            while cur_start + 1 < len(grp) and grp.loc[cur_start + 1, '更新时间'] == time:
                cur_start = cur_start + 1

            cur[exp['dst']] = cur[exp['src1']] = cur[exp['src2']] = cur_start

            while cur[exp['dst']] >= 0 and grp.loc[cur[exp['dst']], '交易品种'] != exp['dst']:
                cur[exp['dst']] = cur[exp['dst']] - 1

            while cur[exp['src1']] >= 0 and grp.loc[cur[exp['src1']], '交易品种'] != exp['src1']:
                cur[exp['src1']] = cur[exp['src1']] - 1

            while cur[exp['src2']] >= 0 and grp.loc[cur[exp['src2']], '交易品种'] != exp['src2']:
                cur[exp['src2']] = cur[exp['src2']] - 1

            if cur[exp['dst']] < 0 or cur[exp['src1']] < 0 or cur[exp['src2']] < 0:
                break

            if grp.loc[cur[exp['src1']], exp['src1_field']] == '' or grp.loc[cur[exp['src2']], exp['src2_field']] == '':
                continue

            data_dst = grp.loc[cur[exp['dst']]]
            data_src1 = grp.loc[cur[exp['src1']]]
            data_src2 = grp.loc[cur[exp['src2']]]

            # 保存盘口记录

            grp.loc[ind, name_rd_dst] = str({
                '买1': (data_dst['买1价'], data_dst['买可成交量']),
                '卖1': (data_dst['卖1价'], data_dst['卖可成交量'])
            })
            grp.loc[ind, name_rd_src1] = str({
                '买1': (data_src1['买1价'], data_src1['买可成交量']),
                '卖1': (data_src1['卖1价'], data_src1['卖可成交量'])
            })
            grp.loc[ind, name_rd_src2] = str({
                '买1': (data_src2['买1价'], data_src2['买可成交量']),
                '卖1': (data_src2['卖1价'], data_src2['卖可成交量'])
            })

            grp.loc[ind, name_th_price], grp.loc[ind, '可交易'] = calc_price(exp, data_src1, data_src2)

            if name_price_diff == '买价差' and data_dst['买1价'] != '' and grp.loc[ind, name_th_price] != '':
                grp.loc[ind, name_price_diff] = grp.loc[ind, name_th_price] - data_dst['买1价']
            elif name_price_diff == '卖价差' and data_dst['卖1价'] != '' and grp.loc[ind, name_th_price] != '':
                grp.loc[ind, name_price_diff] = data_dst['卖1价'] - grp.loc[ind, name_th_price]

        # 筛选交易时间段
        grp = grp.loc[(grp['时间'] >= start_time_am) & (grp['时间'] <= end_time_am) | (grp['时间'] >= start_time_pm) & (
                grp['时间'] <= end_time_pm)]

        flag = False  # 表示当前是否在一个交易区间内
        th_price = 0
        price_diff = 0

        # 选出价差为正 且 报单量满足交易的
        for index, row in grp.iterrows():

            if row[name_th_price] != '' and row['可交易'] != '':
                if flag:
                    if row[name_th_price] != th_price or row['可交易'] == 0:
                        flag = False
                        if (row[name_price_diff] == '' or row[name_price_diff] > 0) and row['可交易'] == 1:
                            flag = True
                            th_price = row[name_th_price]
                            res.append(row.copy())
                else:
                    if (row[name_price_diff] == '' or row[name_price_diff] > 0) and row['可交易'] == 1:
                        flag = True
                        th_price = row[name_th_price]
                        res.append(row.copy())

            # if row[name_th_price] != '' and row['可交易'] != '':
            #     if flag:
            #         if row[name_th_price] != th_price or row[name_price_diff] != price_diff or row['可交易'] == 0:
            #             flag = False
            #             if row['可交易'] == 1:
            #                 flag = True
            #                 th_price = row[name_th_price]
            #                 price_diff = row[name_price_diff]
            #                 res.append(row.copy())
            #     else:
            #         if row['可交易'] == 1:
            #             flag = True
            #             th_price = row[name_th_price]
            #             price_diff = row[name_price_diff]
            #             res.append(row.copy())


    res = pd.DataFrame(res)
    if res.shape[0] > 0:
        # 删掉不需要的列
        res.reset_index(inplace=True, drop=True)
        res.drop(columns=['买1价', '买可成交量', '卖1价', '卖可成交量'], axis=1, inplace=True)

    return res


start = '2022-08-01'
end = '2022-09-01'

filepath = './output/IRSresult_' + 'FR007_1Y2Y_0801-0901.xlsx'
df_res = Stats_price_diff(expression, start, end)
# df_res.to_excel(filepath, encoding='utf-8-sig')
print(len(df_res))

# filepath = './output/IRSsource_' + '_0801-0901.xlsx'
# code_list = list(contract_comb.keys())
# df = d.get_irs_deep(code=code_list, start=start, end=end, source='not_real')
# df.to_excel(filepath, encoding='utf-8-sig')