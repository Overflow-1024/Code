from Projects.AmReport.Model.BaseModel import BaseModel
import Projects.AmReport.util as util

import pandas as pd


class Fund(BaseModel):

    def __init__(self, date, db_manager):
        super().__init__(date, db_manager)
        self.df_fig = self.data.copy().sort_index(ascending=False)
        self.df_txt = self.data.copy().head(100).copy()
        self.df_ncd = self.data_ncd.copy().sort_index(ascending=False)

        self.processed_data_dict = {}

    def _load_data(self):
        self.data = util.query_table(db_connect=self.db_manager, cltName='moneymkt', sort_k='date', datetime=self.datetime)
        self.data_ncd = util.query_table(db_connect=self.db_manager, cltName='NCD', sort_k=u'发行起始日', datetime=self.datetime)

    def fig_15(self):
        # ----图15
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 1, 'y')]
        return [data[u'R001'], data[u'R007'], data[u'GC007:加权平均']]

    def fig_16(self):
        # ----图16
        data = self.df_fig[[u'R001成交额', u'R007成交额', u'R014成交额']].resample('w-mon', closed='left', label='left').sum()
        data = data[data.index >= util.getNmonths(self.date, 40, 'm')]
        data['(R007+R014)/R001'] = (data[u'R007成交额'] + data[u'R014成交额']) / data[u'R001成交额'] * 100

        ##################################################################
        '''特殊处理2019-09-30的(R007+R014)/R001比值太小，设该值为空'''
        if '2019-09-30' in data.index:
            data.loc['2019-09-30', '(R007+R014)/R001'] = None
        ##################################################################

        return [data[u'R007成交额'], data[u'R001成交额'], data['(R007+R014)/R001']]

    def fig_17(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 48, 'm')]

        return [data[u'利率互换:FR007:1年'], data[u'利率互换:FR007:5年']]

    def fig_18(self):
        # ----图18
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')].copy()
        data[u'CNH_CNY价差BP'] = (data[u'USDCNH:即期汇率'] - data[u'即期汇率:美元兑人民币']) * 10000

        return [data[u'CNH HIBOR:隔夜'], data[u'CNH_CNY价差BP']]

    def fig_19(self):
        # ----图19
        data = pd.DataFrame()
        data[u'总发行量(亿元)'] = self.df_ncd[u'实际发行总额(亿元)'].resample('w-sun', closed='right').sum()
        data[u'总偿还量(亿元)'] = - self.df_ncd[[u'实际发行总额(亿元)', u'到期日']].set_index(u'到期日').resample('w-sun', closed='right').sum()
        data[u'净融资额'] = data[u'总发行量(亿元)'] + data[u'总偿还量(亿元)']

        data = data[data.index >= util.getNmonths(self.date, 30, 'm')]

        self.processed_data_dict['ncd_summary'] = data.sort_index(ascending=False)

        return [data[u'总发行量(亿元)'], data[u'总偿还量(亿元)'], data[u'净融资额']]

    def fig_20(self):
        # ----图20
        self.df_ncd[u'期限(月)'] = self.df_ncd[u'期限(月)'].round(0)
        data = pd.pivot_table(self.df_ncd, index=[u'发行起始日'], columns=[u'期限(月)'], values=[u'实际发行总额(亿元)'],
                              aggfunc={u'实际发行总额(亿元)': sum})
        data = data.resample('w-sun', closed='right', label='right').sum()

        data.columns = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 24, 36]
        all_sum = self.df_ncd[u'实际发行总额(亿元)'].resample('w-sun', closed='right').sum()
        for i in [1, 3, 6, 9]:
            data[u'%d个月' % i] = data[i] / all_sum * 100
        data[u'6-9个月'] = data[u'6个月'] + data[u'9个月']
        data[u'12个月及以上'] = 100 - data[u'6-9个月'] - data[u'3个月'] - data[u'1个月']

        data = data[data.index >= util.getNmonths(self.date, 2, 'y')]

        self.processed_data_dict['ncd_term_dis'] = data[[u'1个月', u'3个月', u'6个月', u'6-9个月', u'12个月及以上']].sort_index(
            ascending=False)

        return [data[u'1个月'], data[u'3个月'], data[u'6-9个月'], data[u'12个月及以上']]

    def fig_21(self):
        # ----图21
        data = pd.pivot_table(self.df_ncd, index=[u'发行起始日'], columns=[u'发行人类型'], values=[u'实际发行总额(亿元)'],
                              aggfunc={u'实际发行总额(亿元)': sum})
        data = data.resample('w-sun', closed='right').sum()
        data.columns = [u'住房储蓄银行', u'农信社', u'农合行', u'农商行', u'国有商业银行', u'城市商业银行', u'外资银行', u'政策性银行', u'村镇银行', u'民营银行',
                        u'股份制商业银行', u'邮政储蓄银行']
        all_sum = self.df_ncd[u'实际发行总额(亿元)'].resample('w-sun', closed='right').sum()
        for i in [u'国有商业银行', u'股份制商业银行', u'城市商业银行', u'农商行']:
            data[u'%s占比' % i] = data[i] / all_sum * 100
        data[u'城商、农商行发行规模占比'] = data[u'城市商业银行占比'] + data[u'农商行占比']
        data[u'国有、股份行发行规模占比'] = data[u'国有商业银行占比'] + data[u'股份制商业银行占比']

        data = data[data.index >= util.getNmonths(self.date, 40, 'm')]

        self.processed_data_dict['ncd_ocp'] = data[u'城商、农商行发行规模占比'].sort_index(ascending=False)

        return [data[u'城商、农商行发行规模占比'], data[u'国有、股份行发行规模占比']]

    def fig_22(self):
        # ----图22
        data = self.df_ncd[(self.df_ncd[u'期限(月)'] == 3) & (self.df_ncd[u'发行人类型'].isin([u'国有商业银行', u'股份制商业银行', u'城市商业银行', u'农商行']))].copy()
        data[u'收益率过渡'] = data[u'实际发行总额(亿元)'] * data[u'参考收益率(%)']
        data_1 = pd.pivot_table(data, index=[u'发行起始日'], columns=[u'发行人类型'], values=[u'收益率过渡'],
                                aggfunc={u'收益率过渡': sum}).resample('w-sun', closed='right').sum().fillna(0)
        data_2 = pd.pivot_table(data, index=[u'发行起始日'], columns=[u'发行人类型'], values=[u'实际发行总额(亿元)'],
                                aggfunc={u'实际发行总额(亿元)': sum}).resample('w-sun', closed='right').sum().fillna(0)
        data_1.columns = data_2.columns = [u'农商行', u'国有商业银行', u'城市商业银行', u'股份制商业银行']

        data_1.loc[:, u'城商、农商行3个月发行利率'] = (data_1[u'城市商业银行'] + data_1[u'农商行']).replace({0: None}) / (
                data_2[u'城市商业银行'] + data_2[u'农商行']).replace({0: None})
        data_1.loc[:, u'国有、股份行3个月发行利率'] = (data_1[u'国有商业银行'] + data_1[u'股份制商业银行']).replace({0: None}) / (
                data_2[u'国有商业银行'] + data_2[u'股份制商业银行']).replace({0: None})

        data_1 = data_1[data_1.index >= util.getNmonths(self.date, 34, 'm')]

        self.processed_data_dict['ncd_issuerate'] = data_1.sort_index(ascending=False)

        return [data_1[u'城商、农商行3个月发行利率'], data_1[u'国有、股份行3个月发行利率']]

    def txt_mm_1(self):
        # -----<body>mm_1
        df_2 = self.df_txt[[u'R001成交额', u'R007成交额', u'R014成交额']].resample('w-mon', closed='left', label='left').sum().sort_index(
            ascending=False)

        d_1 = util.dailydata(self.df_txt[u'R001'])
        d_2 = util.dailydata(self.df_txt[u'R007'])
        d_3 = util.dailydata(self.df_txt[u'GC007:加权平均'])
        d_4 = util.weekdata(df_2[u'R001成交额'] / 10000)
        d_5 = util.weekdata(df_2[u'R007成交额'])
        d_6 = util.weekdata(df_2[u'R014成交额'])
        d_7 = util.weekdata((df_2[u'R007成交额'] + df_2[u'R014成交额']) / df_2[u'R001成交额'] * 100)

        return [[],
                [util.parse_word(d_1, words=['上行', '下行']), d_1['dif'] * 100, d_1['tw'],
                util.parse_word(d_2, words=['上行', '下行']), d_2['dif'] * 100, d_2['tw'],
                util.parse_word(d_3, words=['上行', '下行']), d_3['dif'] * 100, d_3['tw'],
                util.parse_word(d_4, words=['增加', '减少']), d_4['dif'], d_4['tw'],
                util.parse_word(d_5, words=['增加', '减少']), d_5['dif'], d_5['tw'] / 10000,
                util.parse_word(d_6, words=['增加', '减少']), d_6['dif'], d_6['tw'],
                util.parse_word(d_7, words=['升', '降']), d_7['tw']]
                ]

    def txt_mm_2(self):
        d_1 = util.dailydata(self.df_txt[u'CNH HIBOR:隔夜'])
        d_2 = util.dailydata(self.df_txt[u'利率互换:FR007:1年'])
        d_3 = util.dailydata(self.df_txt[u'利率互换:FR007:5年'])

        return [[],
                [d_1['tw'], util.parse_word(d_1, words=['上行', '下行']), d_1['dif'] * 100,
                util.parse_word(d_2, words=['上行', '下行']), d_2['dif'] * 100, d_2['tw'],
                util.parse_word(d_3, words=['上行', '下行']), d_3['dif'] * 100, d_3['tw']]
                ]

    def txt_ncd_1(self):
        df = self.processed_data_dict

        # -----<body>ncd_1
        d_1 = util.weekdata(df['ncd_summary'][u'净融资额'])
        d_2 = util.weekdata(df['ncd_summary'][u'总发行量(亿元)'])
        d_3 = util.weekdata(df['ncd_summary'][u'总偿还量(亿元)'] * -1)
        d_4 = util.weekdata(df['ncd_term_dis'][u'1个月'])
        d_5 = util.weekdata(df['ncd_term_dis'][u'3个月'])
        d_6 = util.weekdata(df['ncd_term_dis'][u'6个月'])
        d_7 = util.weekdata(df['ncd_term_dis'][u'6-9个月'])
        d_8 = util.weekdata(df['ncd_term_dis'][u'12个月及以上'])
        d_9 = util.weekdata(df['ncd_issuerate'][u'城商、农商行3个月发行利率'])
        d_10 = util.weekdata(df['ncd_issuerate'][u'国有、股份行3个月发行利率'])

        return [[d_1['tw'], d_1['mk'], d_1['dif']],
                [d_2['mk'], d_2['dif'], d_2['tw'],
                d_3['mk'], d_3['dif'], d_3['tw'],
                df['ncd_term_dis'].iloc[0].sort_values(ascending=False).index[0],
                util.parse_word(d_4, words=['升', '降']), d_4['tw'],
                util.parse_word(d_5, words=['升', '降']), d_5['tw'],
                util.parse_word(d_6, words=['升', '降']), d_6['tw'],
                util.parse_word(d_7, words=['升', '降']), d_7['tw'],
                util.parse_word(d_8, words=['升', '降']), d_8['tw'],
                util.parse_word(d_9, words=['上升', '下降']), d_9['dif'] * 100, d_9['tw'],
                util.parse_word(d_10, words=['上升', '下降']), d_10['dif'] * 100, d_10['tw']]
                ]

    def txt_title_fig19(self):
        df = self.processed_data_dict
        d_1 = util.weekdata(df['ncd_summary'][u'净融资额'])

        return [d_1['tw'], d_1['mk'], d_1['dif']]

    def txt_title_fig20(self):
        df = self.processed_data_dict

        return df['ncd_term_dis'].iloc[0].sort_values(ascending=False).index[0]

    def txt_title_fig21(self):
        df = self.processed_data_dict
        d_1 = util.weekdata(df['ncd_ocp'])

        return [d_1['tw']]


