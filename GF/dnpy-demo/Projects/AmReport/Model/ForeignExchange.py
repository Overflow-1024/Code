from Projects.AmReport.Model.BaseModel import BaseModel
import Projects.AmReport.util as util


class ForeignExchange(BaseModel):

    def __init__(self, date, db_manager):
        super().__init__(date, db_manager)
        self.df_fig = self.data.copy().sort_index(ascending=False).replace({0: None})
        self.df_txt = self.data.head(100).copy()

    def _load_data(self):
        self.data = util.query_table(db_connect=self.db_manager, cltName='ForeignExchange', sort_k='date', datetime=self.datetime)

    # ---准备图片数据
    def fig_1(self):
        # ----图1
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 36, 'm')]

        return [data[u'即期汇率:美元兑人民币'], data[u'中间价:美元兑人民币'], data[u'美元指数']]

    def fig_2(self):
        # ----图2
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 16, 'm')]

        return [data[u'USDCNY:NDF:1年']]

    def fig_3(self):
        # ----图3
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 48, 'm')].copy()

        data[u'1年期美元兑人民币远期升水(bp)'] = data[u'买报价:美元兑人民币:1年'] / data[u'即期汇率:美元兑人民币']
        data[u'1年期中美国债利差'] = (data[u'中债国债到期收益率:1年'] - data[u'美国:国债收益率:1年']) * 100

        return [data[u'1年期美元兑人民币远期升水(bp)'], data[u'1年期中美国债利差']]

    def fig_4(self):
        # ----图4
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 34, 'm')].copy()

        data[u'CNH_CNY价差BP'] = (data[u'USDCNH:即期汇率'] - data[u'即期汇率:美元兑人民币']) * 10000

        return [data[u'即期汇率:美元兑人民币'], data[u'USDCNH:即期汇率'], data[u'CNH_CNY价差BP']]

    def fig_5(self):
        # ----图5
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 29, 'm')]

        return [data[u'Wind人民币汇率预估指数'], data[u'CFETS人民币汇率指数']]

    def fig_6(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 60, 'm')].copy()
        data[u'ICE美元指数非商业持仓净多头(多头-空头)'] = (data[u'ICE:美元指数:非商业多头持仓:持仓数量'] - data[u'ICE:美元指数:非商业空头持仓:持仓数量'])

        return [data[u'ICE美元指数非商业持仓净多头(多头-空头)'], data[u'美元指数'].resample('w-tue', closed='right', label='right').last()]

    def fig_7(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')]

        return [data[u'欧元兑美元'], data[u'英镑兑美元']]

    def fig_8(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')]

        return [data[u'伦敦现货黄金:以美元计价'], data[u'美元兑日元']]

    def fig_9(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')].copy()
        data[u'10年期美德国债利差(bp)'] = (data[u'美国:国债收益率:10年'] - data[u'德国:国债收益率:10年']) * 100

        return [data[u'10年期美德国债利差(bp)'], data[u'欧元兑美元']]

    def fig_10(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')]

        return [data[u'澳元兑美元'], data[u'CRB现货指数:综合']]

    def fig_11(self):
        data = self.df_fig[self.df_fig.index >= util.getNmonths(self.date, 24, 'm')]

        return [data[u'澳元兑美元'], data[u'南华工业品指数']]

    # ---准备文字数据
    def txt_title_fig5(self):
        d_1 = util.dailydata(self.df_txt[u'Wind人民币汇率预估指数'])

        return [util.parse_word(d_1, words=['上涨', '下跌']), d_1['dif_r']]

    def txt_title_fig6(self):
        d_1 = util.dailydata(self.df_txt[u'ICE:美元指数:非商业多头持仓:持仓数量'] - self.df_txt[u'ICE:美元指数:非商业空头持仓:持仓数量'])

        return [util.parse_word(d_1, words=['增加', '减少']), d_1['dif']]

    def txt_fx_1(self):
        d_1 = util.dailydata(self.df_txt[u'美元指数'])

        return [[],
                [d_1['tw'], util.parse_word(d_1, words=['上涨', '下跌']), d_1['dif_r']]
                ]

    def txt_fx_2(self):
        d_1 = util.dailydata(self.df_txt[u'即期汇率:美元兑人民币'])
        d_2 = util.dailydata(self.df_txt[u'中间价:美元兑人民币'])
        d_3 = util.dailydata(self.df_txt[u'USDCNY:NDF:1年'])
        d_4 = util.dailydata((self.df_txt[u'USDCNH:即期汇率'] - self.df_txt[u'即期汇率:美元兑人民币']) * 10000)
        d_5 = util.dailydata((self.df_txt[u'中债国债到期收益率:1年'] - self.df_txt[u'美国:国债收益率:1年']) * 100)

        return [[],
                [d_1['tw'], util.parse_word(d_1, words=['升值', '贬值'], rev=True), d_1['dif'] * 10000,
                util.parse_word(d_2, words=['升', '降'], rev=True), d_2['dif'] * 10000, d_2['tw'],
                util.parse_word(d_3, words=['上升', '下降']), d_3['dif_r'], d_3['tw'],
                util.parse_word(d_4, words=['上升', '下降']), d_4['tw'],
                util.parse_word(d_5, words=['升', '降']), d_5['tw']]
                ]

    def txt_fx_3(self):
        d_1 = util.dailydata(self.df_txt[u'欧元兑美元'])
        d_2 = util.dailydata(self.df_txt[u'英镑兑美元'])

        return[[],
               [d_1['tw'], util.parse_word(d_1, words=['升值', '贬值']), d_1['dif'] * 10000,
               d_2['tw'], util.parse_word(d_2, words=['升值', '贬值']), d_2['dif'] * 10000]
               ]

    def txt_fx_4(self):
        d_1 = util.dailydata(self.df_txt[u'伦敦现货黄金:以美元计价'])
        d_2 = util.dailydata(self.df_txt[u'美元兑日元'])

        return [[],
                [d_1['tw'], util.parse_word(d_1, words=['上涨', '下跌']), d_1['dif_r'],
                d_2['tw'], util.parse_word(d_2, words=['升值', '贬值'], rev=True), d_2['dif_r']]
                ]

    def txt_fx_5(self):
        d_1 = util.dailydata(self.df_txt[u'澳元兑美元'])
        d_2 = util.dailydata(self.df_txt[u'南华工业品指数'])

        return [[],
                [d_1['tw'], util.parse_word(d_1, words=['升值', '贬值']), d_1['dif'] * 10000,
                util.parse_word(d_1, words=['涨', '跌']), d_1['dif_r'],
                d_2['tw'], util.parse_word(d_2, words=['上涨', '下跌']), d_2['dif_r']]
                ]


