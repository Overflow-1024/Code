ori_text = {
    'txt_fx_1': [u"美元指数XXXX，XXXXXXX。",
                 u"周五，美元指数收于{:.2f}，较上周{}{:.2f}%。"],
    'txt_fx_2': [u"人民币XXXX，XXXXXX。",
                 u"周五，美元兑人民币即期汇率收于{:.2f}，"
                 "人民币较上周{}{:.2f}bp，"
                 "中间价调{}{:.0f}bp至{:.2f}"
                 "1年期美元/人民币NDF较上周五{}{:.2f}%至{:.2f}，人民币XXXXX。"
                 "CNH-CNY价差{}，周五收于{:.2f}bp，XXXXXX。"
                 "1年期中美国债利差{}至{:.2f}bp"],

    'txt_fx_3': [u"欧元、英镑XXXXXXXXXXXX",
                 u"周五，欧元兑美元收于{:.2f}，较上周五{}{:.0f}bp。"
                 "周五，英镑兑美元收于{:.2f}，较上周五{}{:.0f}bp"
                 ],

    'txt_fx_4': [u"黄金价格XX，日元XX。",
                 u"周五，伦敦现货黄金收于{:.1f}美元/盎司，较上周五{}{:.1f}%。"
                 "周五，美元兑日元收于{:.2f}，日元较上周五{}{:.2f}%。"],

    'txt_fx_5': [u"澳元兑美元XXXXXX，南华工业品指数XXXXXX。",
                 u"周五，澳元兑美元收于{:.2f}，较上周五{}{:.0f}bp，{}幅{:.1f}%。"
                 "南华工业品指数收于{:.0f}，较上周五{}{:.1f}%。"],
    'txt_mm_1': [u"本周资金利率XXXX，资金面XX。",
                 u"全周看，R001{}{:.0f}bp至{:.2f}%，R007{}{:.0f}bp至{:.2f}%，GC007{}{:.0f}bp至{:.2f}%."
                 "成交额方面，R001成交额较上周{}{:.1f}万亿至{:.1f}万亿，R007成交额较上周{}{:.0f}亿至{:.1f}万亿，R014{}{:.0f}亿至{:.0f}亿，"
                 "(R007+R014)/R001成交额占比{}至{:.0f}%。"],
    'txt_mm_2': [u"本周利率互换FR007 XXXX，隔夜CNH HIBOR XXXX，跨境流动性XXXX。",
                 u"隔夜CNH HIBOR周五收于{:.2f}%，较上周五{}{:.0f}bp."
                 "1年期利率互换FR007{}{:.0f}bp至{:.2f}%、5年期利率互换FR007{}{:.0f}bp至{:.2f}%。"
                 ],
    'txt_ncd_1': [u"同业存单净融资{:.0f}亿元，较上周{}{:.0f}亿元。",
                  u"与上周相比，同业存单发行量{}{:.0f}亿至{:.0f}亿元，到期量{}{:.0f}亿至{:.0f}亿元。"
                  "从期限分布看，本周新发行同业存单以{}期限为主，其中，1个月期限占比{}至{:.0f}%，"
                  "3个月期限占比{}至{:.0f}%，6个月期限占比{}至{:.0f}%，6-9个月期限占比{}至{:.0f}%，1年及以上期限占比{}至{:.0f}%。"
                  "同业存单发行成本XX，其中，3个月城商、农商行同业存单发行成本{}{:.0f}bp至{:.2f}%，"
                  "3个月国有、股份行发行利率{}{:.0f}bp至{:.2f}%。"]

}

ori_title_text = {
    'txt_title_fig5': u"Wind人民币汇率预估指数较上周五{}{:.2f}%",
    'txt_title_fig6': u"ICE美元指数非商业持仓净多头{}{:.0f}张",
    'txt_title_fig19': u"同业存单净融资{:.0f}亿元，较上周{}{:.0f}亿元",
    'txt_title_fig20': u"本周新发行同业存单以{}期限为主",
    'txt_title_fig21': u"城商、农商行同业存单发行规模占比为{:.0f}%",
}

table_list = ['tbl_1']

figure_list = ['fig_1', 'fig_2', 'fig_3', 'fig_4', 'fig_5', 'fig_6', 'fig_7', 'fig_8', 'fig_9', 'fig_10',
               'fig_11', 'fig_12', 'fig_13', 'fig_14', 'fig_15', 'fig_16', 'fig_17', 'fig_18', 'fig_19', 'fig_20',
               'fig_21', 'fig_22']

# 颜色
blue = '#5B9DD5'
orange = '#ED7D31'
gray = '#A5A5A5'

figure_config = {
    'fig_1': {'lines_left': [0, 1], 'lines_right': [2],
              'fname': 'figure_1', 'range_yax1': [6, 7.6], 'range_yax2': [85, 105],
              'lines_config': [
                  {'color': blue},
                  {'color': orange},
                  {'color': gray}
                ]
              },
    'fig_2': {'lines_left': [0],
              'fname': 'figure_2', 'month_gap': 1,
              'lines_config': [
                  {'color': blue}
                ]
              },
    'fig_3': {'lines_left': [0, 1],
              'fname': 'figure_3', 'n_col': 1, 'range_yax1': [-100, 300],
                'lines_config': [
                    {'color': blue},
                    {'color': orange}
                  ]
              },
    'fig_4': {'lines_left': [0, 1], 'lines_right': [2],
              'fname': 'figure_4',
                'lines_config': [
                    {'color': blue},
                    {'color': orange},
                    {'color': gray, 'kinds': 'bar'}
                  ]
                },
    'fig_5': {'lines_left': [0, 1],
              'fname': 'figure_5', 'month_gap': 1, 'figure_size': 'large',
              'lines_config': [
                  {'color': blue},
                  {'color': orange},
                ]
              },
    'fig_6': {'lines_left': [0], 'lines_right': [1],
              'fname': 'figure_6', 'range_yax2': [70, 120],
              'lines_config': [
                  {'color': blue},
                  {'color': orange},
                ]
              },
    'fig_7': {'lines_left': [0, 1],
              'fname': 'figure_7',
              'lines_config': [
                  {'color': blue},
                  {'color': orange},
                ]
              },
    'fig_8': {'lines_left': [0], 'lines_right': [1],
              'fname': 'figure_8', 'range_yax2': [100, 120],
              'lines_config': [
                  {'color': orange},
                  {'color': blue},
                ]
              },
    'fig_9': {'lines_left': [0], 'lines_right': [1],
              'fname': 'figure_9',
              'lines_config': [
                  {'color': orange},
                  {'color': blue},
                ]
              },
    'fig_10': {'lines_left': [0], 'lines_right': [1],
               'fname': 'figure_10',
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                 ]
               },
    'fig_11': {'lines_left': [0], 'lines_right': [1],
               'fname': 'figure_11', 'range_yax1': [0.55, 0.85],
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                 ]
               },
    'fig_12': {'lines_left': [0], 'lines_right': [1],
               'fname': 'figure_12', 'range_yax2': [2, 3], 'month_gap': 2,
               'lines_config': [
                   {'color': blue, 'keep_0': True},
                   {'color': orange},
                 ]
               },
    'fig_13': {'lines_left': [0], 'lines_right': [1],
               'fname': 'figure_13', 'month_gap': 2, 'range_yax1': [30000, 60000],
               'lines_config': [
                   {'color': blue, 'keep_0': True},
                   {'color': orange},
                 ]
               },
    'fig_14': {'lines_left': [0, 1, 2, 3, 4, 5, 6, 7],
               'fname': 'figure_14', 'n_col': 4, 'month_gap': 1, 'margins_ax1': 0.005,
               'lines_config': [
                   {'kinds': 'scatter', 'color': 'dodgerblue', 'marker': 'o'},
                   {'kinds': 'scatter', 'color': 'orangered', 'marker': 'o'},
                   {'kinds': 'scatter', 'color': 'darkorange', 'marker': '*'},
                   {'kinds': 'scatter', 'color': 'orange', 'marker': 'o'},
                   {'kinds': 'scatter', 'color': 'darkblue', 'marker': '^'},
                   {'kinds': 'scatter', 'color': 'limegreen', 'marker': 's'},
                   {'kinds': 'scatter', 'color': 'lightsalmon', 'marker': 'd'},
                   {'kinds': 'scatter', 'color': gray, 'marker': '^'},
                 ]
               },
    'fig_15': {'lines_left': [0, 1, 2],
               'fname': 'figure_15', 'n_col': 3, 'month_gap': 1, 'figure_size': 'large',
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                   {'color': gray}
                 ]
               },
    'fig_16': {'lines_left': [0, 1], 'lines_right': [2],
               'fname': 'figure_16', 'n_col': 3, 'Fmter_yax2': '%.f%%', 'range_yax2': [0, 90], 'figure_size': 'large',
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                   {'color': gray}
                 ]
               },
    'fig_17': {'lines_left': [0, 1],
               'fname': 'figure_17',
               'lines_config': [
                   {'color': blue},
                   {'color': orange}
                ]
               },

    'fig_18': {'lines_left': [0], 'lines_right': [1],
               'fname': 'figure_18',
               'lines_config': [
                   {'color': blue, 'lw': 3},
                   {'color': orange, 'lw': 3}
                ]
               },
    'fig_19': {'lines_left': [0, 1], 'lines_right': [2],
               'fname': 'figure_19', 'n_col': 3, 'month_gap': 2, 'figure_size': 'large',
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                   {'color': gray, 'kinds': 'bar', 'width': 5}
                 ]
               },
    'fig_20': {'lines_left': [0, 1, 2, 3],
               'fname': 'figure_20', 'n_col': 4, 'Formatter_axis1': '%.f%%', 'month_gap': 2, 'figure_size': 'large',
               'lines_config': [
                   {'color': blue},
                   {'color': orange},
                   {'color': gray},
                   {'color': 'gold'}
                 ]
               },
    'fig_21': {'lines_left': [0, 1],
               'fname': 'figure_21', 'n_col': 1, 'Formatter_axis1': '%.f%%',
               'lines_config': [
                   {'color': blue},
                   {'color': orange}
                 ]
               },
    'fig_22': {'lines_left': [0, 1],
               'fname': 'figure_22', 'n_col': 1,
               'lines_config': [
                   {'color': blue},
                   {'color': orange}
                 ]
               },
}

INDEX_NAME = {"801210.SI": u"休闲服务(申万)", "801760.SI": u"传媒(申万)", "801150.SI": u"医药生物(申万)",
              "801200.SI": u"商业贸易(申万)", "801730.SI": u"电气设备(申万)", "801750.SI": u"计算机(申万)",
              "801710.SI": u"建筑材料(申万)", "801040.SI": u"钢铁(申万)", "801130.SI": u"纺织服装(申万)",
              "801880.SI": u"汽车(申万)", "801740.SI": u"国防军工(申万)", "801890.SI": u"机械设备(申万)",
              "801120.SI": u"食品饮料(申万)", "801050.SI": u"有色金属(申万)", "801110.SI": u"家用电器(申万)",
              "801080.SI": u"电子(申万)", "801770.SI": u"通信(申万)", "801030.SI": u"化工(申万)",
              "801170.SI": u"交通运输(申万)", "801180.SI": u"房地产(申万)", "801230.SI": u"综合(申万)",
              "801140.SI": u"轻工制造(申万)", "801020.SI": u"采掘(申万)", "801160.SI": u"公用事业(申万)",
              "801790.SI": u"非银金融(申万)", "801780.SI": u"银行(申万)", "801720.SI": u"建筑装饰(申万)",
              "801010.SI": u"农林牧渔(申万)", "801003.SI": u"申万A股"}

