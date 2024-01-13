# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 选取板块
    context.stks = []
    context.stks.append(sector("consumer discretionary"))
    context.stks.append(sector("consumer staples"))
    context.stks.append(sector("health care"))
    context.stks.append(sector("telecommunication services"))
    # context.stks.append(sector("utilities"))
    # context.stks.append(sector("materials"))

    context.flag = True
    scheduler.run_weekly(rebalance, weekday=1)

    # 手动添加银行板块
    context.banks = ['000001.XSHE', '002142.XSHE', '600000.XSHG', '600015.XSHG', '600016.XSHG', '600036.XSHG',
                     '601009.XSHG', '601166.XSHG', '601169.XSHG', '601288.XSHG', '601328.XSHG', '601398.XSHG',
                     '601818.XSHG', '601939.XSHG', '601988.XSHG', '601998.XSHG']


def data_proc(df):
    # 用均值填充缺失值
    for col in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)

    return df


def get_stocks(context, bar_dict):
    stocks = set()

    for sector in context.stks:
        fund_df = get_factor(sector, ['pe_ratio', 'pb_ratio']).reset_index(1, drop=True)
        fund_df = data_proc(fund_df)
        fund_df = fund_df.loc[sector]
        fund_df = fund_df[(fund_df.pe_ratio > 0) & (fund_df.pe_ratio < 100)]

        fund_df = fund_df.sort_values('pb_ratio')
        fund_df['pb_score'] = np.linspace(1, len(fund_df), len(fund_df))
        fund_df = fund_df.sort_values('pe_ratio')
        fund_df['pe_score'] = np.linspace(1, len(fund_df), len(fund_df))

        scores = []
        fund_df['scores'] = fund_df['pb_score'] + fund_df['pe_score']
        fund_df = fund_df.sort_values('scores')
        fund_df = fund_df.head(2)

        stocks = stocks | set(fund_df.index.values)

    # 银行板块单独取市净率最低的两个
    fund_df = get_factor(context.banks, ['pb_ratio']).to_frame(name='pb_ratio')
    fund_df = data_proc(fund_df)
    fund_df = fund_df.loc[context.banks]

    fund_df = fund_df.sort_values('pb_ratio')
    fund_df = fund_df.head(2)

    stocks = stocks | set(fund_df.index.values)

    return stocks


def get_holdings(context):
    positions = context.stock_account.positions
    holdings = list(positions.keys())

    return holdings


def rebalance(context, bar_dict):
    stocks = get_stocks(context, bar_dict)
    holdings = set(get_holdings(context))

    to_buy = stocks - holdings
    to_sell = holdings - stocks

    for stock in to_sell:
        order_target_percent(stock, 0)

    total_value = context.portfolio.stock_account.total_value * 0.95
    avg_value = total_value / len(stocks)

    for stock in to_buy:
        if context.portfolio.stock_account.cash > avg_value:
            order_target_value(stock, avg_value)

    if context.flag == True:
        sell_open('IF88', 1)
        context.flag = False


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
