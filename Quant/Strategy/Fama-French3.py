# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.base = "000300.XSHG"
    context.num = 5
    context.window = 60
    context.rf = 0.04 / 252
    context.flag = True

    scheduler.run_monthly(rebalance, tradingday=1)


def get_data(context, stocks_pool):
    today = context.now.date()
    end_date = get_previous_trading_date(today, n=1)
    start_date = get_previous_trading_date(end_date, n=context.window - 1)
    logger.info("start_date: {} end_date: {}".format(start_date, end_date))

    df = get_factor(stocks_pool, ['market_cap', 'book_to_market_ratio_ttm'], count=context.window)
    df.reset_index(level='date', drop=False, inplace=True)
    df = df[df['date'] == start_date]

    total = df.shape[0]
    df.sort_values(by='market_cap', inplace=True, ascending=True)
    SMB_small = list(df.iloc[:total // 3].index.values)
    SMB_big = list(df.iloc[total - total // 3:].index.values)

    df.sort_values(by='book_to_market_ratio_ttm', inplace=True, ascending=False)
    HML_high = list(df.iloc[:total // 3].index.values)
    HML_low = list(df.iloc[total - total // 3:].index.values)

    pre_start_date = get_previous_trading_date(start_date, n=1)
    stock_price = get_price(stocks_pool, pre_start_date, end_date, '1d', 'close')
    stock_ret = stock_price.pct_change()
    stock_ret = stock_ret.iloc[1:]

    SMB = stock_ret[SMB_small].sum(axis=1) / len(SMB_small) - stock_ret[SMB_big].sum(axis=1) / len(SMB_big)

    HML = stock_ret[HML_high].sum(axis=1) / len(HML_high) - stock_ret[HML_low].sum(axis=1) / len(HML_low)

    market_price = history_bars(context.base, context.window + 1, "1d", "close")
    market_price_diff = np.diff(market_price)
    market_ret = market_price_diff / market_price[:-1] - context.rf

    stock_ret = stock_ret - context.rf

    return stock_ret, market_ret, SMB, HML


def get_stocks_FFmodel(context):
    stocks_pool_uni = index_components(context.base)
    stocks_pool = []
    # 排除掉不可交易和数据不足的股票
    for stock in stocks_pool_uni:
        stock_price = history_bars(stock, context.window + 1, "1d", "close")
        if is_suspended(stock, count=1) or len(stock_price) < context.window + 1:
            continue
        stocks_pool.append(stock)

    stock_ret, market_ret, SMB, HML = get_data(context, stocks_pool)

    SMB = np.array(SMB)
    HML = np.array(HML)
    market_ret = np.expand_dims(market_ret, axis=1)
    SMB = np.expand_dims(SMB, axis=1)
    HML = np.expand_dims(HML, axis=1)

    X = np.concatenate((market_ret, SMB, HML), axis=1)

    stock_alpha = []
    model = LinearRegression()
    for stock in stocks_pool:
        y = np.array(stock_ret[stock])
        model.fit(X, y)
        alpha = model.intercept_
        stock_alpha.append((stock, alpha))

    stock_alpha.sort(key=lambda s: s[1], reverse=False)
    stock_alpha = stock_alpha[:context.num]
    logger.info(stock_alpha)

    return stock_alpha


def rebalance(context, bar_dict):
    get_stocks_FFmodel(context)
    stock_alpha = get_stocks_FFmodel(context)
    stocks, _ = zip(*stock_alpha)
    stocks = set(stocks)

    positions = context.stock_account.positions
    holdings = set(list(positions.keys()))

    to_buy = stocks - holdings
    to_sell = holdings - stocks

    for stock in to_sell:
        order_target_percent(stock, 0)

    total_value = context.stock_account.total_value * 0.95
    avg_value = total_value / len(stocks)

    for stock in to_buy:
        if context.stock_account.cash > avg_value:
            order_target_value(stock, avg_value)

    # if context.flag == True:
    #     sell_open('IF88', 1)
    #     context.flag = False


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
