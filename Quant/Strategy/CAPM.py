# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np
from scipy import stats


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.base = "000300.XSHG"
    context.num = 5
    context.window = 60
    context.rf = 0.04 / 252

    scheduler.run_monthly(rebalance, tradingday=1)


def price_to_ret(context, price):
    price_diff = np.diff(price)
    ret = price_diff / price[:-1] - context.rf

    return ret


def get_stocks(context):
    stocks_pool = index_components(context.base)
    market_price = history_bars(context.base, context.window + 1, "1d", "close")
    market_ret = price_to_ret(context, market_price)

    stock_alpha = []
    for stock in stocks_pool:
        stock_price = history_bars(stock, context.window + 1, "1d", "close")
        if is_suspended(stock, count=1) or len(stock_price) < context.window + 1:
            continue
        stock_ret = price_to_ret(context, stock_price)
        beta, alpha, rvalue, pvalue, stderr = stats.linregress(market_ret, stock_ret)
        stock_alpha.append((stock, alpha))

    stock_alpha.sort(key=lambda s: s[1], reverse=False)
    stock_alpha = stock_alpha[:context.num]
    logger.info(stock_alpha)

    return stock_alpha


def rebalance(context, bar_dict):
    stock_alpha = get_stocks(context)
    stocks, _ = zip(*stock_alpha)
    stocks = set(stocks)

    positions = context.portfolio.positions
    holdings = set(list(positions.keys()))

    to_buy = stocks - holdings
    to_sell = holdings - stocks

    for stock in to_sell:
        order_target_percent(stock, 0)

    total_value = context.portfolio.total_value * 0.9
    avg_value = total_value / len(stocks)

    for stock in to_buy:
        if context.portfolio.cash > avg_value:
            order_target_value(stock, avg_value)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
