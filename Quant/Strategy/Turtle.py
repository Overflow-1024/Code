# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np
import math


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.s = "000625.XSHE"
    # 唐奇安通道开仓和平仓的观测窗口
    context.channel_window_long = 20
    context.channel_window_short = 10
    # atr观测窗口
    context.atr_window = 20

    context.units_hold = 0
    context.units_max = 4
    context.last_price = 0.0


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 计算average True Range
def calc_atr(high_price, low_price, close_price):
    # high - low
    tr1 = high_price - low_price
    # high - close
    tr2 = high_price - close_price
    # close - low
    tr3 = close_price - low_price
    tr = np.max(np.array([tr1, tr2, tr3]), axis=0)
    atr = np.mean(tr)
    return atr


# 计算唐奇安通道上下轨
def calc_channel(high_price, low_price):
    upper = np.max(high_price)
    lower = np.min(low_price)
    return upper, lower


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # 获取历史最高价，最低价，收盘价
    high_price_atr = history_bars(context.s, context.atr_window + 1, '1d', 'high')
    low_price_atr = history_bars(context.s, context.atr_window + 1, '1d', 'low')
    close_price_atr = history_bars(context.s, context.atr_window + 2, '1d', 'close')

    # 计算atr和unit
    atr = calc_atr(high_price_atr[:-1], low_price_atr[:-1], close_price_atr[:-2])
    total_value = context.portfolio.total_value
    unit = math.floor(total_value * 0.01 / atr)

    # 计算唐奇安通道上下轨
    high_price_channel = history_bars(context.s, context.channel_window_long + 1, '1d', 'high')
    low_price_channel = history_bars(context.s, context.channel_window_short + 1, '1d', 'low')
    upper_price, lower_price = calc_channel(high_price_channel[:-1], low_price_channel[:-1])

    cur_position = context.portfolio.positions[context.s].quantity
    cash = context.portfolio.cash
    new_price = bar_dict[context.s].last

    # 判断条件 产生交易信号
    if context.units_hold == 0:
        # 开仓  首次买入
        if new_price > upper_price and cash > new_price * unit and context.units_hold < context.units_max:
            order = order_shares(context.s, unit)
            if order.status == ORDER_STATUS.FILLED:
                context.last_price = new_price
                context.units_hold += 1
                logger.info("open: {}".format(context.last_price))
    else:
        # 加仓
        if new_price > context.last_price + 0.5 * atr and cash > new_price * unit and context.units_hold < context.units_max:
            order = order_shares(context.s, unit)
            if order.status == ORDER_STATUS.FILLED:
                context.last_price = new_price
                context.units_hold += 1
                logger.info("add: {}".format(context.last_price))
        # 止盈
        elif new_price < lower_price:
            order = order_target_percent(context.s, 0)
            if order.status == ORDER_STATUS.FILLED:
                context.last_price = new_price
                context.units_hold = 0
                logger.info("exit: {}".format(context.last_price))
        # 止损
        elif new_price < context.last_price - 2 * atr:
            order = order_target_percent(context.s, 0)
            if order.status == ORDER_STATUS.FILLED:
                context.last_price = new_price
                context.units_hold = 0
                logger.info("stop: {}".format(context.last_price))


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
