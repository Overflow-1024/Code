# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
def query_fund(context, bar_dict):
    stocks = all_instruments('CS').order_book_id
    fund_df = get_factor(stocks, ['pe_ratio', 'revenue'])
    fund_df = fund_df[(fund_df['pe_ratio'] > 40) & (fund_df['pe_ratio'] < 60)]
    fund_df = fund_df.sort_values(by='revenue', ascending=False).head(10).reset_index(1, drop=True)

    context.fund_df = fund_df
    # 实时打印日志
    logger.info(context.fund_df)

    context.stocks = context.fund_df.index.values
    update_universe(context.stocks)

    stocks_number = len(context.stocks)
    context.average_percent = 0.99 / stocks_number

    for holding_stock in context.portfolio.positions.keys():
        if context.portfolio.positions[holding_stock].quantity != 0:
            order_target_percent(holding_stock, 0)

    for stock in context.stocks:
        order_target_percent(stock, context.average_percent)


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    scheduler.run_monthly(query_fund, monthday=1)


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
