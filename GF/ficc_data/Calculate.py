# -*- coding: utf-8 -*-
from __future__ import division
import datetime
import warnings
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP

import Utils as util

# -----------获取N月后或N月前的日期
def get_nMonth_date(date, n=0):
    '''''
    获取N月后或N月前的日期
    '''
    thisYear = date.year  # 获取当前日期的年份
    thisMon = date.month  # 获取当前日期的月份
    thisDay = date.day  # 获取当前日期的天数
    totalMon = thisMon + n  # 加上n月后的总月份数

    lastYear = 0
    lastMon = 0
    if (n >= 0):  # 如果n大于等于0
        if (totalMon <= 12):  # 如果总月份数少于12
            lastYear = thisYear
            lastMon = totalMon
        else:
            i = totalMon // 12  # 年份递增数
            j = totalMon % 12  # 月份递增数
            if (j == 0):  # 月份递增数等于0
                i -= 1  # 年份减一
                j = 12  # 月份为12
            thisYear += i  # 年份递增

            lastYear = thisYear
            lastMon = j

    else:  # 如果n少于0
        if ((totalMon > 0) and (totalMon < 12)):  # 如果总月份数大于0少于12
            lastYear = thisYear
            lastMon = totalMon
        else:  # 如果总月份数少于0
            i = totalMon // 12  # 年份递减数
            j = totalMon % 12  # 月份递减数

            if (j == 0):  # 月份递减数等于0
                i -= 1  # 年份减一
                j = 12  # 月份为12
            thisYear += i

            lastYear = thisYear
            lastMon = j

    last_date = datetime.date(lastYear, lastMon, thisDay)
    return last_date


# -----------根据起息日和到期日获取利息“日期流”
def get_dateList_by_StartAndEnd(start_date, end_date, freq):
    '''
    start_date:datetiem类型，起息日
    end_date:datetime类型，到期日
    freq:整数，付息频率
    '''
    dateList = []  # 日期流列表
    current_date = start_date  # 当前日期初始化
    n = 12 // freq  # 递增月份
    while current_date < end_date:
        current_date = get_nMonth_date(current_date, n)  # 获取下一次付息日
        dateList.append(current_date)  # 添加到日期流表
    return dateList


# -----------根据当前交易日获取最近付息日和上一次付息日
def get_preAndNextDate_by_trade(trade_date, start_date, end_date, freq):
    dateList = get_dateList_by_StartAndEnd(start_date, end_date, freq)
    pre_date = start_date
    next_date = start_date

    for current_date in dateList:
        if current_date > trade_date:
            next_date = current_date
            break
        else:
            pre_date = current_date
    return (pre_date, next_date)


def newton(func, x0, fprime=None, fprime2=None, args=(), tol=1.48e-13, maxiter=100):
    """
    牛顿法求func(x)=0的解。
    :param func:函数，一元方程
    :param x0:浮点数，根的猜测值，越接近真值收敛速度越快
    :param fprime：可选函数，当一阶导函数可用时，设置该参数
    :param fprime2:可选函数，当二阶导函数可用时，设置该参数
    :param args:数组，方程函数的额外参数
    :param tol:浮点数，零值的允许误差范围
    :param mixiter:整数，最大的迭代次数
    """

    # 检验tol的合法性
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)

    # 检验迭代次数的合法性
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # 检验一阶导数是否可用
    if fprime is not None:

        p0 = 1.0 * x0  # 转换位浮点数，不用float，因为乘以1.0复数仍然可用
        fder2 = 0  # 二阶段初始化
        for iter in range(maxiter):  # 开始迭代
            myargs = (p0,) + args  # 整合所有参数
            fder = fprime(*myargs)  # 运行一阶导函数
            if fder == 0:  # 如果导函数为零
                msg = "一阶导为零"
                warnings.warn(msg, RuntimeWarning)  # 报运行错误
                return p0  # 返回初始猜想值

            fval = func(*myargs)  # 计算方程的值
            if fprime2 is not None:  # 如果二阶导函数可用
                fder2 = fprime2(*myargs)  # 运行二阶导
            if fder2 == 0:  # 如果二阶导为零

                p = p0 - fval / fder  # 使用一阶导函数执行牛顿迭代法步骤
            else:  # 如果二阶导函数不为零
                # 使用Parabolic Halley方法
                discr = fder ** 2 - 2 * fval * fder2
                if discr < 0:
                    p = p0 - fder / fder2
                else:
                    p = p0 - 2 * fval / (fder + np.sign(fder) * np.sqrt(discr))
            if abs(p - p0) < tol:  # 判断是否逼近实际根值
                return p  # 逼近则返回
            p0 = p  # 未逼近则更新根值
    else:  # 如果一阶导函数和二阶导函数都不可用，则使用割线法
        p0 = x0  # 初始根值
        if x0 >= 0:
            p1 = x0 * (1 + 1e-4) + 1e-4
        else:
            p1 = x0 * (1 + 1e-4) - 1e-4
        q0 = func(*((p0,) + args))
        q1 = func(*((p1,) + args))
        for iter in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "偏差达到" % (p1 - p0)
                    warnings.warn(msg, RuntimeWarning)  # 报运行错误
                return (p1 + p0) / 2.0
            else:
                p = p1 - q1 * (p1 - p0) / (q1 - q0)  # 求根
            if abs(p - p1) < tol:  # 是否逼近实际根值
                return p  # 若逼近则返回
            p0 = p1  # 更新次根值
            q0 = q1  # 更新次方程值
            p1 = p  # 更新根值
            q1 = func(*((p1,) + args))  # 更新方程值
    msg = "%d 次迭代后未能收敛,考虑适当加大迭代次数， 最后值是 %s" % (maxiter, p)
    raise RuntimeError(msg)


# %%-----牛顿迭代法自实现
def newton_dyt(func, x0, fprime=None, fprime2=None, args=(), tol=1.48e-13, maxiter=100,
               ):
    '''
    func:函数，一元方程
    x0:浮点数，根的猜测值，越接近真值收敛速度越快
    fprime：可选函数，当一阶导函数可用时，设置该参数
    fprime2:可选函数，当二阶导函数可用时，设置该参数
    args:数组，方程函数的额外参数
    tol:浮点数，零值的允许误差范围
    mixiter:整数，最大的迭代次数
    '''

    # 检验tol的合法性
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)

    # 检验迭代次数的合法性
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # 检验一阶导数是否可用
    if fprime is not None:

        p0 = 1.0 * x0  # 转换位浮点数，不用float，因为乘以1.0复数仍然可用
        fder2 = 0  # 二阶段初始化
        for iter in range(maxiter):  # 开始迭代
            myargs = (p0,) + args  # 整合所有参数
            fder = fprime(*myargs)  # 运行一阶导函数
            if fder == 0:  # 如果导函数为零
                msg = "一阶导为零"
                warnings.warn(msg, RuntimeWarning)  # 报运行错误
                return p0  # 返回初始猜想值

            fval = func(*myargs)  # 计算方程的值
            if fprime2 is not None:  # 如果二阶导函数可用
                fder2 = fprime2(*myargs)  # 运行二阶导
            if fder2 == 0:  # 如果二阶导为零

                p = p0 - fval / fder  # 使用一阶导函数执行牛顿迭代法步骤
            else:  # 如果二阶导函数不为零
                # 使用Parabolic Halley方法
                discr = fder ** 2 - 2 * fval * fder2
                if discr < 0:
                    p = p0 - fder / fder2
                else:
                    p = p0 - 2 * fval / (fder + np.sign(fder) * math.sqrt(discr))
            if abs(p - p0) < tol:  # 判断是否逼近实际根值
                return p  # 逼近则返回
            p0 = p  # 未逼近则更新根值
    else:  # 如果一阶导函数和二阶导函数都不可用，则使用割线法
        p0 = x0  # 初始根值
        if x0 >= 0:
            p1 = x0 * (1 + 1e-4) + 1e-4
        else:
            p1 = x0 * (1 + 1e-4) - 1e-4
        q0 = func(*((p0,) + args))
        q1 = func(*((p1,) + args))
        for iter in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "偏差达到 %f" % (p1 - p0)
                    warnings.warn(msg, RuntimeWarning)  # 报运行错误
                return (p1 + p0) / 2.0
            else:
                p = p1 - q1 * (p1 - p0) / (q1 - q0)  # 求根
            if abs(p - p1) < tol:  # 是否逼近实际根值
                return p  # 若逼近则返回
            p0 = p1  # 更新次根值
            q0 = q1  # 更新次方程值
            p1 = p  # 更新根值
            q1 = func(*((p1,) + args))  # 更新方程值
    msg = "%d 次迭代后未能收敛,考虑适当加大迭代次数， 最后值是 %s" % (maxiter, p)
    raise RuntimeError(msg)


# %%-----到期收益率
# -----------处于最后付息周期的附息债券(一年以内)
def bond_ytm1(PV, M, T, payType=1, C=None, N=None, f=None, guess=0.05):
    '''
    PV:市场价格(债券全价)
    M:面值
    T:剩余付息年数
    C:票面利息(一次还本付息和定期支付)
    N：一次还本付息的偿还期限（一次还本付息）
    f:利息支付频率（定期支付）
    payType:利息支付方式，包括1-贴现，2-一次还本付息(需要填写票面利率和偿还期限)，
    3-定期支付（附息债权,需要填写票面利率和支付频率）
    '''
    if payType == 1:
        # 贴现
        FV = 100
    elif payType == 2:
        # 一次还本付息
        FV = M + N * C / 100. * M
    elif payType == 3:
        # 定期支付
        FV = M + C / 100. * M / f
    else:
        pass

    ytm = (FV - PV) / PV / T  # 计算到期收益率

    return ytm


# -----------处于最后付息周期的固定利率附息债券（一年以上）
def bond_ytm2(PV, M, T, payType=1, C=None, N=None, guess=0.05):
    '''
    PV:市场价格(债券全价)
    M:面值
    T:剩余付息年数（待偿期）
    C:票面利息(一次还本付息)
    N：一次还本付息的偿还期限（一次还本付息）

    payType:利息支付方式，包括1-贴现，2-一次还本付息(需要填写票面利率和偿还期限)，

    '''
    if payType == 1:
        # 贴现
        FV = 100
    elif payType == 2:
        # 一次还本付息
        FV = M + N * C / 100. * M
    else:
        pass

    ytm = (FV / PV) ** (1 / T) - 1
    return ytm


def bond_ytm3(PV, M, n, w, C, f=2, guess=0.05):
    '''
    PV:市场价格(债券全价)
    M:面值
    n:剩余付息次数
    w:不处于付息周期的年化天数
    C:票面利息
    f:每年的利息支付频率
    '''
    f = float(f)  # 转换为浮点数对象
    cp = C / 100. * M / f  # 每期付息数
    dt = [i for i in range(int(n))]  # 遍历付息周期数
    ytm_func = lambda y: sum([cp / (1 + y / f) ** (w + t) for t in dt]) + M / (1 + y / f) ** (w + n - 1) - PV
    return newton_dyt(ytm_func, guess)


def zero_coupon_bond(M, y, t):
    '''
    M:面值
    y：贴现率
    t：期限
    '''
    return M / (1 + y) ** t


# -----------息票债券
def bond_price(M, n, w, y, C, f=2):
    '''
    M:面值
    n:剩余付息次数
    w:不处于付息周期的年化天数
    y:收益率
    C:票面利息
    f:每年的利息支付频率
    '''
    f = float(f)  # 转换为浮点数对象
    cp = C / 100. * M / f  # 每期付息数
    dt = [i for i in range(int(n))]  # 遍历付息周期数
    price = sum([cp / (1 + y / f) ** (w + t) for t in dt]) + M / (1 + y / f) ** (w + n - 1)
    return price


# ----------麦考利久期
# 市场价格已知
def bond_duration_PV(PV, M, n, w, C, f):
    cp = C / 100. * M / f  # 每期付息数
    y = bond_ytm3(PV, M, n, w, C, f)
    dt = [i for i in range(int(n))]  # 遍历付息周期数
    time_pv = sum([cp * (w + t) / (1 + y / f) ** (w + t) for t in dt]) + M * (w + n - 1) / (1 + y / f) ** (w + n - 1)
    duration = time_pv / PV
    return duration


# 市场利率已知
def bond_duration_y(M, n, w, y, C, f):
    cp = C / 100. * M / f  # 每期付息数
    PV = bond_price(M, n, w, y, C, f)
    dt = [i for i in range(int(n))]  # 遍历付息周期数
    time_pv = sum([cp * (w + t) / (1 + y / f) ** (w + t) for t in dt]) + M * (w + n - 1) / (1 + y / f) ** (w + n - 1)
    duration = time_pv / PV
    return duration


# ----------修正久期
def bond_mod_duration(PV, M, n, w, C, f, dy=0.01):
    ytm = bond_ytm3(PV, M, n, w, C, f)
    ytm_minus = ytm - dy
    price_minus = bond_price(M, n, w, ytm_minus, C, f)
    ytm_plus = ytm + dy
    price_plus = bond_price(M, n, w, ytm_plus, C, f)
    mduration = (price_minus - price_plus) / (2 * PV * dy)
    return mduration


def round_dec(num, d=4):
    s = '0.' + '0' * d
    return float(Decimal(str(num)).quantize(Decimal(s), rounding=ROUND_HALF_UP))


def to_dt(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()


def get_nYear_date(date, n=0):
    '''''
    获取N年后或N年前的日期
    '''
    nYear_date = get_nMonth_date(date, n * 12)

    return nYear_date


def calc_ts(settle_date, start_date, end_date, f=1):
    '''
    当前【付息周期】的实际天数
    '''
    dateList = [start_date] + get_dateList_by_StartAndEnd(start_date, end_date, f)

    n = 0
    while True:
        date_range = [dateList[n], dateList[n + 1]]
        if date_range[0] <= settle_date < date_range[1]:
            break
        elif n == len(dateList) - 1:
            break
        else:
            n += 1
            # date_range[1] = CA.FOL(date_range[1])
    ts = (date_range[1] - date_range[0]).days

    return ts, date_range, len(dateList) - 1 - (n)


def calc_ty(settle_date, start_date):
    '''
    当前【计息年度】的实际天数，算头不算尾；
    计息年度是指发行公告中标明的第一个起息日至次一年度对应的同月同日的时间间隔为第一个计息年度，依次类推；
    settle_date:结算日
    start_date:起息日
    '''

    year_beg = start_date
    n = 0

    while True:
        year_range = [year_beg, get_nYear_date(year_beg, 1)]
        if year_range[0] <= settle_date < year_range[1]:
            break
        elif n > 100:
            break
        else:
            year_beg = year_range[1]
            n += 1

    ty = (year_range[1] - year_range[0]).days

    return ty


def CCP_pricing(trade_date, cal_date, strategy_table, base_table, deliverable_base_table, ytm1, ytm2):
    # 日期相关计算
    ccp_deliver_date = base_table['deliver_date']  # CCP的交割日前一日
    #    ccp_next_date=get_nMonth_date(ccp_deliver_date,12)#CCP的最近付息日
    hold_days = (ccp_deliver_date - trade_date).days  # CCP的持有天数

    dl1_start_date = deliverable_base_table['std']['start_interest_date']  # 可交割券1起息日
    dl1_freq = deliverable_base_table['std']['freq']  # 可交割券1付息频率
    dl1_end_date = deliverable_base_table['std']['end_interest_date']  # 可交割券1到期日
    dl1_pre_next_date = get_preAndNextDate_by_trade(cal_date, dl1_start_date, dl1_end_date, dl1_freq)  # 获取最近付息日和上一次付息日
    dl1_pre_date = dl1_pre_next_date[0]  # 上一次付息日
    dl1_next_date = dl1_pre_next_date[1]  # 最近付息日
    dlD1_pre_next_date = get_preAndNextDate_by_trade(ccp_deliver_date, dl1_start_date, dl1_end_date,
                                                     dl1_freq)  # 获取最近付息日和上一次付息日
    dlD1_pre_date = dlD1_pre_next_date[0]  # 交割上一次付息日
    dlD1_next_date = dlD1_pre_next_date[1]  # 交割最近付息日

    dl2_start_date = deliverable_base_table['nonstd']['start_interest_date']  # 可交割券2起息日
    dl2_freq = deliverable_base_table['nonstd']['freq']  # 可交割券2付息频率
    dl2_end_date = deliverable_base_table['nonstd']['end_interest_date']  # 可交割券2到期日
    dl2_pre_next_date = get_preAndNextDate_by_trade(cal_date, dl2_start_date, dl2_end_date, dl2_freq)  # 获取最近付息日和上一次付息日
    dl2_pre_date = dl2_pre_next_date[0]  # 上一次付息日
    dl2_next_date = dl2_pre_next_date[1]  # 最近付息日
    dlD2_pre_next_date = get_preAndNextDate_by_trade(ccp_deliver_date, dl2_start_date, dl2_end_date,
                                                     dl2_freq)  # 获取最近付息日和上一次付息日
    dlD2_pre_date = dlD2_pre_next_date[0]  # 交割上一次付息日
    dlD2_next_date = dlD2_pre_next_date[1]  # 交割最近付息日

    # 可交割券的到期收益率计算
    import math
    n1 = math.ceil((dl1_end_date - cal_date).days / 365 * dl1_freq)
    n2 = math.ceil((dl2_end_date - cal_date).days / 365 * dl2_freq)
    w1 = (dl1_next_date - cal_date).days / (dl1_next_date - dl1_pre_date).days
    w2 = (dl2_next_date - cal_date).days / (dl2_next_date - dl2_pre_date).days

    n1_dl = math.ceil((dl1_end_date - ccp_deliver_date).days / 365 * dl1_freq)
    n2_dl = math.ceil((dl2_end_date - ccp_deliver_date).days / 365 * dl2_freq)
    w1_dl = (dlD1_next_date - ccp_deliver_date).days / (dlD1_next_date - dlD1_pre_date).days
    w2_dl = (dlD2_next_date - ccp_deliver_date).days / (dlD2_next_date - dlD2_pre_date).days

    # 套利组合参数计算
    bond1_full_price = bond_price(100, n1, w1, ytm1, deliverable_base_table['std']['par_rate'],
                                  deliverable_base_table['std']['freq'])
    bond2_full_price = bond_price(100, n2, w2, ytm2, deliverable_base_table['nonstd']['par_rate'],
                                  deliverable_base_table['nonstd']['freq'])

    comb_bond_price = (bond1_full_price + bond2_full_price) / 2
    com_ccp_ytm = (ytm1 + ytm2) / 2

    # 实际资金成本
    Capital_cost_bond = comb_bond_price * strategy_table['capital_cost'] * hold_days / 365

    # 到期交割价

    ccp_deliver_price = bond_price(100, base_table['term'], 1., com_ccp_ytm, base_table['par_rate'], base_table['freq'])
    bond1_deliver_price = bond_price(100, n1_dl, w1_dl, ytm1, deliverable_base_table['std']['par_rate'],
                                     deliverable_base_table['std']['freq'])
    bond2_deliver_price = bond_price(100, n2_dl, w2_dl, ytm2, deliverable_base_table['nonstd']['par_rate'],
                                     deliverable_base_table['nonstd']['freq'])

    # 计算部分（现券持有期收益）
    com_bond_deliver_price = (bond1_deliver_price + bond2_deliver_price) / 2
    r1 = 0 if ccp_deliver_date < dl1_next_date else deliverable_base_table['std']['par_rate'] / \
                                                    deliverable_base_table['std']['freq']
    r2 = 0 if ccp_deliver_date < dl2_next_date else deliverable_base_table['nonstd']['par_rate'] / \
                                                    deliverable_base_table['nonstd']['freq']
    r_hold = com_bond_deliver_price - comb_bond_price + (r1 + r2) / 2

    if strategy_table['type'] == 'IRR':

        IRR = strategy_table['parameter']
        # 计算部分（标债的做市价格）
        ccp_current_price = np.round(ccp_deliver_price - r_hold + (IRR * hold_days * comb_bond_price) / 365, 4)

        # 计算部分（净基差）
        net_basis = np.round(ccp_current_price - ccp_deliver_price + r_hold - Capital_cost_bond, 4)

    elif strategy_table['type'] == 'net_basis':
        net_basis = strategy_table['parameter']

        # 计算部分（标债的做市价格）
        ccp_current_price = ccp_deliver_price + net_basis - r_hold + Capital_cost_bond
        # 计算部分（IRR）
        IRR = np.round((ccp_current_price - ccp_deliver_price + r_hold) / comb_bond_price * 365 / hold_days, 6)
    elif strategy_table['type'] == 'price':
        ccp_current_price = strategy_table['parameter']
        IRR = np.round((ccp_current_price - ccp_deliver_price + r_hold) / comb_bond_price * 365 / hold_days, 6)
        net_basis = np.round(ccp_current_price - ccp_deliver_price + r_hold - Capital_cost_bond, 4)
    else:
        pass

    # 最后策略计算结果
    result_dict = {'ccp_price': ccp_current_price, 'ccp_dl_price': ccp_deliver_price,
                   'r_hold': r_hold, 'avg_return': com_ccp_ytm,
                   'IRR': IRR, 'net_basis': net_basis}

    return result_dict


# %% BOND定价引擎(只定价不含权的固息/零息/贴现债)
class BondPricingEngine:  # to do 统一百分和小数格式
    def __init__(self, settle_date, bond_detail):
        self.M = bond_detail['面值']
        self.coupon_type = bond_detail['息票类型']
        self.coupon = bond_detail['固定利率']
        self.issue_price = bond_detail['发行价格']
        self.settle_date = to_dt(settle_date)  # 结算日（定价时间点)
        self.start_date = to_dt(bond_detail['起息日'])
        self.end_date = to_dt(bond_detail['到期日'])
        self.k = 0  # 债券起息日至计算日的整年数
        self.f_type = bond_detail['息票付息频率']  # 年付息频率
        f_dict = {'半年': 2, '年': 1, '季': 4, '到期': 1, '月': 12}
        if self.f_type in f_dict.keys():
            self.f = f_dict[bond_detail['息票付息频率']]
        self.YTM = float(bond_detail['YTM']) / 100
        self.clean_price = bond_detail['净价']

    def Cal_AccuredInterest(self, n_reserve=7):
        self.d = (self.end_date - self.settle_date).days
        self.t = (self.settle_date - self.start_date).days
        self.ty = calc_ty(self.settle_date, self.start_date)
        self.fv = self.M

        if (self.coupon_type == '贴现') & (self.f_type == '到期'):
            self.coupon = 100 - self.issue_price
            self.ts = (self.end_date - self.start_date).days
            self.ai = self.coupon / self.f * self.t / self.ts

        elif (self.coupon_type == '零息利率') & (self.f_type == '到期'):
            self.ai = self.k * self.coupon + self.coupon / self.ty * self.t
            # 算YTM、clean_price需要的其他变量
            self.fv = self.M + ((self.end_date - self.start_date).days / self.ty) * (self.coupon / 100. * self.M)

        elif self.coupon_type == '固定利率':
            self.ts, self.coupon_period, self.n = calc_ts(self.settle_date, self.start_date, self.end_date, self.f)
            self.t = (self.settle_date - self.coupon_period[0]).days
            self.ai = self.coupon / self.f * self.t / self.ts

            # 算YTM、clean_price需要的其他变量
            self.d = (self.coupon_period[1] - self.settle_date).days

            if self.coupon_period[1] == self.end_date:
                self.fv = self.M + self.coupon / self.f
            else:
                self.coupon_per = self.coupon / 100. * self.M / self.f  # 每期付息
                self.w = self.d / self.ts

        self.ai = round_dec(self.ai, d=n_reserve)

    def Cal_YTM(self, n_reserve=4):
        self.Cal_AccuredInterest()
        self.dirty_price = self.clean_price + self.ai

        if (self.coupon_type == '贴现') & (self.f_type == '到期'):
            self.YTM = (self.fv - self.dirty_price) / self.dirty_price / (self.d / self.ty)

        elif (self.coupon_type == '零息利率') & (self.f_type == '到期'):
            self.YTM = (self.fv - self.dirty_price) / self.dirty_price / (self.d / self.ty)

        elif self.coupon_type == '固定利率':

            if self.coupon_period[1] == self.end_date:
                self.YTM = (self.fv - self.dirty_price) / self.dirty_price / (self.d / self.ty)

            else:
                ytm_func = lambda y: sum(
                    [self.coupon_per / ((1 + y / self.f) ** (self.w + i)) for i in range(self.n)]) + self.M / (
                                                 1 + y / self.f) ** (self.w + self.n - 1) - self.dirty_price
                self.YTM = newton(ytm_func, 0.05)

        self.YTM = round_dec(self.YTM * 100., d=n_reserve)

    def Cal_CleanPrice(self, n_reserve=4):
        self.Cal_AccuredInterest()

        if (self.coupon_type == '贴现') & (self.f_type == '到期'):
            self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)

        elif (self.coupon_type == '零息利率') & (self.f_type == '到期'):
            self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)

        elif self.coupon_type == '固定利率':

            if self.coupon_period[1] == self.end_date:
                self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)

            else:
                self.dirty_price = sum(
                    [self.coupon_per / (1 + self.YTM / self.f) ** (self.w + i) for i in range(self.n)]) + self.M / (
                                               (1 + self.YTM / self.f) ** (self.w + self.n - 1))

        self.clean_price = round_dec(self.dirty_price - self.ai, d=n_reserve)
        self.dirty_price = round_dec(self.dirty_price, d=n_reserve)


class SBFPricingEngine():
    def __init__(self, forward_detail, bond_detail):
        self.r = 0.03  # 远期合约票面利率
        self.rc = forward_detail['融资成本'] / 100
        self.settle_date = to_dt(forward_detail['交割券结算日'])
        self.delivery_date = to_dt(forward_detail['第二交割日'])
        self.N = (self.delivery_date - self.settle_date).days  # 合约存续天数
        self.F = forward_detail['合约价格']
        self.IRR = forward_detail['IRR'] / 100
        self.BNOC = forward_detail['净基差'] / 100

        # 在结算日定价交割券并计算对应参数（应计利息将做节假日调整）
        Bond_settle = BondPricingEngine(forward_detail['交割券结算日'], bond_detail)  # 默认bond_detail包含债券净价
        Bond_settle.Cal_AccuredInterest(n_reserve=7)
        self.f = Bond_settle.f  # 交割券年付息频率
        self.c = Bond_settle.coupon / 100  # 交割券票面利率
        self.settle_clean = Bond_settle.clean_price  # 结算日交割券净价
        self.Ps = round_dec(Bond_settle.clean_price + Bond_settle.ai, d=7)  # 结算日交割券结算价

        # 在交割日计算交割券应计利息（应计利息将做节假日调整）
        Bond_delivery = BondPricingEngine(forward_detail['第二交割日'], bond_detail)
        Bond_delivery.Cal_AccuredInterest(n_reserve=7)
        self.delivery_ai = Bond_delivery.ai
        # 节假日调整
        self.m = (self.delivery_date - util.GetNextSettleDate(Bond_delivery.coupon_period[0])).days
        # 交割月到下一付息月的月份数
        self.x = (Bond_delivery.coupon_period[1].year - self.delivery_date.year) * 12 + (
                    Bond_delivery.coupon_period[1].month - self.delivery_date.month)
        self.I = 0 if self.delivery_date < Bond_settle.coupon_period[1] else self.c * 100. / self.f  # 合约存续期间交割券利息支付
        self.n = Bond_delivery.n  # 在交割日交割券剩余付息次数
        # 交割券转换因子(严格四舍五入至4位)
        self.cf = round_dec((1 / (1 + self.r / self.f) ** (self.x * self.f / 12)) * (
                    self.c / self.f + self.c / self.r + (1 - self.c / self.r) * 1 / (1 + self.r / self.f) ** (
                        self.n - 1)) - self.c / self.f * (1 - self.x * self.f / 12))
        self.Pd = round_dec(self.F * self.cf + self.delivery_ai, d=7)

        util.WriteLog("N[%s],f[%s],c[%s],settle_clean[%s],Ps[%s],delivery_ai[%s],m[%s],x[%s],I[%s],n[%s],cf[%s],Pd[%s]" % (
        self.N, self.f, self.c, self.settle_clean, self.Ps, self.delivery_ai, self.m, self.x, self.I, self.n, self.cf,
        self.Pd))

    # 计算持有期收益
    def Cal_Y(self):
        self.Y = (self.I - self.Ps * self.rc) * self.N / 365

    # 已知F算IRR,BNOC
    def Cal_from_F(self):
        self.Cal_Y()
        self.IRR = round_dec(((self.Pd + self.I - self.Ps) / (self.Ps * self.N - self.I * self.m)) * 365 * 100.)
        self.B = round_dec(self.settle_clean - self.F * self.cf)
        self.BNOC = round_dec(self.settle_clean - self.Y - self.F * self.cf)

    def Cal_from_IRR(self):
        self.Cal_Y()
        self.F = (self.IRR * (self.Ps * self.N - self.I * self.m) / 365 + self.Ps - self.I - self.delivery_ai) / self.cf
        self.BNOC = self.settle_clean - self.Y - self.F * self.cf

        util.WriteLog("Y[%s],F[%s],BNOC[%s]" % (self.Y, self.F, self.BNOC))

    def Cal_from_BNOC(self):
        self.Cal_Y()
        self.F = (self.settle_clean - self.Y - self.BNOC) / self.cf
        self.IRR = (self.F * self.cf + self.delivery_ai + self.I - self.Ps) / (self.Ps * self.N - self.I * self.m) * 365


# 求国债期货和标债远期的合约切换日
def getContractLastTradingDate(today, type):

    firstdate = datetime.date(today.year, today.month, 1)
    firstday = firstdate.weekday()

    # 国债期货，求第二个星期五
    if type == 'TF':

        if firstday % 7 <= 4:
            target = 12 - firstday
        else:
            target = 19 - firstday

        last_date = datetime.date(today.year, today.month, target)
        return last_date

    # 标债远期，求第三个星期三的前一个营业日
    elif type == 'CCP':

        if firstday % 7 <= 2:
            target = 17 - firstday
        else:
            target = 24 - firstday

        last_date = datetime.date(today.year, today.month, target)
        last_date_str = last_date.strftime('%Y%m%d')
        last_date_str = util.GetPreTradingDate(last_date_str)
        last_date = datetime.datetime.strptime(last_date_str, '%Y%m%d').date()
        return last_date

    else:
        return None


# 生成国债期货和标债远期的时间字符串
def generateCodeList(obj, today, type):

    count = 0
    # 国债期货 往后推3个季月
    if type == 'TF':
        count = 3
    # 标债远期 往后推2个季月
    elif type == 'CCP':
        count = 2

    if today.month in (3, 6, 9, 12):
        print(getContractLastTradingDate(today, type))
        start_month = today.month + (0 if today <= getContractLastTradingDate(today, type) else 3)
    else:
        start_month = today.month + (3 - today.month % 3)
    start_year = today.year

    if start_month > 12:
        start_month = start_month % 12
        start_year += 1

    year = start_year
    month = start_month
    timeList = []
    for i in range(0, count):
        timeList.append(str(year)[-2:] + (str(month) if month >= 10 else ("0" + str(month))))
        month += 3
        if month > 12:
            month = month % 12
            year = year + 1

    res = []
    for time in timeList:
        for template in obj.contractTemplate:
            res.append(template.replace('0000', time))

    # 把每个元素封装成元组
    result = [(item,) for item in res]

    codelist = ['\'' + item + '\'' for item in res]
    codelist = '(' + ','.join(codelist) + ')'

    return result, codelist