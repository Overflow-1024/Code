# -*- coding: utf-8 -*-

# 20220920
# 增加CF相关错误日志
# 20220614
# CCP存续天数在最后交易日使用合约交割日-交易日的差,其他使用合约交割日-下一交易日的差
# 20220407
# 增加同步债券市场代码表
# 20220324
# 增加交易所债券信息获取
# 20220314
# 支持国债期货持仓、价格获取
# 修复CCP基准价函数中的BUG
# 20211130
# 新增标债实物交割券自动补齐
# 新增标债实物交割合约基准价计算
# 新增标债实物交割合约对应的债券转换因子计算
# 20211123
# 计算CCP基准价格过滤实物交割券
# 20210927
# CMDS查询无数据不再记录到错误日志
# 20210901
# 增加导入数据数量的校验
# 20210722
# 交易对手表新增KIND_CODE KIND_NAME
# 衡泰持仓汇总内证下的所有股东持仓
# 20210709
# 新增衡泰持仓表字段BondExtendType
# 20210520
# 新增私募可交换债对应正股表导入
# 新增衡泰持仓导入
# 20210518
# Xbond自动订阅券修改剩余期限计算
# 20210517
# Xbond自动订阅券增加剩余期限筛选
# 20210420
# 增加周末读不到CMDS的IRS曲线数据从衡泰资讯获取的逻辑
# 20210420
# 修改Xbond自动订阅券筛选逻辑
# 20210409
# 去除债券类型的过滤
# 20210325
# 新增导入自动订阅XBOND债券信息
# 20210310
# 新增导入中债净价与全价
# 20210304
# 新增删除数据库过期债券数据
# 20210114
# 新增IRS昨日收盘曲线导入
# 20201030
# 中债估值sql添加市场及资产过滤
# 结算价表增加对ADBC品种的支持
# 20201015
# 新增标债品种ADBC
# 交易中心可交割券中债券名称不用加IB
# 20200430
# 去除添加后缀为_T1的合约逻辑
# 20200408
# 计算基准价时,策略表中的参数字段值改为从数据库读取的资金成本,资金成本字段写死0.027
# 20200403
# 资金成本从数据库中获取
# 20200122
# 增加兼容select时报错的逻辑
# 20200120
# 更改CCP结算价与基准价更新到HistoricalPriceTable_CFETS表
# 20191206
# 新增基准券计算逻辑
# 20190823
# BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 中导入DurationString字段的时候把"365D"替换成为"1Y"
# 20190815
# BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 中添加 AccrualBasis字段值导入
# 20190812
# 增加导入XBOND可交易债券
# 20190726
# 债券基础信息导入添加字段
# 20190724
# 过滤到期的债券,给表添加表空间,配置文件添加表空间
# 20190716
# 债券信息中固定利率的取FIXED_COUPON_RATE字段，浮动利率取CURRENT_COUPON_RATE字段
# 20190614
# 过滤IssueCode里有中文的代码
# 20190514
# 修正中债估值到期收益率字段为yield
# 20190507
# 在插入IssueMasterTable时,CCP合约的UnderlyingAssetCode写死CDB
# 在插入BondEvalTable_CFETS的Yield时,netprice需要除以100
# 20190506
# 在插入IssueMarketTable时,借用字段ReserveString填写ProductCode,用于区分不同的品种
from __future__ import division


import datetime
import sys
import numpy as np
import json


from Global import context
import Utils as util
import Calculate as calc

from Sync import SyncBondInfo
from Sync import SyncExchangeBondInfo
from Sync import SyncAvlTradeBondInfo
from Sync import SyncBondConversion
from Sync import SyncBondEval
from Sync import UpdateExchangeBondClosePrice
from Sync import SyncExchangeBondCodeInMarket
from Sync import SyncTFPrePrice
from Sync import SyncCCPInfo
from Sync import SyncCCPClearPrice
from Sync import SyncIRSCurve
from Sync import SyncTradingDate
from Sync import SyncTradeMember
from Sync import SyncTradeParty
from Sync import SyncAutoSubscribeXBOND
from Sync import SyncBondPosition


if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


def ClearBondInfo():

    sql = "DELETE FROM {db}.IssueMarketTable " \
          "WHERE IssueCode IN (" \
          "SELECT IssueCode " \
          "FROM {db}.IssueMasterTable " \
          "WHERE PriorMarket IN ('1','2','9') AND ProductCode IN ('11','40') " \
          "AND ExpirationDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d')" \
          ")".format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.IssueMasterTable " \
          "WHERE PriorMarket IN ('1','2','9') AND ProductCode IN ('11','40') " \
          "AND ExpirationDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d')"\
        .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.BondInfoTable_CFETS".format(db=context.mysql_db)
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "delete FROM {db}.HisBondInfoTable_CFETS  " \
          "where DataDate < (" \
          "SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d')" \
          ")".format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.BondEvalTable_CFETS " \
          "WHERE DataDate < (" \
          "SELECT MAX(DTSDate) FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d') " \
          ") ".format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.BondInfo_XBond_CFETS " \
          "WHERE DataDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d')"\
        .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.XBOND_SubscribeTable_CFETS " \
          "WHERE DataDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d')"\
        .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    # 新添的清空数据语句
    sql = "DELETE FROM {db}.BondCoversionTable_CFETS".format(db=context.mysql_db)
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.BondCodeInMarket_CFETS".format(db=context.mysql_db)
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.TradeMemberTable_CFETS".format(db=context.mysql_db)
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.HisBondCoversionTable_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))"\
          .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.HisBondCodeInMarket_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))" \
        .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)

    sql = "DELETE FROM {db}.HisTradeMemberTable_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))" \
        .format(db=context.mysql_db, cleardays=context.GlobalSettingTable["Clear_Days"])
    context.mysql.delete(sql)
    util.WriteLog(sql)







def CalcCCPPreBasePrice():

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    nextTradingDay = context.gCurrentDate

    sql = "select min(DTSDate) from {db}.CalendarTable_CFETS" \
          " where DTSDate > DATE_FORMAT(now(),'%Y%m%d') and BondTrade = 0".format(db=context.mysql_db)
    util.WriteLog(sql)
    context.mysql.query(sql)
    results = context.mysql.fetchall()

    for row in results:
        nextTradingDay = row[0]

    sql = "SELECT t2.Yield,t2.CCPCode,t2.SettleDate SettleDate_ccp,bondInfo.IssueCode,bondInfo.FirstValueDate," \
          "bondInfo.ExpirationDate ExpirationDate_cbt,bondInfo.CouponFrequency,bondInfo.FixedCouponRate" \
          " from {db}.BondInfoTable_CFETS bondInfo INNER JOIN" \
          " (SELECT eval.Yield,eval.IssueCode BondCode,t.IssueCode CCPCode,t.SettleDate" \
          " from {db}.BondEvalTable_CFETS eval INNER JOIN" \
          " (SELECT delivery.IssueCode,delivery.BondCode,ccp.SettleDate" \
          " from {db}.IssueDeliveryTable_CFETS delivery" \
          " INNER JOIN" \
          " (SELECT IssueCode,SettleDate from {db}.IssueMasterTable_CFETS" \
          " WHERE ListDate <= '{curdate}' and ExpirationDate >= '{curdate}'" \
          " and right(IssueCode,1) != 'P') ccp ON (ccp.IssueCode = delivery.IssueCode)" \
          " WHERE delivery.UpdateDate = '{curdate}') t ON (t.BondCode = eval.IssueCode)" \
          " WHERE eval.DataDate = '{predate}') t2 ON (t2.BondCode = bondInfo.IssueCode)"\
        .format(db=context.mysql_db, curdate=curdate, predate=predate)

    util.WriteLog(sql)
    context.mysql.query(sql)
    results = context.mysql.fetchall()
    if results == ():
        util.WriteErrorLog("计算昨基准0条")

    lt = {}
    for row in results:
        ytm = row[0]
        ccpCode = row[1]
        deliver_date = row[2]
        bondCode = row[3]
        start_interest_date = row[4]
        end_interest_date = row[5]
        freq = row[6]
        par_rate = float(row[7])

        if freq == '年':
            freq = 1
        elif freq == '半年':
            freq = 2
        elif freq == '季度' or freq == '季':
            freq = 4
        elif freq == '月':
            freq = 12
        else:
            freq = 1

        if not lt.has_key(ccpCode):
            lt[ccpCode] = {}

        lt[ccpCode]['deliver_date'] = deliver_date

        if ccpCode.find('CDB3') != -1:
            lt[ccpCode]['term'] = 3
        elif ccpCode.find('CDB5') != -1:
            lt[ccpCode]['term'] = 5
        elif ccpCode.find('CDB10') != -1:
            lt[ccpCode]['term'] = 10
        elif ccpCode.find('ADBC5') != -1:
            lt[ccpCode]['term'] = 5
        elif ccpCode.find('ADBC10') != -1:
            lt[ccpCode]['term'] = 10

        if not lt[ccpCode].has_key('1'):
            lt[ccpCode]['1'] = {}
            lt[ccpCode]['1']['ytm'] = ytm
            lt[ccpCode]['1']['start_interest_date'] = start_interest_date
            lt[ccpCode]['1']['end_interest_date'] = end_interest_date
            lt[ccpCode]['1']['freq'] = freq
            lt[ccpCode]['1']['par_rate'] = par_rate
            lt[ccpCode]['1']['bondCode'] = bondCode
        elif not lt[ccpCode].has_key('2'):
            lt[ccpCode]['2'] = {}
            lt[ccpCode]['2']['ytm'] = ytm
            lt[ccpCode]['2']['start_interest_date'] = start_interest_date
            lt[ccpCode]['2']['end_interest_date'] = end_interest_date
            lt[ccpCode]['2']['freq'] = freq
            lt[ccpCode]['2']['par_rate'] = par_rate
            lt[ccpCode]['2']['bondCode'] = bondCode

    util.WriteLog('CalcCCPPreBasePrice,lt:' + str(lt))
    yyyy = int(nextTradingDay[0:4])
    mm = int(nextTradingDay[4:6])
    dd = int(nextTradingDay[6:8])
    trade_date = datetime.date(yyyy, mm, dd)  # 当前交易日
    cal_date = datetime.date(yyyy, mm, dd)  # 债券结算日

    # 从数据库获取资金成本
    capitalCost = 0.027
    sql = "SELECT Field1 FROM {db}.StrategyEventTable WHERE EventID = 'CCPBase'".format(db=context.mysql_db)
    util.WriteLog(sql)
    context.mysql.query(sql)
    results = context.mysql.fetchall()
    for row in results:
        if util.is_number(row[0]):
            capitalCost = float(row[0]) / 100.0

    util.WriteLog("capitalCost:" + str(capitalCost))

    strategy_table = {'type': 'IRR', 'parameter': capitalCost, 'capital_cost': 0.027}  # 策略参数
    for ccpIssueCode in lt:
        if lt[ccpIssueCode].has_key('1') and lt[ccpIssueCode].has_key('2'):
            deliver_date_1 = lt[ccpIssueCode]['deliver_date']
            base_table = {'term': lt[ccpIssueCode]['term'], 'par_rate': 3.0, 'freq': 1,
                          'deliver_date': datetime.date(int(deliver_date_1[0:4]), int(deliver_date_1[4:6]),
                                                        int(deliver_date_1[6:8]))}  # ccp合约基础信息

            start_interest_date_1 = lt[ccpIssueCode]['1']['start_interest_date']
            end_interest_date_1 = lt[ccpIssueCode]['1']['end_interest_date']
            start_interest_date_2 = lt[ccpIssueCode]['2']['start_interest_date']
            end_interest_date_2 = lt[ccpIssueCode]['2']['end_interest_date']
            # 外汇交易中心官方公布的可交割券基础信息
            deliverable_base_table = {
                'std': {
                    'par_rate': lt[ccpIssueCode]['1']['par_rate'],
                    'freq': lt[ccpIssueCode]['1']['freq'],
                    'start_interest_date': datetime.date(int(start_interest_date_1[0:4]),
                                                         int(start_interest_date_1[4:6]),
                                                         int(start_interest_date_1[6:8])),
                    'end_interest_date': datetime.date(int(end_interest_date_1[0:4]), int(end_interest_date_1[4:6]),
                                                       int(end_interest_date_1[6:8]))
                },
                'nonstd': {
                    'par_rate': lt[ccpIssueCode]['2']['par_rate'],
                    'freq': lt[ccpIssueCode]['2']['freq'],
                    'start_interest_date': datetime.date(int(start_interest_date_2[0:4]),
                                                         int(start_interest_date_2[4:6]),
                                                         int(start_interest_date_2[6:8])),
                    'end_interest_date': datetime.date(int(end_interest_date_2[0:4]), int(end_interest_date_2[4:6]),
                                                       int(end_interest_date_2[6:8]))
                }
            }  # 可交割券基础数据
            ret = calc.CCP_pricing(trade_date, cal_date, strategy_table, base_table, deliverable_base_table,
                              lt[ccpIssueCode]['1']['ytm'], lt[ccpIssueCode]['2']['ytm'])
            util.WriteLog(
                'CalcCCPPreBasePrice,%s,trade_date:%s,strategy_table:%s,base_table:%s,deliverable_base_table:%s,ytm1:%s,ytm2:%s,ret:%s' % (
                ccpIssueCode, str(trade_date), str(strategy_table), str(base_table), str(deliverable_base_table),
                str(lt[ccpIssueCode]['1']['ytm']), str(lt[ccpIssueCode]['2']['ytm']), str(ret)))

            # 防止没有数据不能update,先尝试插入
            ccpClearPriceSQL = "INSERT IGNORE INTO {db}.HistoricalPriceTable_CFETS (IssueCode,MarketCode,DataDate,BasicPrice)" \
                               " VALUES ('{code}','9','{predate}',{price})"\
                .format(db=context.mysql_db, code=ccpIssueCode, predate=predate, price=np.round(ret['ccp_price'], 4))

            util.WriteLog(ccpClearPriceSQL)

            if not context.mysql.updateone(ccpClearPriceSQL):
                util.WriteLog("ERRData-ccpClearPriceSQL")

            # 再执行更新
            ccpClearPriceSQL = "update {db}.HistoricalPriceTable_CFETS set BasicPrice = {price}" \
                               " where DataDate = '{predate}' and IssueCode = '{code}'"\
                .format(db=context.mysql_db, price=np.round(ret['ccp_price'], 4), predate=predate, code=ccpIssueCode)
            util.WriteLog(ccpClearPriceSQL)

            if not context.mysql.updateone(ccpClearPriceSQL):
                util.WriteLog("ERRData-ccpClearPriceSQL")


def CalcDeliverBondCF():

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    # 从数据库获取资金成本
    capitalCost = 2.7
    sql = "SELECT Field1 FROM {db}.StrategyEventTable WHERE EventID = 'CCPBase'".format(db=context.mysql_db)
    util.WriteLog(sql)
    context.mysql.query(sql)
    results = context.mysql.fetchall()

    for row in results:
        if util.is_number(row[0]):
            capitalCost = float(row[0])

    util.WriteLog("capitalCost:" + str(capitalCost))

    sql = "SELECT main_table.IssueCode,main_table.BondCode,hisPrice.ClearingPrice,issueMaster.SettleDate," \
          "(select DTSDate from {db}.CalendarTable_CFETS" \
          " where DTSDate > DATE_FORMAT(now(),'%Y%m%d') and BondTrade = 0 order by DTSDate limit 1)," \
          "bondEval.Yield*100,bondEval.NetPrice,bondEval.FullPrice,convert(bondInfo.CouponType,binary)," \
          "convert(bondInfo.CouponFrequency,binary),bondInfo.FixedCouponRate,bondInfo.IssuerPrice," \
          "bondInfo.FirstValueDate,bondInfo.ExpirationDate,bondInfo.FaceValue" \
          " from {db}.IssueDeliveryTable_CFETS main_table" \
          " left join" \
          " (SELECT IssueCode,ClearingPrice from {db}.HistoricalPriceTable where DataDate ={predate}) hisPrice" \
          " on (main_table.IssueCode = hisPrice.IssueCode)" \
          " left join" \
          " (SELECT IssueCode,Yield,NetPrice,FullPrice from {db}.BondEvalTable_CFETS where DataDate = {predate}) bondEval" \
          " on (main_table.BondCode = bondEval.IssueCode)" \
          " left join {db}.BondInfoTable_CFETS bondInfo" \
          " on (main_table.BondCode = bondInfo.IssueCode)" \
          " left join" \
          " (select * from {db}.IssueMasterTable_CFETS where ExpirationDate >= {curdate}) issueMaster" \
          " on (main_table.IssueCode = issueMaster.IssueCode) where main_table.UpdateDate = {curdate}" \
          " and main_table.IssueCode in" \
          " (select IssueCode from {db}.IssueMasterTable_CFETS where ExpirationDate >= {curdate} and right(IssueCode,1) = 'P')"\
        .format(db=context.mysql_db, curdate=curdate, predate=predate)

    util.WriteLog(sql)
    context.mysql.query(sql)
    results = context.mysql.fetchall()

    lt = {}
    for row in results:
        issueCode = row[0]
        bondCode = row[1]
        ccp_price = row[2]
        expirationDate = row[3]
        settleDate = row[4]
        ytm = row[5]
        bondPrice = row[6]
        bondFullPrice = row[7]
        couponType = row[8]
        couponFrequency = row[9]
        couponRate = row[10]
        issuerPrice = row[11]
        firstValueDate = row[12]
        lastValueDate = row[13]
        faceValue = row[14]

        util.WriteLog(
            "CalcDeliverBondCF,issueCode[%s],bondCode[%s],ccp_price[%s],expirationDate[%s],settleDate[%s],ytm[%s],bondPrice[%s],bondFullPrice[%s],couponType[%s],couponFrequency[%s],couponRate[%s],issuerPrice[%s],firstValueDate[%s],lastValueDate[%s],faceValue[%s]" % (
            issueCode, bondCode, ccp_price, expirationDate, settleDate, ytm, bondPrice, bondFullPrice, couponType,
            couponFrequency, couponRate, issuerPrice, firstValueDate, lastValueDate, faceValue))

        if ccp_price != None and bondPrice != None and ytm != None:
            if util.is_number(ccp_price) and util.is_number(bondPrice) and util.is_number(ytm):
                ccp_price = float(ccp_price)
                bondPrice = float(bondPrice)
                ytm = float(ytm)

                bond_detail = {
                    '息票类型': couponType,
                    '息票付息频率': couponFrequency,
                    '固定利率': couponRate * 1.0,
                    '发行价格': issuerPrice * 1.0,
                    '起息日': firstValueDate[0:4] + '-' + firstValueDate[4:6] + '-' + firstValueDate[6:8],
                    '到期日': lastValueDate[0:4] + '-' + lastValueDate[4:6] + '-' + lastValueDate[6:8],
                    '面值': float(faceValue),
                    '净价': bondPrice,
                    'YTM': ytm
                }

                if settleDate == expirationDate:
                    settleDate = context.gCurrentDate

                forward_detail = {
                    '交割券结算日': settleDate[0:4] + '-' + settleDate[4:6] + '-' + settleDate[6:8],
                    '第二交割日': expirationDate[0:4] + '-' + expirationDate[4:6] + '-' + expirationDate[6:8],
                    # 合约最后交易日后的第二个交易日
                    'IRR': 0,
                    '合约价格': ccp_price,
                    '净基差': 0,
                    '融资成本': capitalCost
                }

                ret = calc.SBFPricingEngine(forward_detail, bond_detail)
                util.WriteLog("forward_detail:[%s],bond_detail:[%s],CF:[%s]" % (
                json.dumps(forward_detail, ensure_ascii=False), json.dumps(bond_detail, ensure_ascii=False),
                str(ret.cf)))

                cf_calc = np.round(ret.cf, 6)
                selectedCF = ''
                sql_selectCF = "select CF from {db}.IssueDeliveryTable_CFETS where UpdateDate = {curdate} and IssueCode = '{icode}' and BondCode = '{bcode}'"\
                    .format(db=context.mysql_db, curdate=curdate, icode=issueCode, bcode=bondCode)

                util.WriteLog(sql_selectCF)
                context.mysql.query(sql_selectCF)
                resultsCF = context.mysql.fetchall()
                for row in resultsCF:
                    if util.is_number(row[0]):
                        selectedCF = np.round(float(row[0]), 6)
                    else:
                        util.WriteErrorLog(
                            "IssueCode[%s]BondCode[%s]同步CF:[%s]为空" % (issueCode, bondCode, str(selectedCF)))

                if selectedCF != '' and selectedCF != cf_calc:
                    util.WriteErrorLog("IssueCode[%s]BondCode[%s]同步CF:[%s]与计算CF:[%s]不一致" % (
                    issueCode, bondCode, str(selectedCF), str(cf_calc)))

                cfSQL = "update {db}.IssueDeliveryTable_CFETS set CF = {cf}" \
                        " where UpdateDate = {curdate} and IssueCode = '{icode}' and BondCode = '{bcode}' AND CF = ''"\
                    .format(db=context.mysql_db, cf=np.round(ret.cf, 6), curdate=curdate, icode=issueCode, bcode=bondCode)

                util.WriteLog(cfSQL)
                if not context.mysql.updateone(cfSQL):
                    util.WriteLog("ERRData-cfSQL")

                ret.IRR = capitalCost / 100.0
                ret.Cal_from_IRR()
                net_basis_cf = ret.BNOC / ret.cf
                util.WriteLog("net_basis_cf:[%s],ccp_price:[%s]" % (str(net_basis_cf), str(ret.F)))

                if not lt.has_key(issueCode):
                    lt[issueCode] = []
                lt[issueCode].append({'bondCode': bondCode, 'net_basis_cf': net_basis_cf, 'ccp_price': ret.F})


    # 获取到CTD券后更新基准价
    for ccpIssueCode, value1 in lt.items():
        sort_ret = sorted(value1, key=lambda k: k['net_basis_cf'])
        util.WriteLog(str(sort_ret))
        if sort_ret[0]:
            # 防止没有数据不能update,先尝试插入
            ccpBasePriceSQL = "INSERT IGNORE INTO {db}.HistoricalPriceTable_CFETS (IssueCode,MarketCode,DataDate,BasicPrice)" \
                              " VALUES ('{code}','9','{predate}',{price})"\
                .format(db=context.mysql_db, code=ccpIssueCode, predate=predate, price=np.round(sort_ret[0]['ccp_price'], 4))

            util.WriteLog(ccpBasePriceSQL)

            if not context.mysql.updateone(ccpBasePriceSQL):
                util.WriteLog("ERRData-ccpBasePriceSQL")

            # 再执行更新
            ccpBasePriceSQL = "update {db}.HistoricalPriceTable_CFETS set BasicPrice = {price} where DataDate = '{predate}' and IssueCode = '{code}'"\
                .format(db=context.mysql_db, price=np.round(sort_ret[0]['ccp_price'], 4), predate=predate, code=ccpIssueCode)
            util.WriteLog(ccpBasePriceSQL)

            if not context.mysql.updateone(ccpBasePriceSQL):
                util.WriteLog("ERRData-ccpBasePriceSQL")


# -------------------------------- 初始化 -------------------------------------

'''
初始化操作都在GlobalManager的init函数中
GlobalManager：负责管理全局变量，包括数据库连接，日志文件指针，配置等
'''

# -------------------------------- 初始化结束 -------------------------------------

# -------------------------------- 处理开始 -------------------------------------


util.WriteLog("Sync START...")

# 清除过期债券数据
ClearBondInfo()
# 同步交易日
SyncTradingDate.sync()
# 同步债券数据
SyncBondInfo.sync()
# 同步交易所债券基础信息
SyncExchangeBondInfo.sync()
# 同步XBOND可交易债
SyncAvlTradeBondInfo.sync()
# 同步私募可交换债与正股关系
SyncBondConversion.sync()
# 同步中债估值
SyncBondEval.sync()
# 同步交易所债券收盘价
UpdateExchangeBondClosePrice.sync()
# 同步交易所债券代码与其他市场代码的转换
SyncExchangeBondCodeInMarket.sync()
# 同步国债期货基准价和结算价
SyncTFPrePrice.sync()
# 同步标债远期数据
SyncCCPInfo.sync()
# 同步CCP昨结算价
SyncCCPClearPrice.sync()
# 同步IRS昨日收盘曲线
SyncIRSCurve.sync()
# 同步交易成员信息
SyncTradeMember.sync()
# 同步交易对手信息（文本解析）
SyncTradeParty.sync()
# 获取自动订阅的XBOND债券
SyncAutoSubscribeXBOND.sync()
# 计算昨基准
CalcCCPPreBasePrice()
# 计算实物交割券的转换因子
CalcDeliverBondCF()
# 同步衡泰持仓
SyncBondPosition.sync()

util.WriteLog("Sync FINISH...")

util.WriteLog("Check START...")

# 数据校验
SyncTradingDate.check()
SyncBondInfo.check()
SyncExchangeBondInfo.check()
SyncAvlTradeBondInfo.check()
SyncBondConversion.check()
SyncBondEval.check(threshold=context.GlobalSettingTable['Threshold_BondEval'])
UpdateExchangeBondClosePrice.check(threshold=context.GlobalSettingTable['Threshold_ExBondClosePrice'])
SyncExchangeBondCodeInMarket.check()
SyncTFPrePrice.check()
SyncCCPInfo.check()
SyncCCPClearPrice.check()
SyncIRSCurve.check()
SyncTradeMember.check()

util.WriteLog("Check FINISH...")

'''
# bond_detail = {
# '息票类型':'固定利率',
# '息票付息频率':'年',
# '固定利率':3.05,
# '发行价格':100.0,
# '起息日':'2021-03-04',
# '到期日':'2023-03-04',
# '面值':100.0,
# '净价':100.5707,
# 'YTM':2.575
# }

# forward_detail = {
# '交割券结算日':'2021-12-06',
# '第二交割日':'2022-03-16', #合约最后交易日后的第二个交易日
# 'IRR':0,
# '合约价格':100.3775,
# '净基差':0,
# '融资成本':5.0001
# }

# ret = SBFPricingEngine(forward_detail, bond_detail)
# util.WriteLog("forward_detail:[%s],bond_detail:[%s],CF:[%s]" % (json.dumps(forward_detail,ensure_ascii=False),json.dumps(bond_detail,ensure_ascii=False),str(ret.cf)))


# ret.IRR = 5.0001/100.0
# ret.Cal_from_IRR()
# net_basis_cf = ret.BNOC / ret.cf
# util.WriteLog("BNOC:[%s],cf:[%s],net_basis_cf:[%s],ccp_price:[%s]" % (str(ret.BNOC),str(ret.cf),str(net_basis_cf),str(ret.F)))
'''

util.WriteLog('Sync END...')

# -------------------------------- 处理结束 -------------------------------------

'''
退出操作都在GlobalManager的del函数中
包括关闭数据库连接，关闭日志文件指针等
'''



