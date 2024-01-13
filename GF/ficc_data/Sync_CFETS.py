# -*- coding: utf-8 -*-
#20221109
#增加导入衡泰债券持仓:持仓有的债券对应的风控主体关联债券
#20221102
#增加标债远期交割券的校验
#20221010
#同步国债期货基准价和结算价时,使用交易所日程表
#20220920
#增加CF相关错误日志
#20220614
#CCP存续天数在最后交易日使用合约交割日-交易日的差,其他使用合约交割日-下一交易日的差
#20220407
#增加同步债券市场代码表
#20220324
#增加交易所债券信息获取
#20220314
#支持国债期货持仓、价格获取
#修复CCP基准价函数中的BUG
#20211130
#新增标债实物交割券自动补齐
#新增标债实物交割合约基准价计算
#新增标债实物交割合约对应的债券转换因子计算
#20211123
#计算CCP基准价格过滤实物交割券
#20210927
#CMDS查询无数据不再记录到错误日志
#20210901
#增加导入数据数量的校验
#20210722
#交易对手表新增KIND_CODE KIND_NAME
#衡泰持仓汇总内证下的所有股东持仓
#20210709
#新增衡泰持仓表字段BondExtendType
#20210520
#新增私募可交换债对应正股表导入
#新增衡泰持仓导入
#20210518
#Xbond自动订阅券修改剩余期限计算
#20210517
#Xbond自动订阅券增加剩余期限筛选
#20210420
#增加周末读不到CMDS的IRS曲线数据从衡泰资讯获取的逻辑
#20210420
#修改Xbond自动订阅券筛选逻辑
#20210409
#去除债券类型的过滤
#20210325
#新增导入自动订阅XBOND债券信息
#20210310
#新增导入中债净价与全价
#20210304
#新增删除数据库过期债券数据
#20210114
#新增IRS昨日收盘曲线导入
#20201030
#中债估值sql添加市场及资产过滤
#结算价表增加对ADBC品种的支持
#20201015
#新增标债品种ADBC
#交易中心可交割券中债券名称不用加IB
#20200430
#去除添加后缀为_T1的合约逻辑
#20200408
#计算基准价时,策略表中的参数字段值改为从数据库读取的资金成本,资金成本字段写死0.027
#20200403
#资金成本从数据库中获取
#20200122
#增加兼容select时报错的逻辑
#20200120
# 更改CCP结算价与基准价更新到HistoricalPriceTable_CFETS表
#20191206
# 新增基准券计算逻辑
#20190823
# BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 中导入DurationString字段的时候把"365D"替换成为"1Y"
#20190815
# BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 中添加 AccrualBasis字段值导入
#20190812
#增加导入XBOND可交易债券
#20190726
#债券基础信息导入添加字段
#20190724
#过滤到期的债券,给表添加表空间,配置文件添加表空间
#20190716
#债券信息中固定利率的取FIXED_COUPON_RATE字段，浮动利率取CURRENT_COUPON_RATE字段
#20190614
#过滤IssueCode里有中文的代码
#20190514
#修正中债估值到期收益率字段为yield
#20190507
#在插入IssueMasterTable时,CCP合约的UnderlyingAssetCode写死CDB
#在插入BondEvalTable_CFETS的Yield时,netprice需要除以100
#20190506
#在插入IssueMarketTable时,借用字段ReserveString填写ProductCode,用于区分不同的品种
from __future__ import division

import warnings

import cx_Oracle
import datetime
import os
import MySQLdb
import re
import time
import sys
import numpy as np
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
import json

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
    

def WriteLog(log):
    global file
    nowTime = time.strftime('%H:%M:%S',time.localtime(time.time()))
    file.write(nowTime+"    "+log+"\n")

def WriteErrorLog(log):
    global errorFile
    nowTime = time.strftime('%H:%M:%S',time.localtime(time.time()))
    errorFile.write(nowTime+"    "+log+"\n")
def GetPreTradingDate():
    preTradingDate = gCurrentDate
    #获取上一交易日
    sql = "select max(DTSDate) from CalendarTable_CFETS where DTSDate < '%s' AND BondTrade = '0'" % (gCurrentDate)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    for row in results:
       preTradingDate = row[0]
    WriteLog("GetPreTradingDate,preTradingDate:"+preTradingDate)
    return preTradingDate

#获取下一个结算日
def GetNextSettleDate(obj):
    nextSettleDate = obj.strftime('%Y%m%d')
    sql = "select min(DTSDate) from CalendarTable_CFETS where DTSDate >= '%s' AND BondTrade = '0'" % (nextSettleDate)
    WriteLog('GetNextSettleDate,sql:'+sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    for row in results:
       nextSettleDate = row[0]
       
    WriteLog("GetNextSettleDate,nextSettleDate:"+nextSettleDate)
    return datetime.date(int(nextSettleDate[0:4]),int(nextSettleDate[4:6]),int(nextSettleDate[6:8]))

def GetPreTradingDateExchange():
    preTradingDate = gCurrentDate
    #获取上一交易日
    sql = "select max(DTSDate) from CalendarTable where DTSDate < '%s' AND DayOffFlag = '0'" % (gCurrentDate)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    for row in results:
       preTradingDate = row[0]
    WriteLog("GetPreTradingDate,preTradingDate:"+preTradingDate)
    return preTradingDate


#查询Oracle
def ClearBondInfo():
    global GlobalSettingTable
    sql = "DELETE FROM IssueMarketTable WHERE IssueCode IN (SELECT IssueCode FROM IssueMasterTable WHERE PriorMarket IN ('1','2','9') AND ProductCode IN ('11','40') AND ExpirationDate < date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d'))" % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    sql = "DELETE FROM IssueMasterTable WHERE PriorMarket IN ('1','2','9') AND ProductCode IN ('11','40') AND ExpirationDate < date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d')" % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    sql = "DELETE FROM BondInfoTable_CFETS "
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    sql = "delete FROM HisBondInfoTable_CFETS  where DataDate < (SELECT MAX(DTSDate) FROM CalendarTable_CFETS where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d') )" % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    
    sql = "DELETE FROM BondEvalTable_CFETS WHERE DataDate < (SELECT MAX(DTSDate) FROM CalendarTable_CFETS where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d') ) " % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    sql = "DELETE FROM BondInfo_XBond_CFETS WHERE DataDate < date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d')" % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)
    
    sql = "DELETE FROM XBOND_SubscribeTable_CFETS WHERE DataDate <  date_format(DATE_ADD(NOW(),INTERVAL -%s DAY),'%%Y%%m%%d')" % (GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)

    # 新添的清空数据语句
    sql = "DELETE FROM {db}.BondCoversionTable_CFETS".format(db='dtsdb')
    MySqlCR.execute(sql)
    WriteLog(sql)

    sql = "DELETE FROM {db}.BondCodeInMarket_CFETS".format(db='dtsdb')
    MySqlCR.execute(sql)
    WriteLog(sql)

    sql = "DELETE FROM {db}.TradeMemberTable_CFETS".format(db='dtsdb')
    MySqlCR.execute(sql)
    WriteLog(sql)

    sql = "DELETE FROM {db}.HisBondCoversionTable_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))" \
        .format(db='dtsdb', cleardays=GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)

    sql = "DELETE FROM {db}.HisBondCodeInMarket_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))" \
        .format(db='dtsdb', cleardays=GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)

    sql = "DELETE FROM {db}.HisTradeMemberTable_CFETS " \
          "where DataDate < " \
          "(SELECT MAX(DTSDate) " \
          "FROM {db}.CalendarTable_CFETS " \
          "where BondTrade = '0' AND DTSDate < date_format(DATE_ADD(NOW(),INTERVAL -{cleardays} DAY),'%%Y%%m%%d'))" \
        .format(db='dtsdb', cleardays=GlobalSettingTable["Clear_Days"])
    MySqlCR.execute(sql)
    WriteLog(sql)

#查询Oracle
def SyncBondInfo():
    global GlobalSettingTable
    sql = "select I_CODE,I_NAME,BOND_TYPE,FACE_VALUE,MATURITY_DATE,BOND_TERM,DURATION,AI,CR_1,COUPON_TYPE,COUPON_FREQUENCY,FIRST_VALUE_DATE,FIRST_PAYMENT_DATE,FIXED_COUPON_RATE,EXERCISE_TYPE_1,SETTLCURRENCY,CUSTODIAN_NAME,CURRENT_COUPON_RATE,DELISTING_DATE,SECURITYTYPEID,TERMTOMATURITYSTRING,ISSUERSHORTPARTYID,ACCRUAL_BASIS,ISSUE_PRICE,LISTING_DATE from %sTTRD_CFETS_B_BOND where to_char(sysdate,'yyyy-MM-dd') <= MATURITY_DATE " %(GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
    
        reportCode = res[0]
        count += 1
        
        zhmodel = re.compile(u'[\u4e00-\u9fa5]')
        if not zhmodel.search(reportCode):
            issueCode = 'IB'+res[0]
            issueName = res[1]
            bondType = res[2]
            faceValue = res[3]
            expirationDate = res[4][0:4]+res[4][5:7]+res[4][8:10]
            tradeLimitDays = res[5]
            duration = res[6]
            ai = res[7]
            CR = res[8]
            couponType = res[9]
            couponFrequency = res[10]
            firstValueDate = res[11][0:4]+res[11][5:7]+res[11][8:10]
            firstPaymentDate = res[12][0:4]+res[12][5:7]+res[12][8:10]
            fixedCouponRate = res[13]
            exerciseType1 = res[14]
            settlCurrency = res[15]
            custodianName = res[16]
            currentCouponRate = res[17]
            delistingDate = res[18][0:4]+res[18][5:7]+res[18][8:10]
            bondTypeID = res[19]
            durationString = str(res[20]).replace("365D","1Y")
            issueShortPartyID = res[21]
            accrualBasis = res[22]
            issuerPrice = res[23]
            listingDate = res[24][0:4]+res[24][5:7]+res[24][8:10]
            
            if str(exerciseType1) == 'None':
                exerciseType1 = ''
            if str(ai) == 'None':
                ai = 0.00
            if str(CR) == 'None':
                CR = ''
            if str(fixedCouponRate) == 'None':
                fixedCouponRate = 0.00
            
            if str(currentCouponRate) == 'None':
                currentCouponRate = 0.00
            
            if couponType == '浮动利率':
                fixedCouponRate = currentCouponRate
            
            today = datetime.date.today()
            d2=datetime.date(int(res[4][0:4]),int(res[4][5:7]),int(res[4][8:10]))
            duration = (d2-today).days

            #插入BondInfoTable
            bondInfoSQL = "REPLACE INTO dtsdb.BondInfoTable_CFETS(IssueCode,MarketCode,IssueName,ReportCode,ProductCode,BondType,BondTypeID,FaceValue,ExpirationDate,ListingDate,DelistingDate,TradeLimitDays,Duration,DurationString,AI,CR,CouponType,CouponFrequency,AccrualBasis,FirstValueDate,FirstPaymentDate,FixedCouponRate,ExerciseType1,SettlCurrency,CustodianName,IssuerShortPartyID,IssuerPrice,TheoryPrice,ClearingSpeed) VALUES ('%s','9','%s','%s','40','%s','%s',%s,'%s','%s','%s','%s','%d','%s',%s,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s',0,1);" % (issueCode,issueName,reportCode,bondType,bondTypeID,faceValue,expirationDate,listingDate,delistingDate,tradeLimitDays,duration,durationString,ai,CR,couponType,couponFrequency,accrualBasis,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName,issueShortPartyID,issuerPrice)
            WriteLog(bondInfoSQL)        
            try:
               MySqlCR.execute(bondInfoSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog('ERRData-bondInfoSQL:issueCode[%s],issueName[%s],bondType[%s],faceValue[%s],expirationDate[%s],tradeLimitDays[%s],duration[%s],ai[%s],CR[%s],couponType[%s],couponFrequency[%s],firstValueDate[%s],firstPaymentDate[%s],fixedCouponRate[%s],exerciseType1[%s],settlCurrency[%s],custodianName[%s],listingDate[%s],delistingDate[%s],bondTypeID[%s],durationString[%s],issueShortPartyID[%s],issuerPrice[%s]'% (issueCode,issueName,bondType,faceValue,expirationDate,tradeLimitDays,duration,ai,CR,couponType,couponFrequency,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName,listingDate,delistingDate,bondTypeID,durationString,issueShortPartyID,issuerPrice))
            
            #插入HisBondInfoTable
            hisBondInfoSQL = "REPLACE INTO dtsdb.HisBondInfoTable_CFETS(IssueCode,MarketCode,IssueName,ReportCode,ProductCode,BondType,BondTypeID,FaceValue,ExpirationDate,ListingDate,DelistingDate,TradeLimitDays,Duration,DurationString,AI,CR,CouponType,CouponFrequency,AccrualBasis,FirstValueDate,FirstPaymentDate,FixedCouponRate,ExerciseType1,SettlCurrency,CustodianName,IssuerShortPartyID,IssuerPrice,DataDate) VALUES ('%s','9','%s','%s','40','%s','%s',%s,'%s','%s','%s','%s','%d','%s',%s,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s',%s,%s);" % (issueCode,issueName,reportCode,bondType,bondTypeID,faceValue,expirationDate,listingDate,delistingDate,tradeLimitDays,duration,durationString,ai,CR,couponType,couponFrequency,accrualBasis,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName,issueShortPartyID,issuerPrice,gCurrentDate)
            WriteLog(hisBondInfoSQL)        
            try:
               MySqlCR.execute(hisBondInfoSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog('ERRData-hisBondInfoSQL:issueCode[%s],issueName[%s],bondType[%s],faceValue[%s],expirationDate[%s],tradeLimitDays[%s],duration[%s],ai[%s],CR[%s],couponType[%s],couponFrequency[%s],firstValueDate[%s],firstPaymentDate[%s],fixedCouponRate[%s],exerciseType1[%s],settlCurrency[%s],custodianName[%s],delistingDate[%s],bondTypeID[%s],durationString[%s],issueShortPartyID[%s],issuerPrice[%d]'% (issueCode,issueName,bondType,faceValue,expirationDate,tradeLimitDays,duration,ai,CR,couponType,couponFrequency,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName,delistingDate,bondTypeID,durationString,issueShortPartyID,issuerPrice))
            
            
            #插入IssueMasterTable
            issueMasterSQL = "REPLACE INTO dtsdb.IssueMasterTable(IssueCode,IssueShortName,IssueShortLocalName,IssueLongName,IssueLongLocalName,ProductCode,Currency,MarketSectorCode,PriorMarket,UnderlyingAssetCode,PutCall,ContractMonth,OtherContractMonth,StrikePrice,ExpirationDate,FaceValue,EstimateFaceValue,NearMonthIssueCode,OtherMonthIssueCode,GrantedRatio,UnderlyingIssueCode,ExRightsDate,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,Shares,Status,UnitAmount,AmountLeast,AmountMost,BalanceMost,Tick,RaisingLimitRate,DecliningLimitRate,FareRule,PromptRule,TradingMonth,OpenDropFareRatio,OpenDropFareBalance,DropCuFareRatio,DropCuFareBalance,DeliverFareRatio,DeliverFareBalance,SpeculationBailRatio,SpeculationBailBalance,HedgeBailRatio,HedgeBailBalance,ContractSize) VALUES ('%s','%s','%s','%s','%s','40','%s',NULL,'9','0',NULL,'',NULL,0,'%s',0,0,NULL,NULL,0,'',NULL,NULL,now(),now(),0,0,NULL,0,0,0,0,0,0,0,NULL,NULL,NULL,0,0,0,0,0,0,0,0,0,0,1);" % (issueCode,reportCode,issueName,reportCode,issueName,settlCurrency,expirationDate)
            WriteLog(issueMasterSQL)
            try:
               MySqlCR.execute(issueMasterSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog('ERRData-issueMasterSQL:issueCode[%s],issueName[%s],bondType[%s],faceValue[%s],expirationDate[%s],tradeLimitDays[%s],duration[%s],ai[%s],CR[%s],couponType[%s],couponFrequency[%s],firstValueDate[%s],firstPaymentDate[%s],fixedCouponRate[%s],exerciseType1[%s],settlCurrency[%s],custodianName[%s]'% (issueCode,issueName,bondType,faceValue,expirationDate,tradeLimitDays,duration,ai,CR,couponType,couponFrequency,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName))
               
            
            #插入IssueMarketTable
            issueMarketSQL = "REPLACE INTO dtsdb.IssueMarketTable(IssueCode,MarketCode,MarketSystemCode,EvenLot,CashDeliveryFlag,LoanableIssueFlag,MarginIssueFlag,ListedDate,MarketSectionCode,SessionPatternID,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,ApplicationStopFlag) VALUES ('%s','9','',1, NULL,NULL,NULL,'%s','0',NULL,'40',now(),now(),0,NULL);" % (issueCode,listingDate)
            WriteLog(issueMarketSQL)        
            try:
               MySqlCR.execute(issueMarketSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog('ERRData-issueMarketSQL:issueCode[%s],issueName[%s],bondType[%s],faceValue[%s],expirationDate[%s],tradeLimitDays[%s],duration[%s],ai[%s],CR[%s],couponType[%s],couponFrequency[%s],firstValueDate[%s],firstPaymentDate[%s],fixedCouponRate[%s],exerciseType1[%s],settlCurrency[%s],custodianName[%s]'% (issueCode,issueName,bondType,faceValue,expirationDate,tradeLimitDays,duration,ai,CR,couponType,couponFrequency,firstValueDate,firstPaymentDate,fixedCouponRate,exerciseType1,settlCurrency,custodianName))
               
            #插入HistoricalPriceTable
            historicalPriceSQL = "INSERT IGNORE INTO dtsdb.HistoricalPriceTable(IssueCode,MarketCode,DataDate,MarkPrice,ClosePrice,AdjustedClosePrice,OpenPrice,HighPrice,LowPrice,Volume,UpperLimitPrice,LowerLimitPrice,MMLNBestBid,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,WeeklyHighPrice,WeeklyLowPrice,MonthlyHighPrice,MonthlyLowPrice,QuarterHighPrice,QuarterLowPrice,Psychological,WeightGiftCounter,WeightSellCounter,WeightSellPrice,WeightDividend,WeightIncCounter,WeightOwnerShip,WeightFreeCounter,ClearingPrice,MMBestBid1) select '%s','9',(select max(DTSDate) from CalendarTable_CFETS where DTSDate < '%s' AND BondTrade = '0'), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, now(), now(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 from dual" % (issueCode,gCurrentDate)
            WriteLog(historicalPriceSQL)        
            try:
               MySqlCR.execute(historicalPriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-historicalPriceSQL")
            
            
        else:
            err_info = "%s是中文" % (reportCode)
            WriteLog(err_info)
           
    if count == 0:
        WriteErrorLog("同步债券数据0条")
           
     
def SyncTradingDate():
    dateTable = {}
    maxDate = gCurrentDate
    global GlobalSettingTable
    #1.获取交易日
    sql = "select MARKET,HOLIDAY,UPDATETIME from %sTTRD_CFETS_B_TRADE_HOLIDAY where MARKET = '现券'" %(GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        market = res[0]
        holiday = res[1][0:4]+res[1][5:7]+res[1][8:10]
        updateTime = res[2]
        count += 1
        
        #WriteLog("TTRD_CFETS_B_TRADE_HOLIDAY,holiday[%s],updateTime[%s]" % (holiday,updateTime))
        if not dateTable.has_key(holiday):
            dateTable[holiday] = {}
        
        dateTable[holiday]['BondTrade'] = 1
        if maxDate < holiday:
            maxDate = holiday
    if count == 0:
        WriteErrorLog("同步交易日, 获取交易日数据0条")
        
        
    #2.获取结算日
    sql = "select MARKET,HOLIDAY,UPDATETIME from %sTTRD_CFETS_B_SETTLE_HOLIDAY where MARKET = '现券'" %(GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        market = res[0]
        holiday = res[1][0:4]+res[1][5:7]+res[1][8:10]
        updateTime = res[2]
        count += 1
        
        #WriteLog("TTRD_CFETS_B_SETTLE_HOLIDAY,holiday[%s],updateTime[%s]" % (holiday,updateTime))
        if not dateTable.has_key(holiday):
            dateTable[holiday] = {}
            
        
        dateTable[holiday]['BondSettle'] = 1
        if maxDate < holiday:
            maxDate = holiday
    if count == 0:
        WriteErrorLog("同步交易日, 获取结算日数据0条")
    
    #3.生成未来5年的日历
    for i in range (0,5*365):
        date_tmp = str(datetime.date.today() - datetime.timedelta(days=-i))
        date = date_tmp[0:4]+date_tmp[5:7]+date_tmp[8:10]
        
        
        trade = 0
        settle = 0
        
        #如果有日程表数据，取日程表数据值
        if dateTable.has_key(date):
            if dateTable[date].has_key('BondTrade'):
                trade = 1
                
            if dateTable[date].has_key('BondSettle'):
                settle = 1
        #如果大于获取到的最大假日,根据周六日赋值默认
        elif date > maxDate:
            w = datetime.datetime(int(date_tmp[0:4]),int(date_tmp[5:7]),int(date_tmp[8:10])).strftime("%w")
            if w == '6' or w == '0':
                trade = 1
                settle = 1
        
        #插入CalendarTable_CFETS
        calendarSQL = "REPLACE INTO dtsdb.CalendarTable_CFETS(DTSDate,MarketCode,BondTrade,BondSettle,CreateTime,TimeStamp) VALUES ('%s','9','%s','%s',now(),now())" % (date,trade,settle)
        WriteLog(calendarSQL)        
        try:
           MySqlCR.execute(calendarSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-calendarSQL")
    
def SyncTradeMember():
    global  GlobalSettingTable
    sql = "SELECT t.MEMBER_ID,t.ORGCODE,t.CH_NAME,t.CH_SHORT_NAME,t.EN_NAME,EN_SHORT_NAME,t1.T_CODE,t1.T_NAME from %sTTRD_CFETS_B_TRADE_MEMBER t LEFT JOIN (SELECT ti.T_CODE,ti.T_NAME,tc.BANKCODE,tc.ORGCODE from %sTTRD_OTC_COUNTERPARTY tc LEFT JOIN %sTTRD_OTC_INSTITUTIONTYPE ti on tc.CLIENTKIND = ti.T_CODE WHERE tc.PARTY_STATUS = '1' and tc.BANKCODE IS NOT NULL) t1 ON (t1.ORGCODE = t.ORGCODE)" %(GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        memberID = res[0]
        orgCode = res[1]
        CNName = res[2]
        CNShortName = res[3]
        ENName = res[4]
        ENShortName = res[5]
        kindCode = res[6]
        kindName = res[7]
        count += 1
        
#        CNName.replace("'", "\')
#        CNShortName.replace("'", "\\'")
#        ENName.replace("'", "\\'")
#        ENShortName.replace("'", "\\'")
        
        memberSQL = "REPLACE INTO dtsdb.TradeMemberTable_CFETS(MEMBER_ID,ORGCODE,CH_NAME,CH_SHORT_NAME,EN_NAME,EN_SHORT_NAME,KIND_CODE,KIND_NAME,TimeStamp) VALUES ('%s','%s',\"%s\",\"%s\",\"%s\",\"%s\",'%s','%s',now())" % (memberID,orgCode,CNName,CNShortName,ENName,ENShortName,kindCode,kindName)
        WriteLog(memberSQL)        
        try:
           MySqlCR.execute(memberSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-memberSQL")
    if count == 0:
        WriteErrorLog("同步交易成员信息0条")
       
def SyncTradeParty():

    global GlobalSettingTable
    sql = "SELECT PARTYID,PARTYNAME,PARTYNAME_SHORT From %sTTRD_OTC_COUNTERPARTY where PARTY_STATUS='1'" % (GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break

        memberID = res[0]
        CNName = res[1]
        CNShortName = res[2]
        count += 1

        partySQL = "REPLACE INTO dtsdb.TradePartyTable_CFETS (PARTY_ID,CH_NAME,CH_SHORT_NAME,TimeStamp) VALUES ('%s','%s','%s',now())" % (memberID,CNName,CNShortName)
        WriteLog(partySQL)
        try:
            MySqlCR.execute(partySQL)
            MySQlDB.commit()
        except:
            MySQlDB.rollback()
            WriteLog("ERRData-partySQL")
    if count == 0:
        WriteErrorLog("同步交易对手信息0条")
       
def SyncCCPInfo():
    #1.同步CCP基础信息
    global  GlobalSettingTable
    sql = "select BENCH_MARK_PRICE,ISSUE_DATE,I_CODE,I_NAME,LAST_TRD_DATE,CLEARING_METHOD,TRAD_SES_END_TIME,SETTLDATE,FACEVALUE,SECURITY_STATUS,MINQTY,MAXQTY from %sTTRD_CFETS_B_SBF" % (GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        count += 1
        benchMarketPrice = res[0]#挂牌基准价
        listDate = res[1][0:4]+res[1][5:7]+res[1][8:10]#上市日期
        reportCode = res[2]#代码
        issueCode = "IB"+reportCode
        issueName = res[3]#名称
        expirationDate = res[4][0:4]+res[4][5:7]+res[4][8:10]#到期日
        clearingMethod = res[5]#清算方式
        expirationTime = res[6][9:11]+res[6][12:14]+res[6][15:17]#到期时间 
        settleDate = res[7][0:4]+res[7][5:7]+res[7][8:10]#交割日
        faceValue = int(res[8])*10000#合约面值（万元）
        status = res[9]#合约状态
        minQuantity = int(res[10])#最小单笔报价量（手）
        maxQuantity = int(res[11])#最大单笔报价量（手）
        underlyIssueCode = (re.split('[_]+',reportCode))[0]
        if reportCode[-1:] == 'P':
            underlyIssueCode = underlyIssueCode + 'P'
        
        ccpSQL = "REPLACE INTO dtsdb.IssueMasterTable_CFETS(IssueCode,MarketCode,ReportCode,IssueName,ProductCode,UnderlyingIssueCode,FaceValue,MinQuantity,MaxQuantity,ListDate,ExpirationDate,ExpirationTime,SettleDate,Tick,ContractSize,Status,ClearingMethod,BenchMarkPrice,CreateTime,UpdateTime) VALUES ('%s','9','%s','%s','38','%s',%d,%d,%d,'%s','%s','%s','%s',0.0001,1,'%s','%s',%f,now(),now())" % (issueCode,reportCode,issueName,underlyIssueCode,faceValue,minQuantity,maxQuantity,listDate,expirationDate,expirationTime,settleDate,status,clearingMethod,benchMarketPrice)
        WriteLog(ccpSQL)
        try:
           MySqlCR.execute(ccpSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-ccpSQL")
       
  
        #插入IssueMasterTable
        issueMasterSQL = "REPLACE INTO dtsdb.IssueMasterTable(IssueCode,IssueShortName,IssueShortLocalName,IssueLongName,IssueLongLocalName,ProductCode,Currency,MarketSectorCode,PriorMarket,UnderlyingAssetCode,PutCall,ContractMonth,OtherContractMonth,StrikePrice,ExpirationDate,FaceValue,EstimateFaceValue,NearMonthIssueCode,OtherMonthIssueCode,GrantedRatio,UnderlyingIssueCode,ExRightsDate,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,Shares,Status,UnitAmount,AmountLeast,AmountMost,BalanceMost,Tick,RaisingLimitRate,DecliningLimitRate,FareRule,PromptRule,TradingMonth,OpenDropFareRatio,OpenDropFareBalance,DropCuFareRatio,DropCuFareBalance,DeliverFareRatio,DeliverFareBalance,SpeculationBailRatio,SpeculationBailBalance,HedgeBailRatio,HedgeBailBalance,ContractSize) VALUES ('%s','%s','%s','%s','%s','38','%s',NULL,'9','CDB',NULL,'',NULL,0,'%s',0,0,NULL,NULL,0,'',NULL,NULL,now(),now(),0,0,NULL,0,0,0,0,0,0,0,NULL,NULL,NULL,0,0,0,0,0,0,0,0,0,0,1);" % (issueCode,reportCode,issueName,reportCode,issueName,'RMB',expirationDate)
        WriteLog(issueMasterSQL)
        try:
           MySqlCR.execute(issueMasterSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-issueMasterSQL")
        
        #插入IssueMarketTable
        issueMarketSQL = "REPLACE INTO dtsdb.IssueMarketTable(IssueCode,MarketCode,MarketSystemCode,EvenLot,CashDeliveryFlag,LoanableIssueFlag,MarginIssueFlag,ListedDate,MarketSectionCode,SessionPatternID,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,ApplicationStopFlag) VALUES ('%s','9','',1, NULL,NULL,NULL,'%s','0',NULL,'38',now(),now(),0,NULL);" % (issueCode,listDate)
        WriteLog(issueMarketSQL)        
        try:
           MySqlCR.execute(issueMarketSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-issueMarketSQL")
   
        #插入HistoricalPriceTable
        historicalPriceSQL = "INSERT IGNORE INTO dtsdb.HistoricalPriceTable(IssueCode,MarketCode,DataDate,MarkPrice,ClosePrice,AdjustedClosePrice,OpenPrice,HighPrice,LowPrice,Volume,UpperLimitPrice,LowerLimitPrice,MMLNBestBid,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,WeeklyHighPrice,WeeklyLowPrice,MonthlyHighPrice,MonthlyLowPrice,QuarterHighPrice,QuarterLowPrice,Psychological,WeightGiftCounter,WeightSellCounter,WeightSellPrice,WeightDividend,WeightIncCounter,WeightOwnerShip,WeightFreeCounter,ClearingPrice,MMBestBid1) select '%s','9',(select max(DTSDate) from CalendarTable_CFETS where DTSDate < '%s' AND BondTrade = '0'), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, now(), now(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 from dual" % (issueCode,gCurrentDate)
        WriteLog(historicalPriceSQL)        
        try:
           MySqlCR.execute(historicalPriceSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-historicalPriceSQL")
         
    if count == 0:
        WriteErrorLog("同步标债远期数据, CCP基础信息0条")
    #2.同步CCP的可交割券数据
    flag = 0
    sql = "select FULL_SYMBOL,I_CODE,I_NAME from %sTTRD_CFETS_B_SBF_DELIVERABLE" %(GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        ccpIssueCode = "IB"+res[0]#CCP代码
        bondIssueCode = "IB"+res[1]#CBT代码
        bondName = res[2]#CBT名称
        count += 1
        
        if ccpIssueCode[-1:] == 'P':
            flag = 1
        
        deliverySQL = "INSERT IGNORE INTO IssueDeliveryTable_CFETS(IssueCode,BondCode,BondName,UpdateDate,StandardBond,CF) values('%s','%s','%s','%s','','')" % (ccpIssueCode,bondIssueCode,bondName,gCurrentDate)
        WriteLog(deliverySQL)        
        try:
           MySqlCR.execute(deliverySQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-deliverySQL")
        
    if count == 0:
        WriteErrorLog("同步标债远期数据, CCP的可交割券数据0条")
    
    #3.如果实物交割合约没有提供对应的交割券信息,则按照逻辑产生
    # if flag == 0:
        # deliverySQL = "INSERT IGNORE INTO IssueDeliveryTable_CFETS SELECT t3.IssueCode,t2.IssueCode,t2.IssueName,DATE_FORMAT(now(),'%Y%m%d'),'','' from (SELECT t1.IssueCode,t1.IssueName,CONCAT(t1.underlying,t1.term) UnderlyingIssueCode,t1.duration from (SELECT t.IssueCode,t.IssueName,t.CustodianName,t.BondType,t.Duration/365 duration,CASE WHEN t.Duration/365 > 1.5 and t.Duration/365 <= 2.25 THEN 2 WHEN t.Duration/365 > 2.5 and t.Duration/365 <= 3 THEN 3 WHEN t.Duration/365 > 4.25 and t.Duration/365 <= 5 THEN 5 ELSE 10 END term,CASE WHEN LOCATE('国开',t.IssueName) > 0 then 'CDB' WHEN LOCATE('农发',t.IssueName) > 0 then 'ADBC' END underlying FROM BondInfoTable_CFETS t WHERE t.BondType = '政策性金融债' AND t.ExpirationDate >= DATE_FORMAT(now(),'%Y%m%d') and ((((t.Duration/365 > 2.5 and t.Duration/365 <= 3) or (t.Duration/365 > 4.25 and t.Duration/365 <= 5) or (t.Duration/365 > 8 and t.Duration/365 <= 10)) and t.IssueName LIKE '%国开%') or (((t.Duration/365 > 1.5 and t.Duration/365 <= 2.25) or (t.Duration/365 > 4.25 and t.Duration/365 <= 5) or (t.Duration/365 > 8 and t.Duration/365 <= 10)) and t.IssueName LIKE '%农发%')) order BY CAST(Duration AS DECIMAL)) t1) t2,IssueMasterTable_CFETS t3 where concat(t2.UnderlyingIssueCode,'P') = t3.UnderlyingIssueCode"
        # WriteLog(deliverySQL)
        # try:
           # MySqlCR.execute(deliverySQL)
           # MySQlDB.commit()
        # except:
           # MySQlDB.rollback()
           # WriteLog("ERRData-deliverySQL")
        
    #4.更新实物交割券的转换因子
    sql = "select i_code,d_i_code,ctd_cf,beg_date from %stbnd_future_deliverbonds where a_type = 'FWD_BDS'" %(GlobalSettingTable['Oracle_XIR_MD'])
    WriteLog(sql)
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        ccpIssueCode = "IB"+res[0]#CCP代码
        bondIssueCode = "IB"+res[1]#CBT代码
        cf = res[2]#转换因子
        count += 1
        
        cfSQL = "update IssueDeliveryTable_CFETS set CF = %.6f where UpdateDate = date_format(now(),'%%Y%%m%%d') and IssueCode = '%s' and BondCode = '%s'" % (cf,ccpIssueCode,bondIssueCode)
        WriteLog(cfSQL)
        try:
           MySqlCR.execute(cfSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-cfSQL")
        
    if count == 0:
        WriteErrorLog("同步标债远期实物交割转换因子, 数据0条")
    
    #5.检查标债远期交割券发生变化的信息
    sql = "select * from (select * from IssueDeliveryTable_CFETS where UpdateDate = DATE_FORMAT(now(),'%Y%m%d') UNION select * from IssueDeliveryTable_CFETS where UpdateDate = (SELECT max(DTSDate) from CalendarTable_CFETS WHERE BondTrade = '0' and DTSDate < DATE_FORMAT(now(),'%Y%m%d'))) t group by t.IssueCode,t.BondCode having count(*) = 1"
    WriteLog('检查标债远期交割券发生变化的信息:'+sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    
    for row in results:
        issueCode       = row[0]
        bondCode        = row[1]
        updateDate      = row[3]
        
        WriteLog("检查标债远期交割券发生变化的信息,issueCode[%s],bondCode[%s],updateDate[%s]" % (issueCode,bondCode,updateDate))
        
        if updateDate == gCurrentDate:
            WriteErrorLog("标债远期交割券:标债远期合约 %s 新增交割券 %s " % (issueCode,bondCode))
        else:
            WriteErrorLog("标债远期交割券:标债远期合约 %s 减少交割券 %s " % (issueCode,bondCode))
            
    #6.标债远期合约对应的交割券数量是否正确
    sql = "SELECT * FROM (SELECT issueMaster.IssueCode,case when RIGHT(issueMaster.IssueCode,1) = 'P' then 1 ELSE 0 END AS IssueType,ifnull(issueDeliver.BondCount,0) AS BondCount FROM (SELECT IssueCode FROM IssueMasterTable_CFETS  WHERE ExpirationDate >= DATE_FORMAT(now(),'%Y%m%d') AND ListDate <= DATE_FORMAT(now(),'%Y%m%d')) issueMaster LEFT JOIN (select IssueCode,COUNT(BondCode) AS BondCount from IssueDeliveryTable_CFETS where UpdateDate = DATE_FORMAT(now(),'%Y%m%d') GROUP BY IssueCode) issueDeliver ON issueMaster.IssueCode = issueDeliver.IssueCode) t WHERE (t.IssueType = 1 AND (t.BondCount < 2 OR t.BondCount > 4)) OR (t.IssueType = 0 AND t.BondCount != 2)"
    WriteLog('标债远期合约对应的交割券数量是否正确:'+sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    
    for row in results:
        issueCode       = row[0]
        issueType       = row[1]
        bondCount       = row[2]
        
        WriteLog("标债远期合约对应的交割券数量是否正确,issueCode[%s],issueType[%d],bondCount[%d]" % (issueCode,issueType,bondCount))
        WriteErrorLog("标债远期交割券:标债远期合约 %s 的交割券数量 %d 不符合要求" % (issueCode,bondCount))
        
    
    #7.实物交割合约的交割券转换因子是否变化
    sql = "select t_today.IssueCode,t_today.BondCode,t_today.CF as CF_today,t_pre.CF as CF_pre from (select concat(IssueCode,'.',BondCode) AS KeyCode,IssueCode,BondCode,CF,UpdateDate from IssueDeliveryTable_CFETS where IssueCode like '%P' and UpdateDate = DATE_FORMAT(now(),'%Y%m%d')) t_today LEFT JOIN (select concat(IssueCode,'.',BondCode) AS KeyCode,CF,UpdateDate from IssueDeliveryTable_CFETS where IssueCode like '%P' and UpdateDate = (SELECT max(DTSDate) from CalendarTable_CFETS WHERE BondTrade = '0' and DTSDate < DATE_FORMAT(now(),'%Y%m%d'))) t_pre on t_today.KeyCode = t_pre.KeyCode WHERE t_today.CF+0 != t_pre.CF+0 or isnull(t_today.CF) or t_today.CF = ''"
    WriteLog('实物交割合约的交割券转换因子是否变化:'+sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    
    for row in results:
        issueCode       = row[0]
        bondCode        = row[1]
        cf_today        = row[2]
        cf_pre          = row[3]
        
        WriteLog("实物交割合约的交割券转换因子是否变化,issueCode[%s],bondCode[%s],cf_today[%s],cf_pre[%s]" % (issueCode,bondCode,str(cf_today),str(cf_pre)))
        WriteErrorLog("标债远期交割券:标债远期合约 %s 对应的交割券 %s 的转换因子有变化,昨日值[%s] -> 今日值[%s]" % (issueCode,bondCode,str(cf_pre),str(cf_today)))

def SyncBondEval():
    preTradingDate = GetPreTradingDate()
    global GlobalSettingTable
    sql = "select k.beg_date,k.i_code,k.yield,k.netprice,k.fullprice,k.m_type from %sTCB_BOND_EVAL k where k.beg_date ='%s-%s-%s' and M_TYPE in ('X_CNBD','XSHE','XSHG') order by i_code" % (GlobalSettingTable['Oracle_XIR_MD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    OracleCR.execute(sql)
    WriteLog(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode = res[1]
        if res[5] == 'X_CNBD':
            issueCode = "IB"+issueCode
        elif res[5] == 'XSHE':
            issueCode = "SZ"+issueCode
        elif res[5] == 'XSHG':
            issueCode = "SH"+issueCode
        
        yield1 = res[2]
        netprice1 = res[3]
        fullprice1 = res[4]
        count += 1
        
        bondEvalSQL = "REPLACE INTO dtsdb.BondEvalTable_CFETS(IssueCode,DataDate,Yield,NetPrice,FullPrice) VALUES ('%s','%s','%s','%s','%s')" % (issueCode,preTradingDate,yield1,netprice1,fullprice1)
        WriteLog(bondEvalSQL)        
        try:
           MySqlCR.execute(bondEvalSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-bondEvalSQL")
           
    if count == 0:
        WriteErrorLog("同步中债估值0条")

def SyncCCPClearPrice():
    preTradingDate = GetPreTradingDate()
    #从CMDS表中获取结算价到HistoricalPriceTable
    sql = "SELECT k.securityid,k.settledprice FROM %sTTRD_CMDS_SBFWD_SETTLEDPRICE k where substr(k.updatetime,1,10) = '%s-%s-%s'" % (GlobalSettingTable['Oracle_XIR_TRD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode = "IB"+res[0]
        clearPrice = res[1]
        count += 1
        
        ccpClearPriceSQL = "update HistoricalPriceTable set ClearingPrice = %s,TimeStamp ='%s'  where DataDate = '%s' and IssueCode = '%s'" % (clearPrice,datetime.datetime.now(),preTradingDate,issueCode)
        WriteLog(ccpClearPriceSQL)        
        try:
           MySqlCR.execute(ccpClearPriceSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-ccpClearPriceSQL")   
    
    if count == 0:
        WriteErrorLog("同步CCP昨结算价, 从CMDS表中获取结算价0条")
    #从结算价表中获取结算价到HistoricalPriceTable_CFETS
    sql = "SELECT k.i_code,k.dp_set FROM %sTNON_DAILYPRICE k where (k.i_code like 'CDB%%' or k.i_code like 'ADBC%%') and k.beg_date = '%s-%s-%s'" % (GlobalSettingTable['Oracle_XIR_MD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    WriteLog(sql)
    count = 0
    
    try:
        OracleCR.execute(sql)
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
            
            issueCode = "IB"+res[0]
            clearPrice = res[1]
            count += 1
            
            ccpClearPriceSQL = "REPLACE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,ClearingPrice) VALUES ('%s','9','%s',%s)" % (issueCode,preTradingDate,clearPrice)
            WriteLog(ccpClearPriceSQL)        
            try:
               MySqlCR.execute(ccpClearPriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-ccpClearPriceSQL")   
    except:
        WriteLog("Select Error "+sql)   

    if count == 0:
        WriteErrorLog("同步CCP昨结算价, 从结算价表中获取结算价0条")

def SyncAvlTradeBondInfo():
    sql = "SELECT k.security_id,k.symbol,k.security_desc,k.centraquotebondindic,k.securitytypeid,k.issuershortpartyid,k.termtomaturitystring FROM %sttrd_cfets_b_xbondinfo k where substr(k.updatetime,1,10) = to_char(sysdate,'yyyy-MM-dd')" % (GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode = "IB"+res[0]
        issueName = res[1]
        bondType = res[2]
        centerQuote = res[3]
        bondTypeID = res[4]
        issueShortPartyID = res[5]
        durationString = res[6]
        count += 1
        
        avlTradeBondSQL = "REPLACE INTO dtsdb.BondInfo_XBond_CFETS(IssueCode,IssueName,BondType,BondTypeID,CenterQuote,IssuerShortPartyID,DurationString,AutoSubscribe,DataDate) VALUES ('%s','%s','%s','%s','%s','%s','%s','0',date_format(now(),'%%Y%%m%%d'))" % (issueCode,issueName,bondType,bondTypeID,centerQuote,issueShortPartyID,durationString)
        WriteLog(avlTradeBondSQL)        
        try:
           MySqlCR.execute(avlTradeBondSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-avlTradeBondSQL")

    if count == 0:
        WriteErrorLog("同步XBOND可交易债0条")
def SyncAutoSubscribeXBOND():
    sql = "SELECT I_CODE FROM (SELECT I_CODE FROM %sTTRD_CFETS_B_BOND WHERE (BOND_TYPE = '国债' OR BOND_TYPE = '政策性金融债') AND COUPON_TYPE = '固定利率' AND LENGTH(I_CODE) = 6 AND (TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) / 365 >=0.5 AND (TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) / 365 <=10 AND I_CODE IN (SELECT K.SECURITY_ID FROM %sTTRD_CFETS_B_XBONDINFO K WHERE SUBSTR(K.UPDATETIME, 1, 10) = TO_CHAR(SYSDATE, 'YYYY-MM-DD')) ORDER BY SUBSTR(ISSUE_DATE, 1, 4) DESC,TO_NUMBER(TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) DESC) WHERE ROWNUM <= 200" % (GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
    OracleCR.execute(sql)
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break

        issueCode = "IB"+res[0]
        count += 1
        
        autoSubscribeSQL = "Update BondInfo_XBond_CFETS set AutoSubscribe = '1' where DataDate = date_format(now(),'%%Y%%m%%d') and IssueCode = '%s'" % (issueCode)
        WriteLog(autoSubscribeSQL)
        try:
           MySqlCR.execute(autoSubscribeSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-autoSubscribeSQL")
        
    if count == 0:
        WriteErrorLog("获取自动订阅的XBOND债券0条")


#-----------获取N月后或N月前的日期
def get_nMonth_date(date,n=0):
    '''''
    获取N月后或N月前的日期
    '''
    thisYear = date.year#获取当前日期的年份
    thisMon =date.month#获取当前日期的月份
    thisDay=date.day#获取当前日期的天数
    totalMon = thisMon + n#加上n月后的总月份数
    
    lastYear=0
    lastMon=0
    if (n >= 0):#如果n大于等于0
        if (totalMon <= 12):#如果总月份数少于12
            lastYear=thisYear
            lastMon=totalMon
        else:
            i = totalMon // 12#年份递增数
            j = totalMon % 12#月份递增数
            if (j == 0):#月份递增数等于0
                i -= 1#年份减一
                j = 12#月份为12
            thisYear += i#年份递增
        
            lastYear=thisYear
            lastMon=j
            
    else:#如果n少于0
        if ((totalMon > 0) and (totalMon < 12)):#如果总月份数大于0少于12
            lastYear=thisYear
            lastMon=totalMon
        else:#如果总月份数少于0
            i = totalMon // 12#年份递减数
            j = totalMon % 12#月份递减数
     
            if (j == 0):#月份递减数等于0
                i -= 1#年份减一
                j = 12#月份为12
            thisYear += i
            
            lastYear=thisYear
            lastMon=j
            
    last_date=datetime.date(lastYear,lastMon,thisDay)
    return last_date

#-----------根据起息日和到期日获取利息“日期流”
def get_dateList_by_StartAndEnd(start_date,end_date,freq):
    '''
    start_date:datetiem类型，起息日
    end_date:datetime类型，到期日
    freq:整数，付息频率
    '''
    dateList=[]#日期流列表
    current_date=start_date#当前日期初始化
    n=12//freq#递增月份    
    while current_date<end_date:
        current_date=get_nMonth_date(current_date,n)#获取下一次付息日
        dateList.append(current_date)#添加到日期流表
    return dateList

#-----------根据当前交易日获取最近付息日和上一次付息日
def get_preAndNextDate_by_trade(trade_date,start_date,end_date,freq):
    dateList=get_dateList_by_StartAndEnd(start_date,end_date,freq)
    pre_date=start_date
    next_date=start_date

    for current_date in dateList:
        if current_date>trade_date:
            next_date=current_date
            break
        else:
            pre_date=current_date
    return (pre_date,next_date)

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

#%%-----牛顿迭代法自实现
def newton_dyt(func, x0, fprime=None,fprime2=None,args=(), tol=1.48e-13, maxiter=100,
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
    
    #检验tol的合法性
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
        
    #检验迭代次数的合法性
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
        
    #检验一阶导数是否可用
    if fprime is not None:

        p0 = 1.0 * x0#转换位浮点数，不用float，因为乘以1.0复数仍然可用
        fder2 = 0#二阶段初始化
        for iter in range(maxiter):#开始迭代
            myargs = (p0,) + args#整合所有参数
            fder = fprime(*myargs)#运行一阶导函数
            if fder == 0:#如果导函数为零
                msg = "一阶导为零"
                warnings.warn(msg, RuntimeWarning)#报运行错误
                return p0#返回初始猜想值
            
            fval = func(*myargs)#计算方程的值
            if fprime2 is not None:#如果二阶导函数可用
                fder2 = fprime2(*myargs)#运行二阶导
            if fder2 == 0:#如果二阶导为零

                p = p0 - fval / fder#使用一阶导函数执行牛顿迭代法步骤
            else:#如果二阶导函数不为零
                # 使用Parabolic Halley方法
                discr = fder ** 2 - 2 * fval * fder2
                if discr < 0:
                    p = p0 - fder / fder2
                else:
                    p = p0 - 2*fval / (fder + np.sign(fder) * np.sqrt(discr))
            if abs(p - p0) < tol:#判断是否逼近实际根值
                return p#逼近则返回
            p0 = p#未逼近则更新根值
    else:#如果一阶导函数和二阶导函数都不可用，则使用割线法
        p0 = x0#初始根值
        if x0 >= 0:
            p1 = x0*(1 + 1e-4) + 1e-4
        else:
            p1 = x0*(1 + 1e-4) - 1e-4
        q0 = func(*((p0,) + args))
        q1 = func(*((p1,) + args))
        for iter in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "偏差达到" % (p1 - p0)
                    warnings.warn(msg, RuntimeWarning)#报运行错误
                return (p1 + p0)/2.0
            else:
                p = p1 - q1*(p1 - p0)/(q1 - q0)#求根
            if abs(p - p1) < tol:#是否逼近实际根值
                return p#若逼近则返回
            p0 = p1#更新次根值
            q0 = q1#更新次方程值
            p1 = p#更新根值
            q1 = func(*((p1,) + args))#更新方程值
    msg = "%d 次迭代后未能收敛,考虑适当加大迭代次数， 最后值是 %s" % (maxiter, p)
    raise RuntimeError(msg)

#%%-----到期收益率
#-----------处于最后付息周期的附息债券(一年以内)
def bond_ytm1(PV,M,T,payType=1,C=None,N=None,f=None,guess=0.05):
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
    if payType==1:
        #贴现
        FV=100
    elif payType==2:
        #一次还本付息
        FV=M+N*C/100.*M
    elif payType==3:
        #定期支付
        FV=M+C/100.*M/f
    else:
        pass
    
    ytm=(FV-PV)/PV/T#计算到期收益率
        
    return ytm

#-----------处于最后付息周期的固定利率附息债券（一年以上）
def bond_ytm2(PV,M,T,payType=1,C=None,N=None,guess=0.05):
    '''
    PV:市场价格(债券全价)
    M:面值
    T:剩余付息年数（待偿期）
    C:票面利息(一次还本付息)
    N：一次还本付息的偿还期限（一次还本付息）

    payType:利息支付方式，包括1-贴现，2-一次还本付息(需要填写票面利率和偿还期限)，

    '''
    if payType==1:
        #贴现
        FV=100
    elif payType==2:
        #一次还本付息
        FV=M+N*C/100.*M
    else:
        pass
    
    ytm=(FV/PV)**(1/T)-1
    return ytm

def bond_ytm3(PV,M,n,w,C,f=2,guess=0.05):
    '''
    PV:市场价格(债券全价)
    M:面值
    n:剩余付息次数
    w:不处于付息周期的年化天数
    C:票面利息
    f:每年的利息支付频率
    '''
    f=float(f)#转换为浮点数对象
    cp=C/100.*M/f#每期付息数
    dt=[i for i in range(int(n))]#遍历付息周期数
    ytm_func=lambda y:sum([cp/(1+y/f)**(w+t) for t in dt])+M/(1+y/f)**(w+n-1)-PV
    return newton_dyt(ytm_func,guess)

def zero_coupon_bond(M,y,t):
    '''
    M:面值
    y：贴现率
    t：期限
    '''
    return M/(1+y)**t


#-----------息票债券
def bond_price(M,n,w,y,C,f=2):
    '''
    M:面值
    n:剩余付息次数
    w:不处于付息周期的年化天数
    y:收益率
    C:票面利息
    f:每年的利息支付频率
    '''
    f=float(f)#转换为浮点数对象
    cp=C/100.*M/f#每期付息数
    dt=[i for i in range(int(n))]#遍历付息周期数
    price=sum([cp/(1+y/f)**(w+t) for t in dt])+M/(1+y/f)**(w+n-1)
    return price

#----------麦考利久期
#市场价格已知
def bond_duration_PV(PV,M,n,w,C,f):
    cp=C/100.*M/f#每期付息数
    y=bond_ytm3(PV,M,n,w,C,f)
    dt=[i for i in range(int(n))]#遍历付息周期数
    time_pv=sum([cp*(w+t)/(1+y/f)**(w+t) for t in dt])+M*(w+n-1)/(1+y/f)**(w+n-1)
    duration=time_pv/PV
    return duration

#市场利率已知
def bond_duration_y(M,n,w,y,C,f):
    cp=C/100.*M/f#每期付息数
    PV=bond_price(M,n,w,y,C,f)
    dt=[i for i in range(int(n))]#遍历付息周期数
    time_pv=sum([cp*(w+t)/(1+y/f)**(w+t) for t in dt])+M*(w+n-1)/(1+y/f)**(w+n-1)
    duration=time_pv/PV
    return duration

#----------修正久期
def bond_mod_duration(PV,M,n,w,C,f,dy=0.01):
    ytm=bond_ytm3(PV,M,n,w,C,f)
    ytm_minus=ytm-dy
    price_minus=bond_price(M,n,w,ytm_minus,C,f)
    ytm_plus=ytm+dy
    price_plus=bond_price(M,n,w,ytm_plus,C,f)
    mduration=(price_minus-price_plus)/(2*PV*dy)
    return mduration


def CCP_pricing(trade_date,cal_date,strategy_table,base_table,deliverable_base_table,ytm1,ytm2):
    #日期相关计算
    ccp_deliver_date=base_table['deliver_date']#CCP的交割日前一日
#    ccp_next_date=get_nMonth_date(ccp_deliver_date,12)#CCP的最近付息日
    hold_days=(ccp_deliver_date-trade_date).days#CCP的持有天数

    
    dl1_start_date=deliverable_base_table['std']['start_interest_date']#可交割券1起息日
    dl1_freq=deliverable_base_table['std']['freq']#可交割券1付息频率
    dl1_end_date=deliverable_base_table['std']['end_interest_date']#可交割券1到期日
    dl1_pre_next_date=get_preAndNextDate_by_trade(cal_date,dl1_start_date,dl1_end_date,dl1_freq)#获取最近付息日和上一次付息日
    dl1_pre_date=dl1_pre_next_date[0]#上一次付息日
    dl1_next_date=dl1_pre_next_date[1]#最近付息日
    dlD1_pre_next_date=get_preAndNextDate_by_trade(ccp_deliver_date,dl1_start_date,dl1_end_date,dl1_freq)#获取最近付息日和上一次付息日
    dlD1_pre_date=dlD1_pre_next_date[0]#交割上一次付息日
    dlD1_next_date=dlD1_pre_next_date[1]#交割最近付息日
    
    
    
    dl2_start_date=deliverable_base_table['nonstd']['start_interest_date']#可交割券2起息日
    dl2_freq=deliverable_base_table['nonstd']['freq']#可交割券2付息频率
    dl2_end_date=deliverable_base_table['nonstd']['end_interest_date']#可交割券2到期日
    dl2_pre_next_date=get_preAndNextDate_by_trade(cal_date,dl2_start_date,dl2_end_date,dl2_freq)#获取最近付息日和上一次付息日
    dl2_pre_date=dl2_pre_next_date[0]#上一次付息日
    dl2_next_date=dl2_pre_next_date[1]#最近付息日
    dlD2_pre_next_date=get_preAndNextDate_by_trade(ccp_deliver_date,dl2_start_date,dl2_end_date,dl2_freq)#获取最近付息日和上一次付息日
    dlD2_pre_date=dlD2_pre_next_date[0]#交割上一次付息日
    dlD2_next_date=dlD2_pre_next_date[1]#交割最近付息日
    
    #可交割券的到期收益率计算
    import math
    n1=math.ceil((dl1_end_date-cal_date).days/365*dl1_freq)
    n2=math.ceil((dl2_end_date-cal_date).days/365*dl2_freq)
    w1=(dl1_next_date-cal_date).days/(dl1_next_date-dl1_pre_date).days
    w2=(dl2_next_date-cal_date).days/(dl2_next_date-dl2_pre_date).days
        
    
    n1_dl=math.ceil((dl1_end_date-ccp_deliver_date).days/365*dl1_freq)
    n2_dl=math.ceil((dl2_end_date-ccp_deliver_date).days/365*dl2_freq)
    w1_dl=(dlD1_next_date-ccp_deliver_date).days/(dlD1_next_date-dlD1_pre_date).days
    w2_dl=(dlD2_next_date-ccp_deliver_date).days/(dlD2_next_date-dlD2_pre_date).days
    
    
    #套利组合参数计算
    bond1_full_price=bond_price(100,n1,w1,ytm1,deliverable_base_table['std']['par_rate'],deliverable_base_table['std']['freq'])
    bond2_full_price=bond_price(100,n2,w2,ytm2,deliverable_base_table['nonstd']['par_rate'],deliverable_base_table['nonstd']['freq'])     
    
    comb_bond_price=(bond1_full_price+bond2_full_price)/2
    com_ccp_ytm=(ytm1+ytm2)/2
    
    #实际资金成本
    Capital_cost_bond=comb_bond_price*strategy_table['capital_cost']*hold_days/365
    
    #到期交割价

      
    ccp_deliver_price=bond_price(100,base_table['term'],1.,com_ccp_ytm,base_table['par_rate'],base_table['freq'])
    bond1_deliver_price=bond_price(100,n1_dl,w1_dl,ytm1,deliverable_base_table['std']['par_rate'],deliverable_base_table['std']['freq'])
    bond2_deliver_price=bond_price(100,n2_dl,w2_dl,ytm2,deliverable_base_table['nonstd']['par_rate'],deliverable_base_table['nonstd']['freq'])
    
    #计算部分（现券持有期收益）
    com_bond_deliver_price=(bond1_deliver_price+bond2_deliver_price)/2
    r1=0 if ccp_deliver_date<dl1_next_date else deliverable_base_table['std']['par_rate']/deliverable_base_table['std']['freq']
    r2=0 if ccp_deliver_date<dl2_next_date else deliverable_base_table['nonstd']['par_rate']/deliverable_base_table['nonstd']['freq']
    r_hold=com_bond_deliver_price-comb_bond_price+(r1+r2)/2
    
    if strategy_table['type']=='IRR':
        
        IRR=strategy_table['parameter']
        #计算部分（标债的做市价格）
        ccp_current_price=np.round(ccp_deliver_price-r_hold+(IRR*hold_days*comb_bond_price)/365,4)
        
        #计算部分（净基差）
        net_basis=np.round(ccp_current_price-ccp_deliver_price+r_hold-Capital_cost_bond,4)
        
    elif strategy_table['type']=='net_basis':
        net_basis=strategy_table['parameter']
        
        #计算部分（标债的做市价格）
        ccp_current_price=ccp_deliver_price+net_basis-r_hold+Capital_cost_bond
        #计算部分（IRR）
        IRR=np.round((ccp_current_price-ccp_deliver_price+r_hold)/comb_bond_price*365/hold_days,6)
    elif strategy_table['type']=='price':
        ccp_current_price=strategy_table['parameter']
        IRR=np.round((ccp_current_price-ccp_deliver_price+r_hold)/comb_bond_price*365/hold_days,6)
        net_basis=np.round(ccp_current_price-ccp_deliver_price+r_hold-Capital_cost_bond,4)
    else:
        pass
    
    #最后策略计算结果
    result_dict={'ccp_price':ccp_current_price,'ccp_dl_price':ccp_deliver_price,\
                 'r_hold':r_hold,'avg_return':com_ccp_ytm,\
                 'IRR':IRR,'net_basis':net_basis}
    
    return result_dict

def round_dec(num,d=4):
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
        date_range = [dateList[n], dateList[n+1]]
        if date_range[0] <= settle_date < date_range[1]:
            break
        elif n == len(dateList) - 1:
            break
        else:
            n += 1     
    #date_range[1] = CA.FOL(date_range[1])
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


#%% BOND定价引擎(只定价不含权的固息/零息/贴现债)
class BondPricingEngine: #to do 统一百分和小数格式
    def __init__(self,settle_date, bond_detail):
        self.M = bond_detail['面值']
        self.coupon_type = bond_detail['息票类型']
        self.coupon = bond_detail['固定利率']
        self.issue_price = bond_detail['发行价格']
        self.settle_date = to_dt(settle_date) # 结算日（定价时间点)
        self.start_date = to_dt(bond_detail['起息日'])
        self.end_date = to_dt(bond_detail['到期日'])
        self.k = 0#债券起息日至计算日的整年数   
        self.f_type = bond_detail['息票付息频率'] # 年付息频率
        f_dict = {'半年':2, '年':1, '季':4, '到期':1, '月':12}
        if self.f_type in f_dict.keys():
            self.f = f_dict[bond_detail['息票付息频率']]        
        self.YTM = float(bond_detail['YTM'])/100
        self.clean_price = bond_detail['净价']
    
    def Cal_AccuredInterest(self, n_reserve = 7):
        self.d = (self.end_date - self.settle_date).days
        self.t = (self.settle_date - self.start_date).days
        self.ty = calc_ty(self.settle_date, self.start_date)
        self.fv = self.M
        
        if (self.coupon_type == '贴现')&(self.f_type == '到期'):
            self.coupon = 100 - self.issue_price
            self.ts = (self.end_date - self.start_date).days            
            self.ai = self.coupon / self.f * self.t / self.ts
                         
        elif (self.coupon_type == '零息利率')&(self.f_type == '到期'):
            self.ai = self.k * self.coupon + self.coupon / self.ty * self.t
            # 算YTM、clean_price需要的其他变量
            self.fv = self.M + ((self.end_date-self.start_date).days/self.ty)*(self.coupon/100.*self.M)
            
        elif self.coupon_type == '固定利率':
            self.ts, self.coupon_period, self.n = calc_ts(self.settle_date, self.start_date, self.end_date, self.f)
            self.t = (self.settle_date - self.coupon_period[0]).days
            self.ai = self.coupon / self.f * self.t / self.ts
            
            # 算YTM、clean_price需要的其他变量
            self.d = (self.coupon_period[1] - self.settle_date).days
            
            if self.coupon_period[1] == self.end_date:
                self.fv = self.M + self.coupon/self.f
            else:
                self.coupon_per = self.coupon / 100. * self.M / self.f  # 每期付息
                self.w = self.d / self.ts
        
        self.ai = round_dec(self.ai, d=n_reserve)
                
    def Cal_YTM(self, n_reserve = 4):
        self.Cal_AccuredInterest()
        self.dirty_price = self.clean_price + self.ai
        
        if (self.coupon_type == '贴现')&(self.f_type == '到期'):        
            self.YTM = (self.fv - self.dirty_price) / self.dirty_price / (self.d / self.ty)
            
        elif (self.coupon_type == '零息利率')&(self.f_type == '到期'):            
            self.YTM = (self.fv - self.dirty_price) / self.dirty_price / (self.d / self.ty)
            
        elif self.coupon_type == '固定利率':
            
            if self.coupon_period[1] == self.end_date:
                self.YTM = (self.fv- self.dirty_price) / self.dirty_price / (self.d / self.ty)
                
            else:
                ytm_func = lambda y: sum([self.coupon_per / ((1 + y / self.f) ** (self.w + i))   for i in range(self.n)]) + self.M / (1 + y / self.f) ** (self.w + self.n - 1) - self.dirty_price
                self.YTM = newton(ytm_func, 0.05)
                
        self.YTM = round_dec(self.YTM*100., d=n_reserve)
                
    def Cal_CleanPrice(self, n_reserve = 4):
        self.Cal_AccuredInterest()
        
        if (self.coupon_type == '贴现')&(self.f_type == '到期'):
            self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)
            
        elif (self.coupon_type == '零息利率')&(self.f_type == '到期'):
            self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)
            
        elif self.coupon_type == '固定利率':
            
            if self.coupon_period[1] == self.end_date:
                self.dirty_price = self.fv / (self.YTM * (self.d / self.ty) + 1)
                
            else:
                self.dirty_price = sum([self.coupon_per / (1 + self.YTM / self.f) ** (self.w + i) for i in range(self.n)]) + self.M / ((1 + self.YTM / self.f) ** (self.w + self.n - 1))
        
        self.clean_price = round_dec(self.dirty_price - self.ai, d=n_reserve)
        self.dirty_price = round_dec(self.dirty_price, d=n_reserve)

#%% SBF定价引擎(实物交割)
class SBFPricingEngine():
    def __init__(self, forward_detail, bond_detail):
        self.r = 0.03 #远期合约票面利率 
        self.rc = forward_detail['融资成本']/100
        self.settle_date = to_dt(forward_detail['交割券结算日'])
        self.delivery_date = to_dt(forward_detail['第二交割日'])
        self.N = (self.delivery_date - self.settle_date).days #合约存续天数
        self.F = forward_detail['合约价格']
        self.IRR = forward_detail['IRR']/100
        self.BNOC = forward_detail['净基差']/100
        
        
        # 在结算日定价交割券并计算对应参数（应计利息将做节假日调整）
        Bond_settle = BondPricingEngine(forward_detail['交割券结算日'], bond_detail)#默认bond_detail包含债券净价
        Bond_settle.Cal_AccuredInterest(n_reserve=7)
        self.f = Bond_settle.f #交割券年付息频率
        self.c = Bond_settle.coupon/100 #交割券票面利率
        self.settle_clean = Bond_settle.clean_price #结算日交割券净价
        self.Ps = round_dec(Bond_settle.clean_price + Bond_settle.ai, d=7) #结算日交割券结算价
                
        # 在交割日计算交割券应计利息（应计利息将做节假日调整）
        Bond_delivery = BondPricingEngine(forward_detail['第二交割日'], bond_detail)
        Bond_delivery.Cal_AccuredInterest(n_reserve=7)
        self.delivery_ai = Bond_delivery.ai
        # 节假日调整         
        self.m = (self.delivery_date - GetNextSettleDate(Bond_delivery.coupon_period[0])).days
        # 交割月到下一付息月的月份数
        self.x = (Bond_delivery.coupon_period[1].year - self.delivery_date.year)*12 + (Bond_delivery.coupon_period[1].month - self.delivery_date.month)
        self.I = 0 if self.delivery_date < Bond_settle.coupon_period[1] else self.c*100./self.f #合约存续期间交割券利息支付
        self.n = Bond_delivery.n  #在交割日交割券剩余付息次数
        # 交割券转换因子(严格四舍五入至4位)
        self.cf = round_dec((1/(1+self.r/self.f)**(self.x*self.f/12))*(self.c/self.f+self.c/self.r+(1-self.c/self.r)*1/(1+self.r/self.f)**(self.n-1))-self.c/self.f*(1-self.x*self.f/12))
        self.Pd = round_dec(self.F*self.cf + self.delivery_ai, d=7)
        
        WriteLog("N[%s],f[%s],c[%s],settle_clean[%s],Ps[%s],delivery_ai[%s],m[%s],x[%s],I[%s],n[%s],cf[%s],Pd[%s]" % (self.N,self.f,self.c,self.settle_clean,self.Ps,self.delivery_ai,self.m,self.x,self.I,self.n,self.cf,self.Pd))
        
    # 计算持有期收益
    def Cal_Y(self):
        self.Y = (self.I - self.Ps*self.rc)*self.N/365

    # 已知F算IRR,BNOC
    def Cal_from_F(self):
        self.Cal_Y()
        self.IRR = round_dec(((self.Pd + self.I - self.Ps)/(self.Ps*self.N - self.I*self.m))*365*100.)
        self.B = round_dec(self.settle_clean - self.F*self.cf)
        self.BNOC = round_dec(self.settle_clean - self.Y - self.F*self.cf)
        
    def Cal_from_IRR(self):
        self.Cal_Y()
        self.F = (self.IRR*(self.Ps*self.N - self.I*self.m)/365 + self.Ps - self.I -self.delivery_ai)/self.cf
        self.BNOC = self.settle_clean - self.Y - self.F*self.cf
        
        WriteLog("Y[%s],F[%s],BNOC[%s]" % (self.Y,self.F,self.BNOC))
        
    def Cal_from_BNOC(self):
        self.Cal_Y()
        self.F = (self.settle_clean - self.Y - self.BNOC)/self.cf
        self.IRR = (self.F*self.cf + self.delivery_ai + self.I - self.Ps)/(self.Ps*self.N - self.I*self.m)*365
    


def CalcCCPPreBasePrice():
    global gCurrentDate
    preTradingDate = GetPreTradingDate()
    nextTradingDay = gCurrentDate
    
    sql = "select min(DTSDate) from CalendarTable_CFETS where DTSDate > DATE_FORMAT(now(),'%Y%m%d') and BondTrade = 0"
    WriteLog(sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    
    for row in results:
        nextTradingDay = row[0]
    
    
    sql = "SELECT t2.Yield,t2.CCPCode,t2.SettleDate SettleDate_ccp,bondInfo.IssueCode,bondInfo.FirstValueDate,bondInfo.ExpirationDate ExpirationDate_cbt,bondInfo.CouponFrequency,bondInfo.FixedCouponRate from BondInfoTable_CFETS bondInfo INNER JOIN (SELECT eval.Yield,eval.IssueCode BondCode,t.IssueCode CCPCode,t.SettleDate from BondEvalTable_CFETS eval INNER JOIN (SELECT delivery.IssueCode,delivery.BondCode,ccp.SettleDate from IssueDeliveryTable_CFETS delivery INNER JOIN (SELECT IssueCode,SettleDate from IssueMasterTable_CFETS WHERE ListDate <= DATE_FORMAT(now(),'%%Y%%m%%d') and ExpirationDate >= DATE_FORMAT(now(),'%%Y%%m%%d') and right(IssueCode,1) != 'P') ccp ON (ccp.IssueCode = delivery.IssueCode) WHERE delivery.UpdateDate = DATE_FORMAT(now(),'%%Y%%m%%d')) t ON (t.BondCode = eval.IssueCode) WHERE eval.DataDate = '%s') t2 ON (t2.BondCode = bondInfo.IssueCode)" % (preTradingDate)
    WriteLog(sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    if results == ():
        WriteErrorLog("计算昨基准0条")
    
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
        
    
    WriteLog('CalcCCPPreBasePrice,lt:'+str(lt))
    yyyy = int(nextTradingDay[0:4])
    mm = int(nextTradingDay[4:6])
    dd = int(nextTradingDay[6:8])
    trade_date = datetime.date(yyyy,mm,dd)#当前交易日
    cal_date = datetime.date(yyyy,mm,dd)#债券结算日
    
    #从数据库获取资金成本
    capitalCost = 0.027;
    sql = "SELECT Field1 FROM dtsdb.StrategyEventTable WHERE EventID = 'CCPBase'";
    WriteLog(sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    for row in results:
        if is_number(row[0]):
            capitalCost = float(row[0])/100.0
        
    
    WriteLog("capitalCost:"+str(capitalCost))
    
    strategy_table = {'type':'IRR','parameter':capitalCost,'capital_cost':0.027}#策略参数
    for ccpIssueCode in lt:
        if lt[ccpIssueCode].has_key('1') and lt[ccpIssueCode].has_key('2'):
            deliver_date_1 = lt[ccpIssueCode]['deliver_date']
            base_table = {'term':lt[ccpIssueCode]['term'],'par_rate':3.0,'freq':1,'deliver_date':datetime.date(int(deliver_date_1[0:4]),int(deliver_date_1[4:6]),int(deliver_date_1[6:8]))}#ccp合约基础信息
            
            start_interest_date_1 = lt[ccpIssueCode]['1']['start_interest_date']
            end_interest_date_1 = lt[ccpIssueCode]['1']['end_interest_date']
            start_interest_date_2 = lt[ccpIssueCode]['2']['start_interest_date']
            end_interest_date_2 = lt[ccpIssueCode]['2']['end_interest_date']
            #外汇交易中心官方公布的可交割券基础信息
            deliverable_base_table={
                        'std':{
                            'par_rate':lt[ccpIssueCode]['1']['par_rate'],
                            'freq':lt[ccpIssueCode]['1']['freq'],
                            'start_interest_date':datetime.date(int(start_interest_date_1[0:4]),int(start_interest_date_1[4:6]),int(start_interest_date_1[6:8])),
                            'end_interest_date':datetime.date(int(end_interest_date_1[0:4]),int(end_interest_date_1[4:6]),int(end_interest_date_1[6:8]))
                        },
                        'nonstd':{
                            'par_rate':lt[ccpIssueCode]['2']['par_rate'],
                            'freq':lt[ccpIssueCode]['2']['freq'],
                            'start_interest_date':datetime.date(int(start_interest_date_2[0:4]),int(start_interest_date_2[4:6]),int(start_interest_date_2[6:8])),
                            'end_interest_date':datetime.date(int(end_interest_date_2[0:4]),int(end_interest_date_2[4:6]),int(end_interest_date_2[6:8]))
                        }
                }#可交割券基础数据
            ret = CCP_pricing(trade_date,cal_date,strategy_table,base_table,deliverable_base_table,lt[ccpIssueCode]['1']['ytm'],lt[ccpIssueCode]['2']['ytm'])
            WriteLog('CalcCCPPreBasePrice,%s,trade_date:%s,strategy_table:%s,base_table:%s,deliverable_base_table:%s,ytm1:%s,ytm2:%s,ret:%s'%(ccpIssueCode,str(trade_date),str(strategy_table),str(base_table),str(deliverable_base_table),str(lt[ccpIssueCode]['1']['ytm']),str(lt[ccpIssueCode]['2']['ytm']),str(ret)))
            
            #防止没有数据不能update,先尝试插入
            ccpClearPriceSQL = "INSERT IGNORE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,BasicPrice) VALUES ('%s','9','%s',%s)" % (ccpIssueCode,preTradingDate,np.round(ret['ccp_price'],4))
            WriteLog(ccpClearPriceSQL)
            try:
               MySqlCR.execute(ccpClearPriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-ccpClearPriceSQL")
               
            #再执行更新
            ccpClearPriceSQL = "update HistoricalPriceTable_CFETS set BasicPrice = %.4f where DataDate = '%s' and IssueCode = '%s'" % (np.round(ret['ccp_price'],4),preTradingDate,ccpIssueCode)
            WriteLog(ccpClearPriceSQL)
            try:
               MySqlCR.execute(ccpClearPriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-ccpClearPriceSQL")
        
    

def SyncIRSCurve():
    preTradingDate = GetPreTradingDate()
    sql = "SELECT case a.SECURITY_ID WHEN 'CISCD01M' THEN 'FDR001' WHEN 'CISCSONM' THEN 'ShiborON' WHEN 'CISCY1QM' THEN 'LPR1Y' WHEN 'CISCY5QM' THEN 'LPR5Y' WHEN 'CISCS3MM' THEN 'Shibor3M' WHEN 'CISCD07M' THEN 'FDR007' WHEN 'CISCF07M' THEN 'FR007' else a.SECURITY_ID end,b.YIELD_RATE,a.SYMBOL,case to_number(b.YIELD_TERM) WHEN 30 THEN '1M' WHEN 90 THEN '3M' WHEN 180 THEN '6M' WHEN 270 THEN '9M' WHEN 360 THEN '1Y' WHEN 720 THEN '2Y' WHEN 1080 THEN '3Y' WHEN 1440 THEN '4Y' WHEN 1800 THEN '5Y' WHEN 2520 THEN '7Y' WHEN 3600 THEN '10Y' ELSE b.YIELD_TERM END,a.BEGIN_DATE,a.UPDATE_TIME FROM %sTTRD_CMDS_IRS_STD_TM_EXP_CURVE a JOIN %sTTRD_CMDS_IRS_STD_TM_EXP_CV_ET b ON a.id = b.id WHERE a.BEGIN_DATE = '%s-%s-%s' AND a.SYMBOL LIKE '%%收盘%%均值%%'" % (GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    WriteLog(sql)
    OracleCR.execute(sql)
    
    count = 0
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode       = res[0]
        curve           = res[1]
        comment         = res[2]
        term            = res[3]
        beginDate       = res[4]
        updateTime      = res[5]
        
        count += 1
        
        WriteLog("SyncIRSCurve_1,issueCode:%s,curve:%s,comment:%s,term:%s,beginDate:%s,updateTime:%s" % (issueCode,curve,comment,term,beginDate,updateTime))
        irsCurveSQL = "REPLACE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,ClearingPrice,BasicPrice) VALUES ('%s','9','%s',0,%s)" % ('IB'+issueCode+'_'+term,preTradingDate,curve)
        WriteLog(irsCurveSQL)
        try:
           MySqlCR.execute(irsCurveSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-irsCurveSQL")
        
    #没有查到CMDS数据,查询衡泰资讯
    if count == 0:
        WriteLog('没有查到IRS利率曲线')
        
        sql2 = "select I_CODE,DP_CLOSE,BEG_DATE from %sTIRSWAP_SERIES  where BEG_DATE = '%s-%s-%s' and DP_BANK = 'CFETS' and Q_TYPE = '1' and end_date = '2050-12-31'" % (GlobalSettingTable['Oracle_XIR_MD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
        WriteLog(sql2)
        OracleCR.execute(sql2)
        count = 0
    
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
            
            issueCode       = res[0]
            curve           = res[1]
            beginDate       = res[2]
            count += 1
            
            if issueCode[:6] == 'LPR_1Y':
                issueCode = 'LPR1Y' + issueCode[6:]
            elif issueCode[:6] == 'LPR_5Y':
                issueCode = 'LPR5Y' + issueCode[6:]
            elif issueCode[:9] == 'SHIBOR-1D':
                issueCode = 'ShiborO/N' + issueCode[9:]
            elif issueCode[:9] == 'SHIBOR-3M':
                issueCode = 'Shibor3M' + issueCode[9:]
            else:
                pass;
            
            WriteLog("SyncIRSCurve_2,issueCode:%s,curve:%s,beginDate:%s" % (issueCode,curve,beginDate))
            irsCurveSQL = "REPLACE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,ClearingPrice,BasicPrice) VALUES ('%s','9','%s',0,%s)" % ('IB'+issueCode,preTradingDate,curve)
            WriteLog(irsCurveSQL)
            try:
               MySqlCR.execute(irsCurveSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-irsCurveSQL")
        if count == 0:
            WriteErrorLog("同步IRS昨日收盘曲线, 没有查到CMDS数据,查询衡泰资讯0条")
       
   
def SyncBondConversion():
    sql = "SELECT TB.I_CODE,TB.A_TYPE,TB.M_TYPE,TB.B_NAME,BC.BEG_DATE,BC.CONV_CODE,BC.CONV_PRICE,TB.P_CLASS,TB.B_EXTEND_TYPE FROM %sTBND TB JOIN %sGF_BONDCONVERSION BC ON TB.I_CODE = BC.I_CODE AND TB.M_TYPE = BC.M_TYPE AND BC.END_DATE = '2050-12-31' WHERE TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND TB.P_CLASS IN ('可交换债','可转换债')" % (GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_MD'])
    WriteLog(sql)
    OracleCR.execute(sql)
    count = 0
    
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode       = res[0]#债券代码
        assetType       = res[1]#资产类型
        marketCode      = res[2]#市场代码
        issueName       = res[3]#债券名称
        beginDate       = res[4]#转股日期
        convCode        = res[5]#转股代码
        convPrice       = res[6]#转股价格
        bondType        = res[7]#债券类型
        bondExtendType  = res[8]#债券扩展类型
        count += 1
        
        if marketCode == 'XSHG':
            marketCode = '1';
        elif marketCode == 'XSHE':
            marketCode = '2';
        
        beginDate = beginDate.replace('-','')
        
        WriteLog("SyncBondConversion,issueCode:%s,assetType:%s,marketCode:%s,issueName:%s,beginDate:%s,convCode:%s,convPrice:%s,bondType:%s,bondExtendType:%s" % (issueCode,assetType,marketCode,issueName,beginDate,convCode,convPrice,bondType,bondExtendType))
        bondConversionSQL = "REPLACE INTO BondCoversionTable_CFETS(IssueCode,MarketCode,IssueName,ConvDate,ConvCode,ConvPrice,BondType,BondExtredType) VALUES ('%s','%s','%s','%s','%s',%s,'%s','%s')" % (issueCode,marketCode,issueName,beginDate,convCode,convPrice,bondType,bondExtendType)
        WriteLog(bondConversionSQL)
        try:
           MySqlCR.execute(bondConversionSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-bondConversionSQL")
    if count == 0:
        WriteErrorLog("同步私募可交换债与正股关系0条")
    


def SyncBondPosition():

    #债券持仓
    #sql = "SELECT SE.I_CODE,TB.B_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT),TB.B_EXTEND_TYPE FROM %sTBND TB JOIN %sTTRD_ACC_BALANCE_SECU SE ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0) GROUP BY SE.I_CODE,TB.B_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID,B_EXTEND_TYPE" % (GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
    sql = "SELECT T.I_CODE,T.B_NAME,T.A_TYPE,T.M_TYPE,A.ACCID,A.PS_L_AMOUNT,T.B_EXTEND_TYPE,T.PENETRATEISSUER FROM %sTBND T LEFT JOIN (SELECT TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT) PS_L_AMOUNT FROM %sTBND TB LEFT JOIN %sTTRD_ACC_BALANCE_SECU SE ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0) GROUP BY TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID)A  ON T.I_CODE = A.I_CODE AND T.A_TYPE = A.A_TYPE AND T.M_TYPE = A.M_TYPE WHERE T.PENETRATEISSUER IN (SELECT DISTINCT TB.PENETRATEISSUER FROM %sTBND TB LEFT JOIN %sTTRD_ACC_BALANCE_SECU SE ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0)) AND T.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND T.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND T.I_CODE NOT LIKE 'UL%%' UNION SELECT T.I_CODE,T.B_NAME,T.A_TYPE,T.M_TYPE,A.ACCID,A.PS_L_AMOUNT,T.B_EXTEND_TYPE,T.PENETRATEISSUER FROM %sTBND T JOIN (SELECT TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT) PS_L_AMOUNT FROM %sTBND TB LEFT JOIN %sTTRD_ACC_BALANCE_SECU SE ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0) AND TB.PENETRATEISSUER IS NULL GROUP BY TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID) A ON T.I_CODE = A.I_CODE AND T.A_TYPE = A.A_TYPE AND T.M_TYPE = A.M_TYPE" % (GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD']);
    WriteLog(sql)
    OracleCR.execute(sql)
    count = 0
    
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode       = res[0]#债券代码
        issueName       = res[1]#债券名称
        assetType       = res[2]#资产类型
        marketCode      = res[3]#市场代码
        accountCode     = res[4]#内证
        amount          = res[5]#持仓金额
        bondExtendType  = res[6]#债券扩展类型
        penetrateIssuer = res[7]#债券风控主体
        count += 1
        
        if marketCode == 'XSHG':
            marketCode = '1';
        elif marketCode == 'XSHE':
            marketCode = '2';
        elif marketCode == 'X_CNBD':
            marketCode = '9';
        
        if accountCode == None:
            accountCode = '';
            
        if amount == None:
            amount = 0;
        
        if penetrateIssuer == None:
            penetrateIssuer = '';
        
        WriteLog("SyncBondPosition,issueCode:%s,issueName:%s,assetType:%s,marketCode:%s,accountCode:%s,amount:%s,bondExtendType:%s,penetrateIssuer:%s" % (issueCode,issueName,assetType,marketCode,accountCode,amount,bondExtendType,penetrateIssuer))
        bondPositionSQL = "REPLACE INTO HT_PositionTable_CFETS(IssueCode,IssueName,MarketCode,AssetType,HTAccountCode,PositionAmount,BondExtendType,PenetrateIssuer,DataDate) VALUES ('%s','%s','%s','%s','%s','%s','%s','%s',date_format(now(),'%%Y%%m%%d'))" % (issueCode,issueName,marketCode,assetType,accountCode,amount,bondExtendType,penetrateIssuer)
        WriteLog(bondPositionSQL)
        try:
           MySqlCR.execute(bondPositionSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-bondPositionSQL")
    if count == 0:
        WriteErrorLog("同步衡泰持仓, 债券持仓0条")
        
    #基金持仓
    sql = "SELECT SE.I_CODE,TF.F_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT) FROM %sTFND TF JOIN %sTTRD_ACC_BALANCE_SECU SE ON TF.I_CODE = SE.I_CODE AND TF.A_TYPE = SE.A_TYPE AND TF.M_TYPE = SE.M_TYPE WHERE TF.M_TYPE IN ('XSHG','XSHE') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0) GROUP BY SE.I_CODE,TF.F_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID" % (GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
    WriteLog(sql)
    OracleCR.execute(sql)
    count = 0
    
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode       = res[0]#债券代码
        issueName       = res[1]#债券名称
        assetType       = res[2]#资产类型
        marketCode      = res[3]#市场代码
        accountCode     = res[4]#内证
        amount          = res[5]#持仓金额
        count += 1
        
        if marketCode == 'XSHG':
            marketCode = '1';
        elif marketCode == 'XSHE':
            marketCode = '2';
        elif marketCode == 'X_CNBD':
            marketCode = '9';
        
        
        WriteLog("SyncBondPosition,issueCode:%s,issueName:%s,assetType:%s,marketCode:%s,accountCode:%s,amount:%s" % (issueCode,issueName,assetType,marketCode,accountCode,amount))
        bondPositionSQL = "REPLACE INTO HT_PositionTable_CFETS(IssueCode,IssueName,MarketCode,AssetType,HTAccountCode,PositionAmount,DataDate) VALUES ('%s','%s','%s','%s','%s','%s',date_format(now(),'%%Y%%m%%d'))" % (issueCode,issueName,marketCode,assetType,accountCode,amount)
        WriteLog(bondPositionSQL)
        try:
           MySqlCR.execute(bondPositionSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-bondPositionSQL")
        
    if count == 0:
        WriteErrorLog("同步衡泰持仓, 基金持仓0条")    
    
    
    #国债期货持仓
    sql = "SELECT SE.I_CODE,SE.ACCID,SE.RT_L_AVAAMOUNT,SE.LS,SE.SECU_EXT_ACCID,SE.PS_L_COST,SE.BEG_DATE,SE.A_TYPE,SE.M_TYPE FROM %sTTRD_ACC_BALANCE_SECU SE WHERE SE.A_TYPE = 'FUT_BD' AND SE.ACCID IN (SELECT COMPONENT ACCID FROM (SELECT * FROM %sTTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('证券投资总部') CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0)" % (GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
    WriteLog(sql)
    OracleCR.execute(sql)
    count = 0
    
    while 1:
        res = OracleCR.fetchone()
        if res == None:
            break
        
        issueCode       = res[0]#合约代码
        accountCode     = res[1]#内证
        quantity        = res[2]#多仓数量
        baSubID         = res[3]#多空方向
        investorID      = res[4]#资金账号
        amount          = res[5]#持仓成本
        
        count += 1
        
        if baSubID == 'S':
            baSubID = '1'
        else:
            baSubID = '3'
        
        WriteLog("SyncTFPosition,issueCode:%s,accountCode:%s,quantity:%s,baSubID:%s,investorID:%s,amount:%s" % (issueCode,accountCode,quantity,baSubID,investorID,amount))
        tfPositionSQL = "REPLACE INTO HT_PositionTable_CFETS(IssueCode,IssueName,MarketCode,AssetType,HTAccountCode,PositionAmount,LS,Amount,DataDate) VALUES ('%s','%s','3','FUT_BD','%s','%s','%s',%s,date_format(now(),'%%Y%%m%%d'))" % (issueCode,issueCode,accountCode,quantity,baSubID,amount)
        WriteLog(tfPositionSQL)
        try:
           MySqlCR.execute(tfPositionSQL)
           MySQlDB.commit()
        except:
           MySQlDB.rollback()
           WriteLog("ERRData-tfPositionSQL")
        
    if count == 0:
        WriteErrorLog("同步衡泰持仓, 国债期货持仓0条")
    

def CalcDeliverBondCF():
    preTradingDate = GetPreTradingDate()
    
    #从数据库获取资金成本
    capitalCost = 2.7;
    sql = "SELECT Field1 FROM dtsdb.StrategyEventTable WHERE EventID = 'CCPBase'";
    WriteLog(sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    for row in results:
        if is_number(row[0]):
            capitalCost = float(row[0])
        
    WriteLog("capitalCost:"+str(capitalCost))
    
    sql = "SELECT main_table.IssueCode,main_table.BondCode,hisPrice.ClearingPrice,issueMaster.SettleDate,(select DTSDate from CalendarTable_CFETS where DTSDate > DATE_FORMAT(now(),'%Y%m%d') and BondTrade = 0 order by DTSDate limit 1),bondEval.Yield*100,bondEval.NetPrice,bondEval.FullPrice,convert(bondInfo.CouponType,binary),convert(bondInfo.CouponFrequency,binary),bondInfo.FixedCouponRate,bondInfo.IssuerPrice,bondInfo.FirstValueDate,bondInfo.ExpirationDate,bondInfo.FaceValue from IssueDeliveryTable_CFETS main_table left join (SELECT IssueCode,ClearingPrice from HistoricalPriceTable where DataDate = (select max(DTSDate) from CalendarTable_CFETS WHERE DTSDate < date_format(now(),'%Y%m%d') and BondTrade = 0)) hisPrice on (main_table.IssueCode = hisPrice.IssueCode) left join (SELECT IssueCode,Yield,NetPrice,FullPrice from BondEvalTable_CFETS where DataDate = (select max(DTSDate) from CalendarTable_CFETS WHERE DTSDate < date_format(now(),'%Y%m%d') and BondTrade = 0)) bondEval on (main_table.BondCode = bondEval.IssueCode) left join BondInfoTable_CFETS bondInfo  on (main_table.BondCode = bondInfo.IssueCode) left join (select * from IssueMasterTable_CFETS where ExpirationDate >=  date_format(now(),'%Y%m%d')) issueMaster  on (main_table.IssueCode = issueMaster.IssueCode) where main_table.UpdateDate = date_format(now(),'%Y%m%d') and main_table.IssueCode in (select IssueCode from IssueMasterTable_CFETS where ExpirationDate >=  date_format(now(),'%Y%m%d') and right(IssueCode,1) = 'P')"
    WriteLog(sql)
    MySqlCR.execute(sql)
    results = MySqlCR.fetchall()
    
    lt = {}
    for row in results:
        issueCode       = row[0]
        bondCode        = row[1]
        ccp_price       = row[2]
        expirationDate  = row[3]
        settleDate      = row[4]
        ytm             = row[5]
        bondPrice       = row[6]
        bondFullPrice   = row[7]
        couponType      = row[8]
        couponFrequency = row[9]
        couponRate      = row[10]
        issuerPrice     = row[11]
        firstValueDate  = row[12]
        lastValueDate   = row[13]
        faceValue       = row[14]
        
        
        WriteLog("CalcDeliverBondCF,issueCode[%s],bondCode[%s],ccp_price[%s],expirationDate[%s],settleDate[%s],ytm[%s],bondPrice[%s],bondFullPrice[%s],couponType[%s],couponFrequency[%s],couponRate[%s],issuerPrice[%s],firstValueDate[%s],lastValueDate[%s],faceValue[%s]" % (issueCode,bondCode,ccp_price,expirationDate,settleDate,ytm,bondPrice,bondFullPrice,couponType,couponFrequency,couponRate,issuerPrice,firstValueDate,lastValueDate,faceValue))
        
        if ccp_price != None and bondPrice != None and ytm != None:
            if is_number(ccp_price) and is_number(bondPrice) and is_number(ytm):
                ccp_price = float(ccp_price)
                bondPrice = float(bondPrice)
                ytm = float(ytm)
                
                bond_detail = {
                '息票类型':couponType,
                '息票付息频率':couponFrequency,
                '固定利率':couponRate*1.0,
                '发行价格':issuerPrice*1.0,
                '起息日':firstValueDate[0:4]+'-'+firstValueDate[4:6]+'-'+firstValueDate[6:8],
                '到期日':lastValueDate[0:4]+'-'+lastValueDate[4:6]+'-'+lastValueDate[6:8],
                '面值':float(faceValue),
                '净价':bondPrice,
                'YTM':ytm
                }
                
                if settleDate == expirationDate:
                    settleDate = gCurrentDate
                
                forward_detail = {
                '交割券结算日':settleDate[0:4]+'-'+settleDate[4:6]+'-'+settleDate[6:8],
                '第二交割日':expirationDate[0:4]+'-'+expirationDate[4:6]+'-'+expirationDate[6:8], #合约最后交易日后的第二个交易日
                'IRR':0,
                '合约价格':ccp_price,
                '净基差':0,
                '融资成本':capitalCost
                }
                
                ret = SBFPricingEngine(forward_detail, bond_detail)
                WriteLog("forward_detail:[%s],bond_detail:[%s],CF:[%s]" % (json.dumps(forward_detail,ensure_ascii=False),json.dumps(bond_detail,ensure_ascii=False),str(ret.cf)))
                
                cf_calc = np.round(ret.cf,6)
                selectedCF = ''
                sql_selectCF = "select CF from IssueDeliveryTable_CFETS where UpdateDate = date_format(now(),'%%Y%%m%%d') and IssueCode = '%s' and BondCode = '%s'" % (issueCode,bondCode)
                WriteLog(sql_selectCF)
                MySqlCR.execute(sql_selectCF)
                resultsCF = MySqlCR.fetchall()
                for row in resultsCF:
                    if is_number(row[0]):
                        selectedCF = np.round(float(row[0]),6)
                    else:
                        WriteErrorLog("IssueCode[%s]BondCode[%s]同步CF:[%s]为空" %(issueCode,bondCode,str(selectedCF)))

                if selectedCF != '' and selectedCF != cf_calc:
                    WriteErrorLog("IssueCode[%s]BondCode[%s]同步CF:[%s]与计算CF:[%s]不一致" %(issueCode,bondCode,str(selectedCF),str(cf_calc)))
                
                
                cfSQL = "update IssueDeliveryTable_CFETS set CF = %.6f where UpdateDate = date_format(now(),'%%Y%%m%%d') and IssueCode = '%s' and BondCode = '%s' AND CF = ''" % (np.round(ret.cf,6),issueCode,bondCode)
                WriteLog(cfSQL)
                try:
                   MySqlCR.execute(cfSQL)
                   MySQlDB.commit()
                except:
                   MySQlDB.rollback()
                   WriteLog("ERRData-cfSQL")
                   
                
                
                ret.IRR = capitalCost/100.0
                ret.Cal_from_IRR()
                net_basis_cf = ret.BNOC / ret.cf
                WriteLog("net_basis_cf:[%s],ccp_price:[%s]" % (str(net_basis_cf),str(ret.F)))
                
                if not lt.has_key(issueCode):
                    lt[issueCode] = [];
                lt[issueCode].append({'bondCode':bondCode,'net_basis_cf':net_basis_cf,'ccp_price':ret.F})
                
            
        
    preTradingDate = GetPreTradingDate()
    #获取到CTD券后更新基准价
    for ccpIssueCode,value1 in lt.items():
        sort_ret = sorted(value1,key=lambda k:k['net_basis_cf'])
        WriteLog(str(sort_ret))
        if sort_ret[0]:
            #防止没有数据不能update,先尝试插入
            ccpBasePriceSQL = "INSERT IGNORE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,BasicPrice) VALUES ('%s','9','%s',%s)" % (ccpIssueCode,preTradingDate,np.round(sort_ret[0]['ccp_price'],4))
            WriteLog(ccpBasePriceSQL)
            try:
               MySqlCR.execute(ccpBasePriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-ccpBasePriceSQL")
               
            #再执行更新
            ccpBasePriceSQL = "update HistoricalPriceTable_CFETS set BasicPrice = %.4f where DataDate = '%s' and IssueCode = '%s'" % (np.round(sort_ret[0]['ccp_price'],4),preTradingDate,ccpIssueCode)
            WriteLog(ccpBasePriceSQL)
            try:
               MySqlCR.execute(ccpBasePriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-ccpBasePriceSQL")
        
def SyncTFPrePrice():
    preTradingDate = GetPreTradingDateExchange()
    
    #获取基准价和结算价到HistoricalPriceTable_CFETS
    sql = "select p.i_code,p.list_set_price,q.dp_set from %ststk_idx_future p left join %stnon_dailyprice q on p.i_code = q.i_code and p.a_type = q.a_type and p.m_type = q.m_type and q.end_date = '2050-12-31' where p.a_type = 'FUT_BD' and p.maturity_date >= to_char(sysdate, 'yyyy-mm-dd') and q.beg_date = '%s-%s-%s'" % (GlobalSettingTable['Oracle_XIR_MD'],GlobalSettingTable['Oracle_XIR_MD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    WriteLog(sql)
    count = 0
    
    try:
        OracleCR.execute(sql)
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
            
            issueCode = res[0]
            basisPrice = res[1]
            clearPrice = res[2]
            count += 1
            
            tfClearPriceSQL = "REPLACE INTO dtsdb.HistoricalPriceTable_CFETS(IssueCode,MarketCode,DataDate,ClearingPrice,BasicPrice) VALUES ('%s','3','%s',%s,%s)" % (issueCode,preTradingDate,clearPrice,basisPrice)
            WriteLog(tfClearPriceSQL)
            try:
               MySqlCR.execute(tfClearPriceSQL)
               MySQlDB.commit()
            except:
               MySQlDB.rollback()
               WriteLog("ERRData-tfClearPriceSQL")
    except:
        WriteLog("Select Error "+sql)

    if count == 0:
        WriteErrorLog("同步国债期货昨结算价, 从结算价表中获取结算价0条")


def getExchangeBondInfo(res):
    reportCode = res[0]
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')
    if not zhmodel.search(reportCode):
        marketCode='0'
        issueCode = res[0]
        if res[1]=='XSHG':
            marketCode='1'
            issueCode = 'SH'+res[0]
        elif res[1]=='XSHE':
            marketCode='2'
            issueCode = 'SZ'+res[0]
        issueName = res[2]
        #productCode='11' 
        bondType = res[3]
        faceValue = res[4]
        expirationDate = res[5][0:4]+res[5][5:7]+res[5][8:10]
        firstValueDate = res[10][0:4]+res[10][5:7]+res[10][8:10]
        tradeLimitDays = res[6]
        couponTypedict={'1':'固定利率','2':'浮动利率','3':'零息票利率'}
        couponType = couponTypedict[res[7]]
        couponFrequency = res[8]
        accrualBasis = res[9]
        fixedCouponRate = res[11]*100
        settlCurrency = res[12]
        issueShortPartyID = res[13]
        issuerPrice = res[14]
        try:listDate=res[15][0:4]+res[15][5:7]+res[15][8:10]#上市日期
        except:listDate='None'
        try:delistDate=res[16][0:4]+res[16][5:7]+res[16][8:10]#摘牌日期
        except:delistDate='None'
        custodianName = res[17]
        today = datetime.date.today()
        d2=datetime.date(int(res[5][0:4]),int(res[5][5:7]),int(res[5][8:10]))
        duration = (d2-today).days
    return (issueCode,marketCode,issueName,reportCode,bondType,faceValue,expirationDate,tradeLimitDays,duration,couponType,couponFrequency,accrualBasis,firstValueDate,fixedCouponRate,settlCurrency,issueShortPartyID,issuerPrice,listDate,delistDate,custodianName)

def getBondInfoTabledata(res):
    data=getExchangeBondInfo(res)
    return data[:-1]

def getHisBondInfoTabledata(res):
    data=getExchangeBondInfo(res)
    return data[:-1]+(gCurrentDate,)

def getIssueMasterTabledata(res):
    data=getExchangeBondInfo(res)
    return (data[0],data[2],data[-6],data[6],data[5],data[1])

def getIssueMarketTabledata(res):
    data=getExchangeBondInfo(res)
    return (data[0],data[1],data[-3])

def getHistoricalPriceTable(res):
    data=getExchangeBondInfo(res)
    return (data[0],data[1],gCurrentDate)

def SyncExchangeBondInfo():
    global GlobalSettingTable
    sql = "select I_CODE,M_TYPE,B_NAME,WIND_CLASS1,B_PAR_VALUE,B_MTR_DATE,B_TERM,B_COUPON_TYPE,B_CASH_TIMES,B_DAYCOUNT,B_START_DATE,B_COUPON,CURRENCY,ISSUER_CODE,B_ISSUE_PRICE,B_LIST_DATE,B_DELIST_DATE,HOST_MARKET from %sTBND WHERE M_TYPE='XSHG' and B_MTR_DATE>= TO_CHAR(SYSDATE, 'YYYY-MM-DD')" %(GlobalSettingTable['Oracle_XIR_MD'])
    OracleCR.execute(sql)
    WriteLog(sql)
    #I_CODE,M_TYPE,B_NAME,WIND_CLASS1,B_PAR_VALUE,B_MTR_DATE,B_TERM,B_COUPON_TYPE,
    #0,     1,     2,     3,          4,          5,         6,     7, 
    #B_CASH_TIMES,B_DAYCOUNT,B_START_DATE,B_COUPON,CURRENCY,ISSUER_CODE,B_ISSUE_PRICE,B_LIST_DATE,B_DELIST_DATE,HOST_MARKET 
    #8,           9,         10,          11,      12,      13,         14            15上市日    16摘牌日       17托管市场
    count=0
    while True:
        results = OracleCR.fetchmany(int(GlobalSettingTable['NumberofLines_inserted']))
        if not results:
            if count==0:WriteErrorLog("同步上交所债券信息0条")
            break
        else:
            count+=1
            WriteLog('fetch ExchangeBondInfo '+str(len(results))+' row')
            #tablesummary=list(map(getExchangeBondInfo,results))
            BondInfotablesummary=list(map(getBondInfoTabledata,results))
            #插入BondInfoTable
            bondInfoSQL = "REPLACE INTO dtsdb.BondInfoTable_CFETS(IssueCode,MarketCode,IssueName,ReportCode,ProductCode,BondType,FaceValue,ExpirationDate,ListingDate,DelistingDate,TradeLimitDays,Duration,CouponType,CouponFrequency,AccrualBasis,FirstValueDate,FixedCouponRate,SettlCurrency,IssuerShortPartyID,IssuerPrice,CustodianName) VALUES (%s,%s,%s,%s,11,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
            WriteLog(bondInfoSQL)
            try:
                MySqlCR.executemany(bondInfoSQL,BondInfotablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog('ERRData-bondInfoSQL')

            hisBondInfoTablesummary=list(map(getHisBondInfoTabledata,results))
            #插入HisBondInfoTable
            hisBondInfoSQL = "REPLACE INTO dtsdb.HisBondInfoTable_CFETS(IssueCode,MarketCode,IssueName,ReportCode,ProductCode,BondType,FaceValue,ExpirationDate,ListingDate,DelistingDate,TradeLimitDays,Duration,CouponType,CouponFrequency,AccrualBasis,FirstValueDate,FixedCouponRate,SettlCurrency,IssuerShortPartyID,IssuerPrice,CustodianName,DataDate) VALUES (%s,%s,%s,%s,11,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);" 
            WriteLog(hisBondInfoSQL)        
            try:
                MySqlCR.executemany(hisBondInfoSQL,hisBondInfoTablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog('ERRData-hisBondInfoSQL')
                    
            issueMasterTablesummary=list(map(getIssueMasterTabledata,results))
            #插入IssueMasterTable
            issueMasterSQL = "REPLACE INTO dtsdb.IssueMasterTable(IssueCode,IssueShortName,ProductCode,Currency,ExpirationDate,FaceValue,PriorMarket,UnderlyingAssetCode,ContractSize) VALUES (%s,%s,11,%s,%s,%s,%s,0,1);"
            WriteLog(issueMasterSQL)
            try:
                MySqlCR.executemany(issueMasterSQL,issueMasterTablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog('ERRData-issueMasterSQL')

            issueMarketTablesummary=list(map(getIssueMarketTabledata,results))
            #插入IssueMarketTable
            issueMarketSQL = "REPLACE INTO dtsdb.IssueMarketTable(IssueCode,MarketCode,ListedDate) VALUES (%s,%s,%s);"
            WriteLog(issueMarketSQL)        
            try:
                MySqlCR.executemany(issueMarketSQL,issueMarketTablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog('ERRData-issueMarketSQL')
                
            historicalPriceTablesummary=list(map(getHistoricalPriceTable,results))
            #插入HistoricalPriceTable
            historicalPriceSQL = "INSERT IGNORE INTO dtsdb.HistoricalPriceTable(IssueCode,MarketCode,DataDate,MarkPrice,ClosePrice,AdjustedClosePrice,OpenPrice,HighPrice,LowPrice,Volume,UpperLimitPrice,LowerLimitPrice,MMLNBestBid,ReserveString,CreateTime,TimeStamp,DTSTimeStamp,WeeklyHighPrice,WeeklyLowPrice,MonthlyHighPrice,MonthlyLowPrice,QuarterHighPrice,QuarterLowPrice,Psychological,WeightGiftCounter,WeightSellCounter,WeightSellPrice,WeightDividend,WeightIncCounter,WeightOwnerShip,WeightFreeCounter,ClearingPrice,MMBestBid1) select %s,%s,(select max(DTSDate) from CalendarTable_CFETS where DTSDate < %s AND BondTrade = '0'), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, now(), now(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 from dual"
            WriteLog(historicalPriceSQL)        
            try:
                MySqlCR.executemany(historicalPriceSQL,historicalPriceTablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog("ERRData-historicalPriceSQL")


def getExchangeBondClosePrice(res):
    marketCode = '0'
    issueCode=res[1]
    if res[0]=='XSHG':
        marketCode='1'
        issueCode='SH'+res[1]
    elif res[0]=='XSHE':
        marketCode='2'
        issueCode='SZ'+res[1]
    closePrice=res[2]
    return (closePrice,issueCode,marketCode,ExchangepreTradingDate)


def UpdateExchangeBondClosePrice():
    global ExchangepreTradingDate
    ExchangepreTradingDate =GetPreTradingDateExchange().encode('utf-8')
    sql = "select M_TYPE,I_CODE,EVAL_NETPRICE from %sTtrd_Otc_Instrument_Eval p where (p.M_TYPE='XSHG') and p.eval_source='债券交易所收盘价' and p.beg_date='%s-%s-%s'"%(GlobalSettingTable['Oracle_XIR_TRD'],ExchangepreTradingDate[0:4],ExchangepreTradingDate[4:6],ExchangepreTradingDate[6:8])
    OracleCR.execute(sql)
    WriteLog(sql)
    #M_TYPE,I_CODE,EVAL_NETPRICE
    #0,     1,     2,    
    count=0
    while True:
        results = OracleCR.fetchmany(int(GlobalSettingTable['NumberofLines_inserted']))
        if not results:
            if count==0:WriteErrorLog("同步上交所债券收盘价信息0条")
            break
        else:
            count+=1
            WriteLog('fetch ExchangeBondClosePrice '+str(len(results))+' row')
            tablesummary=list(map(getExchangeBondClosePrice,results))
            #for row in tablesummary[:10]:
                #print row
            #插入HistoricalPriceTable
            historicalPriceSQL = "UPDATE dtsdb.HistoricalPriceTable SET AdjustedClosePrice=%s  WHERE IssueCode=%s and MarketCode=%s and DataDate=%s"
            WriteLog(historicalPriceSQL)        
            try:
                MySqlCR.executemany(historicalPriceSQL,tablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog("ERRClosepriceData-historicalPriceSQL")


def getExchangeBondZhongZhaiguzhi(res):
    issueCode=res[1]
    if res[5]=='XSHG':
        issueCode='SH'+res[1]
    elif res[5]=='XSHE':
        issueCode='SZ'+res[1]
    netPrice=res[3]
    fullPrice=res[4]
    zhongzhaiyield=res[2]
    return (issueCode,preTradingDate,zhongzhaiyield,netPrice,fullPrice)


def SyncExchangeBondEval():
    global preTradingDate
    preTradingDate =GetPreTradingDate()
    sql = "select k.beg_date,k.i_code,k.yield,k.netprice,k.fullprice,k.M_TYPE from %sTCB_BOND_EVAL k where k.beg_date ='%s-%s-%s' and (M_TYPE = 'XSHG') order by i_code" % (GlobalSettingTable['Oracle_XIR_MD'],preTradingDate[0:4],preTradingDate[4:6],preTradingDate[6:8])
    #k.beg_date,k.i_code,k.yield,k.netprice,k.fullprice,k.M_TYPE
    #0,          1,       2,       3          4          5
    OracleCR.execute(sql)
    count=0
    while True:
        results = OracleCR.fetchmany(int(GlobalSettingTable['NumberofLines_inserted']))
        if not results:
            if count==0:WriteErrorLog("同步上交所债券估值信息0条")
            break
        else:
            count+=1
            WriteLog('fetch ExchangeBondEval '+str(len(results))+' row')
            tablesummary=list(map(getExchangeBondZhongZhaiguzhi,results))
            bondEvalSQL = "REPLACE INTO dtsdb.BondEvalTable_CFETS(IssueCode,DataDate,Yield,NetPrice,FullPrice) VALUES (%s,%s,%s,%s,%s)"
            WriteLog(bondEvalSQL)        
            try:
                MySqlCR.executemany(bondEvalSQL,tablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog("ERRBondEvalTable_CFETSSQL")


def getExchangeBondCodeInMarket(res):
    issueCode=str(res[0])
    marketcode=0
    if res[2]=='XSHG':
        issueCode='SH'+str(res[0])
        marketcode=1
    elif res[2]=='XSHE':
        issueCode='SZ'+str(res[0])
        marketcode=2
    elif res[2]=='X_CNBD':
        issueCode='IB'+str(res[0])
        marketcode=9
    shcode=''
    szcode=''
    yhcode=''
    if str(res[3]) !='None':
        shcode='SH'+str(res[3])
    if str(res[4]) !='None':
        szcode='SZ'+str(res[4])
    if str(res[5]) !='None':
        yhcode='IB'+str(res[5])
    return (issueCode,marketcode,shcode,szcode,yhcode,res[1])


def SyncExchangeBondCodeInMarket():
    global preTradingDate
    preTradingDate =GetPreTradingDate()
    sql = "select I_CODE,A_TYPE,M_TYPE,SH_CODE,SZ_CODE,YH_CODE from %stbnd where B_MTR_DATE>= TO_CHAR(SYSDATE, 'YYYY-MM-DD')"% (GlobalSettingTable['Oracle_XIR_MD'])
    #I_CODE,A_TYPE,M_TYPE,SH_CODE,SZ_CODE,YH_CODE
    #0,     1,      2,    3        4      5
    OracleCR.execute(sql)
    #print len(results)
    #for row in results[:10]:
        #print row
    count=0
    while True:
        results = OracleCR.fetchmany(int(GlobalSettingTable['NumberofLines_inserted']))
        if not results:
            if count==0:WriteErrorLog("同步债券不同市场关系信息0条")
            break
        else:
            count+=1
            WriteLog('fetch ExchangeBondCodeInMarket '+str(len(results))+' row')
            tablesummary=list(map(getExchangeBondCodeInMarket,results))
            bondcodeSQL = "REPLACE INTO dtsdb.BondCodeInMarket_CFETS(IssueCode,MarketCode,SH_Code,SZ_Code,YH_Code,AssetType) VALUES (%s,%s,%s,%s,%s,%s)"
            WriteLog(bondcodeSQL)        
            try:
                MySqlCR.executemany(bondcodeSQL,tablesummary)
                MySQlDB.commit()
            except:
                MySQlDB.rollback()
                WriteLog("ERRBondCodeInMarket_CFETSSQL")


#################################################初始化#################################################
global file

#先读取配置文件
configFile = open(sys.path[0]+'\Sync_CFETS.config', 'r')
configText = configFile.readlines()
GlobalSettingTable = {}
for i in range(0, len(configText)):
    text = (configText[i].rstrip('\n')).strip()
    if len(text) > 0 and text[0:1] != "#":
        key = ((re.split('[=]+',text))[0]).strip()
        value = ((re.split('[=]+',text))[1]).strip()
        GlobalSettingTable[key] = value
    
gCurrentDate = time.strftime('%Y%m%d',time.localtime(time.time()))
path = GlobalSettingTable['Log_Path']
if path[-1:] == '\\':
    path = path[:-1]
filePath = "%s\Sync_CFETS_%s.log" % (path, gCurrentDate)

errorFilePath = "%s\Sync_CFETS_Error_%s.log" % (path, gCurrentDate)
file = open(filePath, 'a+')
errorFile = open(errorFilePath, 'a+')

WriteLog('gCurrentDate:'+gCurrentDate)

#设置Oracle的字符集
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

#连接Oracle数据库
OracleDB = cx_Oracle.connect('%s/%s@%s:%s/%s' % (GlobalSettingTable['Oracle_User'],GlobalSettingTable['Oracle_PassWord'],GlobalSettingTable['Oracle_IP'],GlobalSettingTable['Oracle_Port'],GlobalSettingTable['Oracle_DB']))
OracleCR = OracleDB.cursor()

#连接MySQL数据库
MySQlDB = MySQLdb.connect(GlobalSettingTable['MySQL_IP'], GlobalSettingTable['MySQL_User'], GlobalSettingTable['MySQL_PassWord'], GlobalSettingTable['MySQL_DB'], charset='latin1')
MySqlCR = MySQlDB.cursor()

#################################################初始化结束#################################################

#################################################处理开始#################################################


WriteLog("Sync START...")
#清除过期债券数据
ClearBondInfo()
#同步债券数据
SyncBondInfo()
#同步标债远期数据
SyncCCPInfo()
#同步交易日
SyncTradingDate()
#同步交易成员信息
SyncTradeMember()
#同步交易对手信息（文本解析）
SyncTradeParty()
#同步中债估值
SyncBondEval()
#同步CCP昨结算价
SyncCCPClearPrice()
#同步XBOND可交易债
SyncAvlTradeBondInfo()
#获取自动订阅的XBOND债券
SyncAutoSubscribeXBOND()
#计算昨基准
CalcCCPPreBasePrice()
#同步IRS昨日收盘曲线
SyncIRSCurve()
#同步私募可交换债与正股关系
SyncBondConversion()
#同步衡泰持仓
SyncBondPosition()
#计算实物交割券的转换因子
CalcDeliverBondCF()
#同步国债期货基准价和结算价
SyncTFPrePrice()
#同步交易所债券
SyncExchangeBondInfo()
#同步交易所债券的收盘价
UpdateExchangeBondClosePrice()
#同步交易所债券代码与其他市场代码的转换
SyncExchangeBondCodeInMarket()


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
# WriteLog("forward_detail:[%s],bond_detail:[%s],CF:[%s]" % (json.dumps(forward_detail,ensure_ascii=False),json.dumps(bond_detail,ensure_ascii=False),str(ret.cf)))


# ret.IRR = 5.0001/100.0
# ret.Cal_from_IRR()
# net_basis_cf = ret.BNOC / ret.cf
# WriteLog("BNOC:[%s],cf:[%s],net_basis_cf:[%s],ccp_price:[%s]" % (str(ret.BNOC),str(ret.cf),str(net_basis_cf),str(ret.F)))
'''
WriteLog('Sync END...')

#################################################处理结束#################################################

#关闭连接
OracleCR.close()
OracleDB.close()
MySQlDB.close()

file.close()
errorFile.close()
