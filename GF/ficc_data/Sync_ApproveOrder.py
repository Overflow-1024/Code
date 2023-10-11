# -*- coding: utf-8 -*-
#20220328
#增加交易所债券审批单
#20220314
#增加国债期货资金查询
#修改国债期货审批单要素
#20211108
#增加国债期货审批单
#20210406
#双边做市审批单增加收益率字段
#20210120
#使用命令行参数,区分不同业务品种的审批单
#20201105
#审批单新增导入清算方式和结算方式
#20200310
#支持查询衡泰审批单并导入到DTS数据库中
#from __future__ import division
import cx_Oracle
import datetime
import os
import MySQLdb
import re
import time
import sys

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


#查询Oracle
def SyncApproveOrder():
    global GlobalSettingTable
    
    
    #查询衡泰数据库
    
    #双边做市报价
    if str(sys.argv[2]) == 'SBZSBJ':
        sql_2 = "SELECT T.ORDDATE,SUBSTR(T.ORDTIME,12,8),T.SYSORDID,T.ORDSTATUS,T.TRDTYPE,T.SECU_ACCID,T.I_CODE,T.SETDAYS,ROUND(M1.BID_NETPRICE,4),M1.BID_ORDCOUNT/100,BID_ORDCOUNT/100-DECODE(M2.B_ORDERQTY,'',0,M2.B_LEAVESQTY)/10000-NVL(M3.BIDAMOUNT,0)/10000 BIDREMAINQTY,ROUND(M1.ASK_NETPRICE,4),M1.ASK_ORDCOUNT/100,M1.ASK_ORDCOUNT/100-DECODE(M2.S_ORDERQTY,'',0,M2.S_LEAVESQTY)/10000-NVL(M3.ASKAMOUNT,0)/10000 ASKREMAINQTY,T.OPERATOR,T.BND_SETTYPE,M1.BID_YTM,M1.ASK_YTM FROM %sTTRD_OTC_TRADE T INNER JOIN %sTTRD_OTC_MARKETMAKERQUOTE M1 ON T.SYSORDID = M1.SYSORDID LEFT JOIN %sTTRD_CFETS_TRADE_MM M2 ON T.SYSORDID = SUBSTR(M2.CLORDID_CLIENT_ID,1,INSTR(M2.CLORDID_CLIENT_ID, '-') - 1) AND M2.STATUS IN('5','10') LEFT JOIN (SELECT SUBSTR(TB.CLORDID_CLIENT_ID,1,INSTR(TB.CLORDID_CLIENT_ID, '-') - 1) SYSORDID,NVL(SUM(CASE TB.SIDE WHEN '1' THEN TB.LASTQTY END),0) BIDAMOUNT,NVL(SUM(CASE TB.SIDE WHEN '2' THEN TB.LASTQTY END ),0) ASKAMOUNT FROM %sTTRD_CFETS_TRADE_EXEREPORT TB WHERE TB.TRADEMETHOD = '5' AND TB.QUOTETYPE = '107' AND TB.TRADEDATE = TO_CHAR(SYSDATE, 'YYYYMMDD') GROUP BY SUBSTR(TB.CLORDID_CLIENT_ID,1,INSTR(TB.CLORDID_CLIENT_ID, '-') - 1)) M3 ON M3.SYSORDID = T.SYSORDID WHERE T.TRDTYPE = '16' AND T.ORDDATE = TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND T.ORDSTATUS IN ('-4', '5') ORDER BY T.SYSORDID" %(GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
        OracleCR.execute(sql_2)
        WriteLog(sql_2)
        
        sql_Approve_1 = "INSERT INTO dtsdb.HT_ApproveOrderTable_CFETS (IssueCode,OrderID,OrderDate,OrderTime,OrderStatus,HTAccountCode,ClearSpeed,BidNetPrice,BidYTM,BidQuantity,BidRemainQty,AskNetPrice,AskYTM,AskQuantity,AskRemainQty,InvestorID,ClearMethod,SettlType,ApproveType,DataDate) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
       
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
        
            tradeDate       = str(res[0])          #交易日期
            tradeTime       = str(res[1])[11:-1]   #交易时间
            orderID         = str(res[2])          #交易编号
            status          = str(res[3])          #交易状态
            direction       = str(res[4])          #交易方向
            htAccountID     = str(res[5])          #内证
            issueCode       = 'IB'+str(res[6])     #债券代码
            clearSpeed      = res[7]               #清算速度:0表示T+0，1表示T+1
            bidNetPrice     = res[8]               #买净价
            bidQuantity     = res[9]               #买总量
            bidRemainQty    = res[10]              #买剩余量
            askNetPrice     = res[11]              #卖净价
            askQuantity     = res[12]              #卖总量
            askRemainQty    = res[13]              #卖剩余量
            investorID      = res[14]              #衡泰操作员
            settlType       = res[15]              #结算方式
            bidYTM          = res[16]              #买收益率
            askYTM          = res[17]              #卖收益率
            
            clearMethod     = ''
            
            tradeDate = tradeDate[0:4]+tradeDate[5:7]+tradeDate[8:10]
            
            if is_number(clearSpeed):
                clearSpeed = str(clearSpeed + 1)
            if is_number(bidQuantity):
                bidQuantity = str(bidQuantity * 10000)
            if is_number(bidRemainQty):
                bidRemainQty = str(bidRemainQty * 10000)
            if is_number(askQuantity):
                askQuantity = str(askQuantity * 10000)
            if is_number(askRemainQty):
                askRemainQty = str(askRemainQty * 10000)
                
                
            if settlType == 'DVP':
                settlType = "0";
                clearMethod = '13';
            elif settlType == 'NDVP':
                settlType = "0";
                clearMethod = '6';
            else:
                settlType = '';
            
            
            WriteLog('SyncApproveOrder,tradeDate[%s],tradeTime[%s],orderID[%s],status[%s],direction[%s],htAccountID[%s],issueCode[%s],clearSpeed[%s],bidNetPrice[%s],bidYTM[%s],bidQuantity[%s],bidRemainQty[%s],askNetPrice[%s],askYTM[%s],askQuantity[%s],askRemainQty[%s],investorID[%s],clearMethod[%s],settlType[%s]' % (str(tradeDate),str(tradeTime),str(orderID),str(status),str(direction),str(htAccountID),str(issueCode),str(clearSpeed),str(bidNetPrice),str(bidYTM),str(bidQuantity),str(bidRemainQty),str(askNetPrice),str(askYTM),str(askQuantity),str(askRemainQty),str(investorID),str(clearMethod),str(settlType)))
            
            try:
                val = (issueCode,orderID,tradeDate,tradeTime,status,htAccountID,clearSpeed,bidNetPrice,bidYTM,bidQuantity,bidRemainQty,askNetPrice,askYTM,askQuantity,askRemainQty,investorID,clearMethod,settlType,str(sys.argv[2]),gCurrentDate)
                MySQLCR.execute(sql_Approve_1,val)
            except MySQLdb.Error as e:
                WriteLog('insert error!{}'.format(e))

        #插入一条审批单号为-1的记录表示查询结束
        try:
            val = ('','-1','','','','','','','','','','','','','','','','',str(sys.argv[2]),gCurrentDate)
            MySQLCR.execute(sql_Approve_1,val)
        except MySQLdb.Error as e:
            WriteLog('insert error!{}'.format(e))
        
    #XSWAP普通IRS
    elif str(sys.argv[2]) == 'XSWAPPTIRS':
        sql_2 = "SELECT T.ORDDATE,SUBSTR(T.ORDTIME, 12, 8),T.SYSORDID,T.ORDSTATUS,T.TRDTYPE,T.SECU_ACCID,T.I_CODE_EXH,T.SETDAYS,T.ORDPRICE,T.ORDCOUNT,T.ORDCOUNT - (case when M2.ORDAMOUNT is null then 0 else M2.ORDAMOUNT end) REMAINQTY,T.OPERATOR,T.BND_SETTYPE,TE.CFETS_PRICETOPLIMIT,TE.CFETS_PRICELOWLIMIT FROM %sTTRD_OTC_TRADE T LEFT JOIN (SELECT SUM(TO_NUMBER (NVL (O.ORDERQTY, 0))- CASE WHEN O.STATUS IN (7) THEN TO_NUMBER (NVL (O.ORDERQTY, 0)) WHEN O.STATUS IN (6, 2) THEN TO_NUMBER (NVL (O.ORDERQTY, 0)) WHEN O.STATUS IN (15) THEN TO_NUMBER (NVL (O.LEAVESQTY, 0)) ELSE 0 END) AS ORDAMOUNT,SYSORDID FROM %sTTRD_CFETS_TRADE_XSWAP O WHERE O.STATUS in(0,1,2,5,6,7,10,11,14,15) AND  O.TRADEDATE = TO_CHAR(SYSDATE, 'YYYYMMDD')GROUP BY SYSORDID ) M2 ON T.SYSORDID = M2.SYSORDID LEFT JOIN %sTTRD_OTC_TRADE_EXTEND TE ON TE.SYSORDID = T.SYSORDID WHERE T.TRDTYPE IN ('913','914') AND T.ORDDATE = TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND T.ORDSTATUS IN ('-4', '5') ORDER BY T.SYSORDID" %(GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_TRD'])
        OracleCR.execute(sql_2)
        WriteLog(sql_2)
        
        sql_Approve_1 = "INSERT INTO dtsdb.HT_ApproveOrderTable_CFETS (IssueCode,OrderID,OrderDate,OrderTime,OrderStatus,HTAccountCode,ClearSpeed,BidNetPrice,BidQuantity,BidRemainQty,AskNetPrice,AskQuantity,AskRemainQty,InvestorID,ClearMethod,SettlType,ApproveType,DataDate) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
       
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
        
            tradeDate       = str(res[0])          #交易日期
            tradeTime       = str(res[1])[11:-1]   #交易时间
            orderID         = str(res[2])          #交易编号
            status          = str(res[3])          #交易状态
            direction       = str(res[4])          #交易方向
            htAccountID     = str(res[5])          #内证
            issueCode       = 'IB'+str(res[6])     #债券代码
            clearSpeed      = res[7]               #清算速度:0表示T+0，1表示T+1
            netPrice        = res[8]               #净价
            quantity        = str(res[9])          #总量
            remainQty       = str(res[10])         #剩余量
            investorID      = res[11]              #衡泰操作员
            settlType       = res[12]              #结算方式
            upperLimitPrice = res[13]              #上限
            lowerLimitPrice = res[14]              #下限
            
            clearMethod     = ''
            bidQuantity     = ''
            bidRemainQty    = ''
            askQuantity     = ''
            askRemainQty    = ''
            
            tradeDate = tradeDate[0:4]+tradeDate[5:7]+tradeDate[8:10]
            
            if direction == '913':
                bidQuantity = quantity
                bidRemainQty = remainQty
            if direction == '914':
                askQuantity = quantity
                askRemainQty = remainQty
            
            
            WriteLog('SyncApproveOrder,tradeDate[%s],tradeTime[%s],orderID[%s],status[%s],direction[%s],htAccountID[%s],issueCode[%s],clearSpeed[%s],upperLimitPrice[%s],bidQuantity[%s],bidRemainQty[%s],lowerLimitPrice[%s],askQuantity[%s],askRemainQty[%s],investorID[%s],clearMethod[%s],settlType[%s]' % (str(tradeDate),str(tradeTime),str(orderID),str(status),str(direction),str(htAccountID),str(issueCode),str(clearSpeed),str(upperLimitPrice),str(bidQuantity),str(bidRemainQty),str(lowerLimitPrice),str(askQuantity),str(askRemainQty),str(investorID),str(clearMethod),str(settlType)))
            
            try:
                val = (issueCode,orderID,tradeDate,tradeTime,status,htAccountID,clearSpeed,upperLimitPrice,bidQuantity,bidRemainQty,lowerLimitPrice,askQuantity,askRemainQty,investorID,clearMethod,settlType,str(sys.argv[2]),gCurrentDate)
                MySQLCR.execute(sql_Approve_1,val)
            except MySQLdb.Error as e:
                WriteLog('insert error!{}'.format(e))

        #插入一条审批单号为-1的记录表示查询结束
        try:
            val = ('','-1','','','','','','','','','','','','','','',str(sys.argv[2]),gCurrentDate)
            MySQLCR.execute(sql_Approve_1,val)
        except MySQLdb.Error as e:
            WriteLog('insert error!{}'.format(e))
    #国债期货审批单
    elif str(sys.argv[2]) == 'GZQH':
        sql_2 = "SELECT T.ORDDATE,SUBSTR(T.ORDTIME,12,8),T.SYSORDID,T.ORDSTATUS,T.TRDTYPE,T.SECU_ACCID,T.I_CODE,T.ORDPRICE,T.ORDCOUNT,T.ORDAMOUNT,T.OCFLAG,T.ORDERPRICETYPE,T.ORDSOURCE,T.OPERATOR FROM %sTTRD_OTC_TRADE T JOIN %sTSTK_IDX_FUTURE TF ON T.I_CODE = TF.I_CODE AND TF.M_TYPE = 'X_CNFFEX' WHERE T.TRDTYPE IN ('310','311') AND T.ORDDATE = TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND T.ORDSTATUS IN ('-4', '5') ORDER BY T.SYSORDID" %(GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_MD'])
        OracleCR.execute(sql_2)
        WriteLog(sql_2)
        
        sql_Approve_1 = "INSERT INTO dtsdb.HT_ApproveOrderTable_CFETS (IssueCode,OrderID,OrderDate,OrderTime,OrderStatus,HTAccountCode,ClearSpeed,BidNetPrice,BidYTM,BidQuantity,BidRemainQty,AskNetPrice,AskYTM,AskQuantity,AskRemainQty,InvestorID,ClearMethod,SettlType,ApproveType,DataDate) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
       
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
        
            tradeDate       = str(res[0])          #交易日期        ordDate
            tradeTime       = str(res[1])[11:-1]   #交易时间        ordTime
            orderID         = str(res[2])          #交易编号        sysOrdID
            status          = str(res[3])          #交易状态        ordStatus
            bidYTM          = str(res[4])          #交易方向  310期货买入，311期货卖出       trdType
            htAccountID     = str(res[5])          #内证            secu_accID
            issueCode       = str(res[6])          #债券代码        i_code
                                                                
            bidNetPrice     = str(res[7])          #净价              ordPrice
            quantity        = str(res[8])          #总量             ordCount
            remainQty       = str(res[9])          #剩余量            ordAmount
            askYTM          = str(res[10])         #开平 字母O表示开仓，C表示平仓       ocFlag
            orderPriceType  = str(res[11])         #结算方式           orderPriceType
            source          = str(res[12])         #上限               source  
            investorID      = str(res[13])         #衡泰操作员

            
            clearSpeed      = ''               #清算速度:0表示T+0，1表示T+1  
            clearMethod     = ''
            bidRemainQty    = ''
            askQuantity     = ''
            askRemainQty    = ''
            settlType       = ''
            askNetPrice     = ''

            tradeDate = tradeDate[0:4]+tradeDate[5:7]+tradeDate[8:10]
            bidQuantity = quantity
            
            if bidYTM == '310': #买
                bidYTM = '3'        
            if bidYTM == '311': #卖
                bidYTM = '1'

            if askYTM == "O": #开
                askYTM = '0'
            if askYTM == "C": #平
                askYTM = '1'
            
            
            WriteLog('SyncApproveOrder,tradeDate[%s],tradeTime[%s],orderID[%s],status[%s],direction[%s],htAccountID[%s],issueCode[%s],clearSpeed[%s],upperLimitPrice[%s],bidQuantity[%s],bidRemainQty[%s],lowerLimitPrice[%s],askQuantity[%s],askRemainQty[%s],investorID[%s],clearMethod[%s],settlType[%s]' % (str(tradeDate),str(tradeTime),str(orderID),str(status),str(bidYTM),str(htAccountID),str(issueCode),str(clearSpeed),str(bidNetPrice),str(bidQuantity),str(bidRemainQty),str(askNetPrice),str(askQuantity),str(askRemainQty),str(investorID),str(clearMethod),str(settlType)))
            
            try:
                val = (issueCode,orderID,tradeDate,tradeTime,status,htAccountID,clearSpeed,bidNetPrice,bidYTM,bidQuantity,bidRemainQty,askNetPrice,askYTM,askQuantity,askRemainQty,
                    investorID,clearMethod,settlType,str(sys.argv[2]),gCurrentDate)
                MySQLCR.execute(sql_Approve_1,val)
            except MySQLdb.Error as e:
                WriteLog('insert error!{}'.format(e))

        #插入一条审批单号为-1的记录表示查询结束
        try:
            val = ('','-1','','','','','','','','','','','','','','','','',str(sys.argv[2]),gCurrentDate)
            MySQLCR.execute(sql_Approve_1,val)
        except MySQLdb.Error as e:
            WriteLog('insert error!{}'.format(e))
    #国债期货资金
    elif str(sys.argv[2]) == 'Fund_GZQH':
        sql_2 = "SELECT SC.BEG_DATE ,SC.ACCID,SC.RT_AMOUNT,SC.RT_MARGIN,SC.RT_AVAAMOUNT,SC.RT_FREAMOUNT,SC.RT_UPDATETIME FROM %sTTRD_EXH_ACC_BALANCE_CASH_EXT SC WHERE SC.ACCID = '%s'" %(GlobalSettingTable['Oracle_XIR_TRD'],str(sys.argv[3]))
        OracleCR.execute(sql_2)
        WriteLog(sql_2)
        
        sql_Fund_1 = "INSERT INTO dtsdb.StrategyEventTable (PortfolioID,StrategyID,EventID,KeyField,VersionID,Field1,Field2,Field3,Field4,Field5,Field6) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
       
        while 1:
            res = OracleCR.fetchone()
            if res == None:
                break
        
            htAccountID     = res[1]
            amount          = res[2]
            margin          = res[3]
            avlAmount       = res[4]
            freAmount       = res[5]
            
            WriteLog('SyncFund,htAccountID[%s],amount[%s],margin[%s],avlAmount[%s],freAmount[%s]' % (str(htAccountID),str(amount),str(margin),str(avlAmount),str(freAmount)))
            
            try:
                val = ("","",str(sys.argv[2]),"",gCurrentDate,htAccountID,amount,margin,avlAmount,freAmount,gCurrentDate)
                MySQLCR.execute(sql_Fund_1,val)
            except MySQLdb.Error as e:
                WriteLog('insert error!{}'.format(e))

        #插入一条审批单号为-1的记录表示查询结束
        try:
            val = ('','-1',str(sys.argv[2]),'',gCurrentDate,'','','','','',gCurrentDate)
            MySQLCR.execute(sql_Fund_1,val)
        except MySQLdb.Error as e:
            WriteLog('insert error!{}'.format(e))
    #交易所债券
    elif str(sys.argv[2]) == 'JYSZQ':
        sql_2 = "SELECT T.ORDDATE, SUBSTR(T.ORDTIME,12,8), T.SYSORDID, T.ORDSTATUS, T.TRDTYPE,T.SECU_ACCID, T.I_CODE ,T.BND_NETPRICE, T.ORDCOUNT, T.ORDAMOUNT, T.ORDERPRICETYPE,T.ORDSOURCE,T.OPERATOR,T.M_TYPE,T.SETDAYS FROM %sTTRD_OTC_TRADE T JOIN %sTBND TB ON T.I_CODE = TB.I_CODE AND (TB.M_TYPE = 'XSHE' or TB.M_TYPE = 'XSHG') WHERE T.TRDTYPE IN ('10','20') AND T.ORDDATE = TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND T.ORDSTATUS IN ('-4', '5') AND (T.M_TYPE = 'XSHE' or T.M_TYPE = 'XSHG') ORDER BY T.SYSORDID" % (GlobalSettingTable['Oracle_XIR_TRD'],GlobalSettingTable['Oracle_XIR_MD'])
        #T.ORDDATE 交易日期, SUBSTR(T.ORDTIME,12,8) 交易时间, T.SYSORDID 审批单号, T.ORDSTATUS 审批单状态, T.TRDTYPE 交易类型, T.SECU_ACCID 内证代码, T.I_CODE 债券代码,
        #0                  1                                  2                  3                         4                   5                 6
        #T.BND_NETPRICE 交易净价, T.ORDCOUNT 交易数量, T.ORDAMOUNT 交易金额, T.ORDERPRICETYPE　价格方式,T.ORDSOURCE 报价来源,T.OPERATOR,T.M_TYPE,T.SETDAYS
        #7                   8                    9                     10                         11                   12           13      14
        OracleCR.execute(sql_2)
        results=OracleCR.fetchall()
        sql_Approve_1 = "INSERT ignore INTO dtsdb.HT_ApproveOrderTable_CFETS (IssueCode,OrderID,OrderDate,OrderTime,OrderStatus,HTAccountCode,ClearSpeed,BidNetPrice,BidQuantity,BidRemainQty,AskNetPrice,AskQuantity,AskRemainQty,InvestorID,ApproveType,DataDate) Values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        tablesummary=list(map(getJYSZQApprove,results))
        if len(tablesummary)>0:
            try:
                MySQLCR.executemany(sql_Approve_1,tablesummary)
                MySQLDB.commit()
            except:
                MySQLDB.rollback()
                WriteLog("insert error")
        else:
            WriteLog("同步审批单0条")
        try:
            val = ('','-1','','','','','','','','','','','','',str(sys.argv[2]),gCurrentDate)
            MySQLCR.execute(sql_Approve_1,val)
        except MySQLdb.Error as e:
            WriteLog('insert error!{}'.format(e))


def getJYSZQApprove(res):
    tradeDate = str(res[0])[0:4]+str(res[0])[5:7]+str(res[0])[8:10]  #交易日期
    tradeTime = str(res[1])[11:-1]                                   #交易时间
    orderID         = str(res[2])          #交易编号/审批单号
    status          = str(res[3])          #交易状态/审批单状态
    direction       = str(res[4])          #交易方向/交易类型
    htAccountID     = str(res[5])          #内证
    issueCode       = str(res[6])          #债券代码
    if res[13]=='XSHG':
        issueCode       = 'SH'+str(res[6])     #债券代码
    elif res[13]=='XSHE':
        issueCode       = 'SZ'+str(res[6])     #债券代码
    tradequantity   = str(res[8]*100)              #交易数量
    tradeprice      = round(res[7]*1000)/1000  #交易价格
    investorID      = str(res[12])             #衡泰操作员
    askprice        = ''
    bidprice        = ''
    bidQuantity     = ''
    askQuantity     = ''
    clearspeed      = str(res[14])

    if direction == '10':
        bidQuantity = tradequantity
        bidprice    = tradeprice
    if direction == '20':
        askQuantity = tradequantity
        askprice    = tradeprice
    
    return (issueCode,orderID,tradeDate,tradeTime,status,htAccountID,clearspeed,bidprice,bidQuantity,'',askprice,askQuantity,'',investorID,str(sys.argv[2]),gCurrentDate)



#################################################初始化#################################################
global file

    
#先读取配置文件
configFile = open(sys.argv[1],'r')
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
if path[-1:] == '/':
    path = path[:-1]
filePath = "%s/Sync_ApproveOrder_%s.log" % (path, gCurrentDate)

file = open(filePath, 'a+')

WriteLog('gCurrentDate:'+gCurrentDate)

#设置Oracle的字符集
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

global OracleDB
global OracleCR
global MySQLDB
global MySQLCR
#连接Oracle数据库
OracleDB = cx_Oracle.connect('%s/%s@%s:%s/%s' % (GlobalSettingTable['Oracle_User'],GlobalSettingTable['Oracle_PassWord'],GlobalSettingTable['Oracle_IP'],GlobalSettingTable['Oracle_Port'],GlobalSettingTable['Oracle_DB']))
OracleCR = OracleDB.cursor()

#连接MySQL数据库
#MySQLDB = MySQLdb.connect(host=GlobalSettingTable['MySQL_IP'],user=GlobalSettingTable['MySQL_User'],passwd=GlobalSettingTable['MySQL_PassWord'],database=GlobalSettingTable['MySQL_DB'],charset='latin1')
MySQLDB = MySQLdb.connect(GlobalSettingTable['MySQL_IP'], GlobalSettingTable['MySQL_User'], GlobalSettingTable['MySQL_PassWord'], GlobalSettingTable['MySQL_DB'], charset='latin1')
MySQLCR = MySQLDB.cursor()


#################################################初始化结束#################################################

#################################################处理开始#################################################


WriteLog("Sync START...")
SyncApproveOrder()
WriteLog('Sync END...')

#################################################处理结束#################################################

#关闭连接
OracleCR.close()
OracleDB.close()
MySQLCR.close()
MySQLDB.close()

file.close()
