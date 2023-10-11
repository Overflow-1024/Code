# -*- coding: utf-8 -*-
from __future__ import division
from Global import context
import Utils as util
import datetime
import sys
import time

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

def printErrorMessage(title, values, fields, flag=False, pos=None):
    # values：list(tuple) 或者 tuple(tuple)
    # pos[0]:keys数量  pos[1]:fields数量  len(fields) = pos[0] + 2 * pos[1]

    print(title + ": ")
    util.WriteErrorLog(title + ": ")

    if flag:
        # 先验证字段数量能否对上号，不然下面逻辑会出错
        if len(fields) != pos[0] + 2 * pos[1]:
            print("invalid fields length in printErrorMessage()")

        for value in values:

            strList = []
            # 处理keys
            for i in range(0, pos[0]):
                strList.append(fields[i] + '[' + str(value[i]).encode('latin1').decode('utf8') + ']')
            # 处理比较字段
            a_start = pos[0]
            b_start = a_start + pos[1]
            for j in range(0, pos[1]):
                if value[a_start + j] != value[b_start + j]:
                    strList.append(fields[a_start + j] + '[' + str(value[a_start + j]).encode('latin1').decode('utf8') + ']')
                    strList.append(fields[b_start + j] + '[' + str(value[b_start + j]).encode('latin1').decode('utf8') + ']')

            print(title + ": " + ', '.join(strList))
            util.WriteErrorLog(title + ": " + ', '.join(strList))

    else:
        for value in values:
            strList = []
            for i in range(len(fields)):
                strList.append(fields[i] + '[' + str(value[i]).encode('latin1').decode('utf8') + ']')

            print(title + ": " + ', '.join(strList))
            util.WriteErrorLog(title + ": " + ', '.join(strList))

class TableStructure(object):

    def __init__(self):
        self.BondInfoTable_CFETS = dict()
        self.BondInfoTable_CFETS['keys'] = ['IssueCode', 'MarketCode']
        self.BondInfoTable_CFETS['check'] = [
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'BondTypeID',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'DurationString',
            'AI',
            'CR',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FirstPaymentDate',
            'FixedCouponRate',
            'ExerciseType1',
            'SettlCurrency',
            'CustodianName',
            'IssuerShortPartyID',
            'IssuerPrice',
            'TheoryPrice',
            'ClearingSpeed'
        ]

        self.HisBondInfoTable_CFETS = dict()
        self.HisBondInfoTable_CFETS['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HisBondInfoTable_CFETS['check'] = [
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'BondTypeID',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'DurationString',
            'AI',
            'CR',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FirstPaymentDate',
            'FixedCouponRate',
            'ExerciseType1',
            'SettlCurrency',
            'CustodianName',
            'IssuerShortPartyID',
            'IssuerPrice'
        ]

        self.IssueMasterTable = dict()
        self.IssueMasterTable['keys'] = ['IssueCode']
        self.IssueMasterTable['check'] = [
            'IssueShortName',
            'IssueShortLocalName',
            'IssueLongName',
            'IssueLongLocalName',
            'ProductCode',
            'Currency',
            'MarketSectorCode',
            'PriorMarket',
            'UnderlyingAssetCode',
            'PutCall',
            'ContractMonth',
            'OtherContractMonth',
            'StrikePrice',
            'ExpirationDate',
            'FaceValue',
            'EstimateFaceValue',
            'NearMonthIssueCode',
            'OtherMonthIssueCode',
            'GrantedRatio',
            'UnderlyingIssueCode',
            'ExRightsDate',
            'ReserveString',
            'DTSTimeStamp',
            'Shares',
            'Status',
            'UnitAmount',
            'AmountLeast',
            'AmountMost',
            'BalanceMost',
            'Tick',
            'RaisingLimitRate',
            'DecliningLimitRate',
            'FareRule',
            'PromptRule',
            'TradingMonth',
            'OpenDropFareRatio',
            'OpenDropFareBalance',
            'DropCuFareRatio',
            'DropCuFareBalance',
            'DeliverFareRatio',
            'DeliverFareBalance',
            'SpeculationBailRatio',
            'SpeculationBailBalance',
            'HedgeBailRatio',
            'HedgeBailBalance',
            'ContractSize'
        ]

        self.IssueMarketTable = dict()
        self.IssueMarketTable['keys'] = ['IssueCode', 'MarketCode']
        self.IssueMarketTable['check'] = [
            'MarketSystemCode',
            'EvenLot',
            'CashDeliveryFlag',
            'LoanableIssueFlag',
            'MarginIssueFlag',
            'ListedDate',
            'MarketSectionCode',
            'SessionPatternID',
            'ReserveString',
            'DTSTimeStamp',
            'ApplicationStopFlag'
        ]

        self.HistoricalPriceTable = dict()
        self.HistoricalPriceTable['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HistoricalPriceTable['check'] = [
            'MarkPrice',
            'ClosePrice',
            'AdjustedClosePrice',
            'OpenPrice',
            'HighPrice',
            'LowPrice',
            'Volume',
            'UpperLimitPrice',
            'LowerLimitPrice',
            'MMLNBestBid',
            'ReserveString',
            'DTSTimeStamp',
            'WeeklyHighPrice',
            'WeeklyLowPrice',
            'MonthlyHighPrice',
            'MonthlyLowPrice',
            'QuarterHighPrice',
            'QuarterLowPrice',
            'Psychological',
            'WeightGiftCounter',
            'WeightSellCounter',
            'WeightSellPrice',
            'WeightDividend',
            'WeightIncCounter',
            'WeightOwnerShip',
            'WeightFreeCounter',
            'ClearingPrice',
            'MMBestBid1'
        ]

        self.HistoricalPriceTable_CFETS = dict()
        self.HistoricalPriceTable_CFETS['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HistoricalPriceTable_CFETS['check'] = [
            'BasicPrice',
            'ClearingPrice'
        ]

        self.BondEvalTable_CFETS = dict()
        self.BondEvalTable_CFETS['keys'] = ['IssueCode', 'DataDate']
        self.BondEvalTable_CFETS['check'] = [
            'Yield',
            'NetPrice',
            'FullPrice'
        ]

        self.BondInfo_XBond_CFETS = dict()
        self.BondInfo_XBond_CFETS['keys'] = ['IssueCode', 'DataDate']
        self.BondInfo_XBond_CFETS['check'] = [
            'IssueName',
            'BondType',
            'BondTypeID',
            'CenterQuote',
            'IssuerShortPartyID',
            'DurationString',
            'AutoSubscribe'
        ]

        self.BondCoversionTable_CFETS = dict()
        self.BondCoversionTable_CFETS['keys'] = ['IssueCode', 'MarketCode']
        self.BondCoversionTable_CFETS['check'] = [
            'IssueName',
            'ConvDate',
            'ConvCode',
            'ConvPrice',
            'BondType',
            'BondExtredType'
        ]
        self.HisBondCoversionTable_CFETS = dict()
        self.HisBondCoversionTable_CFETS['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HisBondCoversionTable_CFETS['check'] = [
            'IssueName',
            'ConvDate',
            'ConvCode',
            'ConvPrice',
            'BondType',
            'BondExtredType'
        ]

        self.BondCodeInMarket_CFETS = dict()
        self.BondCodeInMarket_CFETS['keys'] = ['IssueCode', 'MarketCode']
        self.BondCodeInMarket_CFETS['check'] = [
            'SH_Code',
            'SZ_Code',
            'YH_Code',
            'AssetType'
        ]
        self.HisBondCodeInMarket_CFETS = dict()
        self.HisBondCodeInMarket_CFETS['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HisBondCodeInMarket_CFETS['check'] = [
            'SH_Code',
            'SZ_Code',
            'YH_Code',
            'AssetType'
        ]
        self.IssueMasterTable_CFETS = dict()
        self.IssueMasterTable_CFETS['keys'] = ['IssueCode', 'MarketCode']
        self.IssueMasterTable_CFETS['check'] = [
            'ReportCode',
            'IssueName',
            'ProductCode',
            'UnderlyingIssueCode',
            'FaceValue',
            'MinQuantity',
            'MaxQuantity',
            'ListDate',
            'ExpirationDate',
            'ExpirationTime',
            'SettleDate',
            'Tick',
            'ContractSize',
            'Status',
            'ClearingMethod',
            'BenchMarkPrice'
        ]
        self.HisIssueMasterTable_CFETS = dict()
        self.HisIssueMasterTable_CFETS['keys'] = ['IssueCode', 'MarketCode', 'DataDate']
        self.HisIssueMasterTable_CFETS['check'] = [
            'ReportCode',
            'IssueName',
            'ProductCode',
            'UnderlyingIssueCode',
            'FaceValue',
            'MinQuantity',
            'MaxQuantity',
            'ListDate',
            'ExpirationDate',
            'ExpirationTime',
            'SettleDate',
            'Tick',
            'ContractSize',
            'Status',
            'ClearingMethod',
            'BenchMarkPrice'
        ]
        self.IssueDeliveryTable_CFETS = dict()
        self.IssueDeliveryTable_CFETS['keys'] = ['IssueCode', 'BondCode', 'UpdateDate']
        self.IssueDeliveryTable_CFETS['check'] = [
            'BondName',
            'StandardBond',
            'CF'
        ]
        self.TradeMemberTable_CFETS = dict()
        self.TradeMemberTable_CFETS['keys'] = ['MEMBER_ID']
        self.TradeMemberTable_CFETS['check'] = [
            'ORGCODE',
            'CH_NAME',
            'CH_SHORT_NAME',
            'EN_NAME',
            'EN_SHORT_NAME',
            'KIND_CODE',
            'KIND_NAME'
        ]
        self.HT_PositionTable_CFETS = dict()
        self.HT_PositionTable_CFETS['keys'] = ['IssueCode', 'MarketCode', 'HTAccountCode', 'LS', 'DataDate']
        self.HT_PositionTable_CFETS['check'] = [
            'IssueName',
            'AssetType',
            'PositionAmount',
            'BondExtendType',
            'PenetrateIssuer'
        ]

        self.HisTradeMemberTable_CFETS = dict()
        self.HisTradeMemberTable_CFETS['keys'] = ['MEMBER_ID', 'DataDate']
        self.HisTradeMemberTable_CFETS['check'] = [
            'ORGCODE',
            'CH_NAME',
            'CH_SHORT_NAME',
            'EN_NAME',
            'EN_SHORT_NAME',
            'KIND_CODE',
            'KIND_NAME'
        ]
        self.CalendarTable_CFETS = dict()
        self.CalendarTable_CFETS['keys'] = ['DTSDate', 'MarketCode']
        self.CalendarTable_CFETS['check'] = [
            'BondTrade',
            'BondSettle'
        ]
        self.HisCalendarTable_CFETS = dict()
        self.HisCalendarTable_CFETS['keys'] = ['DTSDate', 'MarketCode', 'DataDate']
        self.HisCalendarTable_CFETS['check'] = [
            'BondTrade',
            'BondSettle'
        ]

    def generateCheckSQL(self, table):
        # 给字段加上前缀 a b （用于两表联查）
        # a 作为当天 b 作为昨天
        keys_a = ["a." + item for item in table['keys']]
        keys_b = ["b." + item for item in table['keys']]
        fields_a = ["a." + item for item in table['check']]
        fields_b = ["b." + item for item in table['check']]
        fields_double = keys_a + fields_a + fields_b

        keys_str = ','.join(table['keys'])
        fields_str = ','.join(table['check'])
        fields_double_str = ','.join(fields_double)

        # 生成sql字符串
        join_cond_list = []
        for i in range(len(table['keys'])):
            cond = keys_a[i] + '=' + keys_b[i]
            join_cond_list.append(cond)
        join_condition = ' AND '.join(join_cond_list)

        check_cond_list = []
        for i in range(len(table['check'])):
            cond = fields_a[i] + '=' + fields_b[i]
            check_cond_list.append(cond)
        check_condition = ' AND '.join(check_cond_list)

        return keys_str, fields_str, fields_double, fields_double_str, join_condition, check_condition


# 清空相关的表
def ClearTables(db):

    for table in TableList:
        if table == 'CalendarTable_CFETS' or table == 'HisCalendarTable_CFETS':
            today = datetime.datetime.today()
            curdate = today.strftime('%Y%m%d')
            sql = "DELETE FROM {db}.{table} WHERE DTSDate >= '{curdate}'".format(db=db, table=table, curdate=curdate)

        elif table == 'IssueMasterTable':
            sql = "DELETE FROM {db}.{table} " \
                  "WHERE (PriorMarket = '9' AND ProductCode != '39') OR ProductCode = '37' OR " \
                  "(ProductCode = '11' AND (IssueCode LIKE 'SH%' OR IssueCode LIKE 'SZ%'))" \
                .format(db=db, table=table)

        elif table == 'IssueMarketTable':
            sql = "DELETE FROM {db}.{table} " \
                  "WHERE IssueCode IN (SELECT IssueCode FROM {db}.IssueMasterTable t " \
                  "WHERE (t.PriorMarket = '9' AND t.ProductCode != '39') OR t.ProductCode = '37' " \
                  "OR (t.ProductCode = '11' AND (t.IssueCode LIKE 'SH%' OR t.IssueCode LIKE 'SZ%'))" \
                  ")"\
                .format(db=db, table=table)

        else:
            sql = "DELETE FROM {db}.{table}".format(db=db, table=table)

        context.mysql.delete(sql)
        util.WriteLog(sql)

# 比对插入结果
def CheckResult():

    max_num = 30    # 控制输出的数量（太多不方便看）

    def CheckOneTable(table_name):

        print("Check " + table_name + " ...")
        util.WriteErrorLog("Check " + table_name + " ...")

        tb = getattr(tbs, table_name)

        keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = tbs.generateCheckSQL(tb)

        if table_name == 'IssueMasterTable':
            checkSQL = "SELECT {keys} FROM {db}.{table} " \
                       "WHERE (PriorMarket = '9' AND ProductCode != '39') OR ProductCode = '37' OR " \
                       "(ProductCode = '11' AND (IssueCode LIKE 'SH%' OR IssueCode LIKE 'SZ%'))"\
                .format(keys=keys_str, db=dtsdb, table=table_name)
        elif table_name == 'IssueMarketTable':
            checkSQL = "SELECT {keys} FROM {db}.{table} " \
                       "WHERE IssueCode IN (SELECT IssueCode FROM {db}.IssueMasterTable t " \
                       "WHERE (t.PriorMarket = '9' AND t.ProductCode != '39') OR t.ProductCode = '37' " \
                       "OR (t.ProductCode = '11' AND (t.IssueCode LIKE 'SH%' OR t.IssueCode LIKE 'SZ%'))" \
                       ")"\
                .format(keys=keys_str, db=dtsdb, table=table_name)
        else:
            checkSQL = "SELECT {keys} FROM {db}.{table}"\
                .format(keys=keys_str, db=dtsdb, table=table_name)

        if not context.mysql.query(checkSQL):
            print('ERR-CheckSQL')
            util.WriteErrorLog('ERR-CheckSQL')
        keys_ref = context.mysql.fetchall()

        if table_name == 'IssueMasterTable':
            checkSQL = "SELECT {keys} FROM {db}.{table} " \
                       "WHERE (PriorMarket = '9' AND ProductCode != '39') OR ProductCode = '37' OR " \
                       "(ProductCode = '11' AND (IssueCode LIKE 'SH%' OR IssueCode LIKE 'SZ%'))" \
                .format(keys=keys_str, db=testdb, table=table_name)
        elif table_name == 'IssueMarketTable':
            checkSQL = "SELECT {keys} FROM {db}.{table} " \
                       "WHERE IssueCode IN (SELECT IssueCode FROM {db}.IssueMasterTable t " \
                       "WHERE (t.PriorMarket = '9' AND t.ProductCode != '39') OR t.ProductCode = '37' " \
                       "OR (t.ProductCode = '11' AND (t.IssueCode LIKE 'SH%' OR t.IssueCode LIKE 'SZ%'))" \
                       ")" \
                .format(keys=keys_str, db=testdb, table=table_name)
        else:
            checkSQL = "SELECT {keys} FROM {db}.{table}" \
                .format(keys=keys_str, db=testdb, table=table_name)

        if not context.mysql.query(checkSQL):
            print('ERR-CheckSQL')
            util.WriteErrorLog('ERR-CheckSQL')
        keys_test = context.mysql.fetchall()

        set_test = set(keys_test)
        set_ref = set(keys_ref)

        # 缺少的
        set_miss = set_ref - set_test
        if set_miss:
            print("Miss: ", set_miss)
            util.WriteErrorLog("Miss: " + str(set_miss))

        # 多出的
        set_extra = set_test - set_ref
        if set_extra:
            print("Extra: ", set_extra)
            util.WriteErrorLog("Extra: " + str(set_extra))

        checkSQL = "SELECT {fields_double}" \
                   " FROM {db}.{table} a INNER JOIN {db1}.{table} b ON {join_cond}" \
                   " WHERE NOT ({check_cond})" \
            .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                    db=dtsdb, db1=testdb, table=table_name)

        if not context.mysql.query(checkSQL):
            print('ERR-CheckSQL')
            util.WriteErrorLog('ERR-CheckSQL')
        diff = context.mysql.fetchall()

        if diff:
            if len(diff) > max_num:
                diff = diff[:max_num]
            print("Diff: ")
            util.WriteErrorLog("Diff: ")
            printErrorMessage("ERRData-diff in {table}".format(table=table_name), diff, fields_double,
                              flag=True, pos=(len(tb['keys']), len(tb['check'])))

    for table in TableList:
        CheckOneTable(table)


TableList = [
    'BondInfoTable_CFETS',
    'HisBondInfoTable_CFETS',
    'IssueMasterTable',
    'IssueMarketTable',

    'HistoricalPriceTable',
    'HistoricalPriceTable_CFETS',

    'BondEvalTable_CFETS',
    'BondInfo_XBond_CFETS',
    'BondCoversionTable_CFETS',
    'BondCodeInMarket_CFETS',


    'IssueMasterTable_CFETS',
    'IssueDeliveryTable_CFETS',
    'TradeMemberTable_CFETS',
    'CalendarTable_CFETS',

    'HT_PositionTable_CFETS',

    # 'HisBondCoversionTable_CFETS',
    # 'HisBondCodeInMarket_CFETS',
    # 'HisIssueMasterTable_CFETS',
    # 'HisTradeMemberTable_CFETS',
    # 'HisCalendarTable_CFETS'
]

if __name__ == "__main__":

    tbs = TableStructure()
    dtsdb = 'dtsdb'
    testdb = 'test1'

    # ClearTables('test1')

    CheckResult()


