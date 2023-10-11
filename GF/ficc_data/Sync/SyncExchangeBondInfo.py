# coding=utf-8
from Global import context
import Utils as util
from Model.ExchangeBondInfo import ExchangeBondInfo

exbondinfo = ExchangeBondInfo()

# 交易所债券基础信息
def sync():

    global exbondinfo

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    fields = ','.join(exbondinfo.fieldSource)
    sql = "select {fields} from {db}.{table} WHERE M_TYPE='XSHG' and B_MTR_DATE>= TO_CHAR(SYSDATE, 'YYYY-MM-DD')" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table='TBND')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0
    # I_CODE,M_TYPE,B_NAME,WIND_CLASS1,B_PAR_VALUE,B_MTR_DATE,B_TERM,B_COUPON_TYPE,
    # 0,     1,     2,     3,          4,          5,         6,     7,
    # B_CASH_TIMES,B_DAYCOUNT,B_START_DATE,B_COUPON,CURRENCY,ISSUER_CODE,B_ISSUE_PRICE,B_LIST_DATE,B_DELIST_DATE,HOST_MARKET
    # 8,           9,         10,          11,      12,      13,         14            15上市日      16摘牌日       17托管市场

    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步债券不同市场信息0条")
            break
        else:
            count += 1
            util.WriteLog('fetch ExchangeBondInfo ' + str(len(res)) + ' row')

            exbondinfo.setData(res, args={'CurrentDate': curdate,
                                          'PreTradingDate': predate})
            exbondinfo.setDefaultValue()

            # 插入BondInfoTable
            fields, placeholder, values = exbondinfo.generateDataSQL(exbondinfo.dataBondInfo,
                                                                     exbondinfo.fieldBondInfo)
            bondInfoSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondInfoTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondInfoSQL, values):
                for row in values:
                    util.WriteLog(bondInfoSQL % row)
            else:
                util.WriteErrorLog('ERRData-bondInfoSQL')

            # 插入HisBondInfoTable
            fields, placeholder, values = exbondinfo.generateDataSQL(exbondinfo.dataHisBondInfo,
                                                                     exbondinfo.fieldHisBondInfo)

            hisBondInfoSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HisBondInfoTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(hisBondInfoSQL, values):
                for row in values:
                    util.WriteLog(hisBondInfoSQL % row)
            else:
                util.WriteErrorLog('ERRData-hisBondInfoSQL')

            # 插入IssueMasterTable
            fields, placeholder, values = exbondinfo.generateDataSQL(exbondinfo.dataIssueMaster,
                                                                     exbondinfo.fieldIssueMaster)

            issueMasterSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='IssueMasterTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(issueMasterSQL, values):
                for row in values:
                    util.WriteLog(issueMasterSQL % row)
            else:
                util.WriteErrorLog('ERRData-issueMasterSQL')

            # 插入IssueMarketTable
            fields, placeholder, values = exbondinfo.generateDataSQL(exbondinfo.dataIssueMarket,
                                                                     exbondinfo.fieldIssueMarket)

            issueMarketSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='IssueMarketTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(issueMarketSQL, values):
                for row in values:
                    util.WriteLog(issueMarketSQL % row)
            else:
                util.WriteErrorLog('ERRData-issueMarketSQL')

            # 插入HistoricalPriceTable
            fields, placeholder, values = exbondinfo.generateDataSQL(exbondinfo.dataHistoricalPrice,
                                                                     exbondinfo.fieldHistoricalPrice)

            historicalPriceSQL = "INSERT IGNORE INTO {db}.{table} ({fields}) values ({placeholder})" \
                .format(db=context.mysql_db, table='HistoricalPriceTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(historicalPriceSQL, values):
                for row in values:
                    util.WriteLog(historicalPriceSQL % row)
            else:
                util.WriteErrorLog("ERRData-historicalPriceSQL")

    print("SyncExchangeBondInfo finish")


def check():

    global exbondinfo

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = exbondinfo.generateCheckSQL()

    # BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 对比
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE MarketCode IN ('1','2') " \
               "AND ListingDate < '{curdate}' AND DelistingDate > '{curdate}'"\
        .format(keys=keys_str, db=context.mysql_db, table='BondInfoTable_CFETS', predate=predate, curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondInfo')

    bond_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE MarketCode IN ('1','2') AND DataDate={predate} " \
               "AND ListingDate < '{curdate}' AND DelistingDate > '{curdate}'"\
        .format(keys=keys_str, db=context.mysql_db, table='HisBondInfoTable_CFETS', predate=predate, curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondInfo')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondInfoTable_CFETS", list(set_miss), exbondinfo.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondInfoTable_CFETS", list(set_extra), exbondinfo.fieldKeys)
        util.WriteErrorLog(msg)

    # 不同的
    checkSQL = "SELECT {fields_double}" \
               " FROM {db}.{table1} a INNER JOIN {db}.{table2} b ON {join_cond}" \
               " WHERE a.MarketCode IN ('1','2') AND b.MarketCode IN ('1','2') AND b.DataDate={predate}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, db=context.mysql_db, table1='BondInfoTable_CFETS', table2='HisBondInfoTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondInfo')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in BondInfoTable_CFETS", diff, fields_double,
                                   flag=True, pos=(len(exbondinfo.fieldKeys), len(exbondinfo.fieldCheck)))
        util.WriteErrorLog(msg)

    print("CheckExchangeBondInfo finish")
