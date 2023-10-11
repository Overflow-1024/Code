# coding=utf-8
from Global import context
import Utils as util
from Model.BondInfo import BondInfo

bondinfo = BondInfo()

# 银行间债券基础信息
def sync():

    global bondinfo

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    fields = ','.join(bondinfo.fieldSource)
    sql = "select {fields} from {db}.{table} where to_char(sysdate,'yyyy-MM-dd') <= MATURITY_DATE" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CFETS_B_BOND')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步债券数据0条")
            break
        else:
            count += 1
            util.WriteLog('fetch BondInfo ' + str(len(res)) + ' row')

            bondinfo.setData(res, args={'CurrentDate': curdate,
                                        'PreTradingDate': predate})
            bondinfo.setDefaultValue()

            # 插入BondInfoTable
            fields, placeholder, values = bondinfo.generateDataSQL(bondinfo.dataBondInfo,
                                                                   bondinfo.fieldBondInfo)
            bondInfoSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondInfoTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondInfoSQL, values):
                for row in values:
                    util.WriteLog(bondInfoSQL % row)
            else:
                util.WriteErrorLog('ERRData-bondInfoSQL')

            # 插入HisBondInfoTable
            fields, placeholder, values = bondinfo.generateDataSQL(bondinfo.dataHisBondInfo,
                                                                   bondinfo.fieldHisBondInfo)
            hisBondInfoSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HisBondInfoTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(hisBondInfoSQL, values):
                for row in values:
                    util.WriteLog(hisBondInfoSQL % row)
            else:
                util.WriteErrorLog('ERRData-hisBondInfoSQL')

            # 插入IssueMasterTable
            fields, placeholder, values = bondinfo.generateDataSQL(bondinfo.dataIssueMaster,
                                                                   bondinfo.fieldIssueMaster)

            issueMasterSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='IssueMasterTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(issueMasterSQL, values):
                for row in values:
                    util.WriteLog(issueMasterSQL % row)
            else:
                util.WriteErrorLog('ERRData-issueMasterSQL')

            # 插入IssueMarketTable
            fields, placeholder, values = bondinfo.generateDataSQL(bondinfo.dataIssueMarket,
                                                                   bondinfo.fieldIssueMarket)

            issueMarketSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='IssueMarketTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(issueMarketSQL, values):
                for row in values:
                    util.WriteLog(issueMarketSQL % row)
            else:
                util.WriteErrorLog('ERRData-issueMarketSQL')

            # 插入HistoricalPriceTable
            fields, placeholder, values = bondinfo.generateDataSQL(bondinfo.dataHistoricalPrice,
                                                                   bondinfo.fieldHistoricalPrice)
            historicalPriceSQL = "INSERT IGNORE INTO {db}.{table} ({fields}) values ({placeholder})" \
                .format(db=context.mysql_db, table='HistoricalPriceTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(historicalPriceSQL, values):
                for row in values:
                    util.WriteLog(historicalPriceSQL % row)
            else:
                util.WriteErrorLog("ERRData-historicalPriceSQL")

    print("SyncBondInfo finish")

def check():

    global bondinfo

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondinfo.generateCheckSQL()

    # BondInfoTable_CFETS 和 HisBondInfoTable_CFETS 对比
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE MarketCode='9' " \
               "AND ListingDate < '{predate}' AND DelistingDate > '{curdate}'"\
        .format(keys=keys_str, db=context.mysql_db, table='BondInfoTable_CFETS', predate=predate, curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondInfo')

    bond_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE MarketCode='9' AND DataDate={predate} " \
               "AND ListingDate < '{predate}' AND DelistingDate > '{curdate}'"\
        .format(keys=keys_str, db=context.mysql_db, table='HisBondInfoTable_CFETS', predate=predate, curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondInfo')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondInfoTable_CFETS", list(set_miss), bondinfo.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondInfoTable_CFETS", list(set_extra), bondinfo.fieldKeys)
        util.WriteErrorLog(msg)

    # 不同的
    checkSQL = "SELECT {fields_double}" \
               " FROM {db}.{table1} a INNER JOIN {db}.{table2} b ON {join_cond}" \
               " WHERE a.MarketCode='9' AND b.MarketCode='9' AND b.DataDate={predate}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, db=context.mysql_db, table1='BondInfoTable_CFETS', table2='HisBondInfoTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondInfo')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in BondInfoTable_CFETS", diff, fields_double,
                                   flag=True, pos=(len(bondinfo.fieldKeys), len(bondinfo.fieldCheck)))
        util.WriteErrorLog(msg)

    print("CheckBondInfo finish")
