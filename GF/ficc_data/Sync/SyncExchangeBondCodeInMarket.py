# coding=utf-8
from Global import context
import Utils as util
from Model.BondCodeInMarket import BondCodeInMarket

bondcode = BondCodeInMarket()


def sync():
    global bondcode

    curdate = context.gCurrentDate

    fields = ','.join(bondcode.fieldSource)
    sql = "select {fields} from {db}.{table} where B_MTR_DATE>= TO_CHAR(SYSDATE, 'YYYY-MM-DD')" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table='tbnd')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0
    # I_CODE,A_TYPE,M_TYPE,SH_CODE,SZ_CODE,YH_CODE
    # 0,     1,      2,    3        4      5

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步债券不同市场信息0条")
            break
        else:
            count += 1
            util.WriteLog('fetch ExchangeBondCodeInMarket ' + str(len(res)) + ' row')

            bondcode.setData(res, args={'CurrentDate': curdate})
            bondcode.setDefaultValue()

            fields, placeholder, values = bondcode.generateDataSQL(bondcode.dataBondCodeInMarket,
                                                                   bondcode.fieldBondCodeInMarket)
            bondcodeSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondCodeInMarket_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondcodeSQL, values):
                for row in values:
                    util.WriteLog(bondcodeSQL % row)
            else:
                util.WriteErrorLog("ERRData-bondCodeSQL")

            # 插入历史表
            fields, placeholder, values = bondcode.generateDataSQL(bondcode.dataHisBondCodeInMarket,
                                                                   bondcode.fieldHisBondCodeInMarket)
            hisBondCodeSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HisBondCodeInMarket_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(hisBondCodeSQL, values):
                for row in values:
                    util.WriteLog(hisBondCodeSQL % row)
            else:
                util.WriteErrorLog("ERRData-hisBondCodeSQL")

    print("SyncExchangeBondCodeInMarket finish")


def check():
    global bondcode

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondcode.generateCheckSQL()

    # BondCodeInMarket_CFETS 和 HisBondCodeInMarket_CFETS 对比
    checkSQL = "SELECT a.IssueCode, a.MarketCode FROM {db}.{table1} a INNER JOIN {db}.{table3} c" \
               " ON a.IssueCode=c.IssueCode AND a.MarketCode=c.MarketCode" \
               " WHERE c.ListingDate < '{curdate}'" \
        .format(keys=keys_str, db=context.mysql_db, table1='BondCodeInMarket_CFETS', table3='BondInfoTable_CFETS', curdate=curdate, predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondCodeInMarket')

    bond_test = context.mysql.fetchall()

    checkSQL = "SELECT b.IssueCode, b.MarketCode FROM {db}.{table2} b INNER JOIN {db}.{table3} c" \
               " ON b.IssueCode=c.IssueCode AND b.MarketCode=c.MarketCode" \
               " WHERE b.DataDate='{predate}' AND c.ListingDate < '{curdate}'" \
        .format(keys=keys_str, db=context.mysql_db, table2='HisBondCodeInMarket_CFETS', table3='BondInfoTable_CFETS', curdate=curdate, predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondCodeInMarket')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondCodeInMarket_CFETS", list(set_miss), bondcode.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondCodeInMarket_CFETS", list(set_extra), bondcode.fieldKeys)
        util.WriteErrorLog(msg)

    # 不同的
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a, {db}.{table2} b, {db}.{table3} c" \
               " WHERE b.DataDate={predate}" \
               " AND a.IssueCode=c.IssueCode AND a.MarketCode=c.MarketCode AND {join_cond}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, db=context.mysql_db,
                table1='BondCodeInMarket_CFETS', table2='HisBondCodeInMarket_CFETS', table3='BondInfoTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncExchangeBondCodeInMarket')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in BondCodeInMarket_CFETS", diff, fields_double,
                                   flag=True, pos=(len(bondcode.fieldKeys), len(bondcode.fieldCheck)))
        util.WriteErrorLog(msg)

    print("CheckExchangeBondCodeInMarket finish")
