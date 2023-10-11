# coding=utf-8
import datetime

from Global import context
import Utils as util
import Calculate as calc

from Model.CCPClearPrice import CCPClearPrice


ccpprice = CCPClearPrice()

def sync():

    global ccpprice

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)
    updatetime = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]

    # 从CMDS表中获取结算价到HistoricalPriceTable
    sql = "SELECT securityid, settledprice FROM {db}.{table} " \
          "where substr(updatetime,1,10) = '{updatetime}'"\
        .format(db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CMDS_SBFWD_SETTLEDPRICE',
                updatetime=updatetime)

    util.WriteLog(sql)
    context.oracle.query(sql)

    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步CCP昨结算价, 从CMDS表中获取结算价0条")
            break

        count += 1

        ccpprice.setData(res, {'PreTradingDate': predate})
        ccpprice.setDefaultValue()

        values = ccpprice.getDataByField(ccpprice.dataHistoricalPrice, ccpprice.fieldHistoricalPrice)
        ccpClearPriceSQL = "update {db}.{table} " \
                           "set ClearingPrice = %s,TimeStamp = %s where DataDate = %s and IssueCode = %s"\
            .format(db=context.mysql_db, table='HistoricalPriceTable')

        if context.mysql.update(ccpClearPriceSQL, values):
            for row in values:
                util.WriteLog(ccpClearPriceSQL % row)
        else:
            util.WriteErrorLog("ERRData-ccpClearPriceSQL")

    # 从结算价表中获取结算价到HistoricalPriceTable_CFETS
    beg_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]

    sql = "SELECT i_code, dp_set FROM {db}.{table} " \
          "where (i_code like 'CDB%%' or i_code like 'ADBC%%') and beg_date = '{beg_date}'"\
        .format(db=context.GlobalSettingTable['Oracle_XIR_MD'], table='TNON_DAILYPRICE', beg_date=beg_date)

    util.WriteLog(sql)
    context.oracle.query(sql)

    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步CCP昨结算价, 从结算价表中获取结算价0条")
            break

        count += 1

        ccpprice.setData(res, {'PreTradingDate': predate})
        ccpprice.setDefaultValue()

        fields, placeholder, values = ccpprice.generateDataSQL(ccpprice.dataHistoricalPriceCFETS, ccpprice.fieldHistoricalPriceCFETS)
        ccpClearPriceSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', fields=fields, placeholder=placeholder)
        util.WriteLog(ccpClearPriceSQL)

        if context.mysql.update(ccpClearPriceSQL, values):
            for row in values:
                util.WriteLog(ccpClearPriceSQL % row)
        else:
            util.WriteErrorLog("ERRData-ccpClearPriceSQL")

    print("SyncCCPClearPrice finish")


def check():
    global ccpprice
    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    # 根据规则生成的合约代码（作为比对的标准）
    contract_ref, codelist = calc.generateCodeList(ccpprice, datetime.datetime.today(), type='CCP')

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = ccpprice.generateCheckSQL()

    # HistoricalPriceTable 和标准对比
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE DataDate='{predate}' AND IssueCode IN {codelist}"\
        .format(db=context.mysql_db, table='HistoricalPriceTable', predate=predate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPClearPrice')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in HistoricalPriceTable", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # HistoricalPriceTable_CFETS 和标准对比
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE DataDate={predate} AND IssueCode IN {codelist}" \
        .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', predate=predate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPClearPrice')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in HistoricalPriceTable_CFETS", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # HistoricalPriceTable_CFETS 和 HistoricalPriceTable 对比
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond} " \
               " WHERE a.IssueCode IN {codelist} AND a.DataDate={predate}" \
               " AND b.IssueCode IN {codelist} AND b.DataDate={predate}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, codelist=codelist,
                db=context.mysql_db, table1='HistoricalPriceTable_CFETS', table2='HistoricalPriceTable')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in HistoricalPriceTable_CFETS and HistoricalPriceTable", diff, fields_double)
        util.WriteErrorLog(msg)

    print("CheckCCPClearPrice finish")