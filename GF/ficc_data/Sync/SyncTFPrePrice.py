# coding=utf-8
import datetime

from Global import context
import Utils as util
import Calculate as calc

from Model.TFClearPrice import TFClearPrice

tfprice = TFClearPrice()

def sync():

    global tfprice

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDateExchange(curdate)

    beg_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]

    fields = ','.join(tfprice.fieldSource)
    # 获取基准价和结算价到HistoricalPriceTable_CFETS
    sql = "select {fields} from {db1}.{table1} p left join {db2}.{table2} q " \
          "on p.i_code = q.i_code and p.a_type = q.a_type and p.m_type = q.m_type and q.end_date = '2050-12-31' " \
          "where p.a_type = 'FUT_BD' and p.maturity_date >= to_char(sysdate, 'yyyy-mm-dd') and q.beg_date = '{beg_date}'"\
        .format(fields=fields, db1=context.GlobalSettingTable['Oracle_XIR_MD'], table1='tstk_idx_future',
                db2=context.GlobalSettingTable['Oracle_XIR_MD'], table2='tnon_dailyprice', beg_date=beg_date)

    context.oracle.query(sql)
    util.WriteLog(sql)

    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步国债期货昨结算价, 从结算价表中获取结算价0条")
            break

        count += 1
        util.WriteLog('fetch TFClearPrice ' + str(len(res)) + ' row')

        tfprice.setData(res, args={'PreTradingDate': predate})
        tfprice.setDefaultValue()

        fields, placeholder, values = tfprice.generateDataSQL(tfprice.dataHistoricalPriceCFETS,
                                                              tfprice.fieldHistoricalPriceCFETS)
        tfClearPriceSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', fields=fields, placeholder=placeholder)

        if context.mysql.update(tfClearPriceSQL, values):
            for row in values:
                util.WriteLog(tfClearPriceSQL % row)
        else:
            util.WriteErrorLog("ERRData-tfClearPriceSQL")

    print("SyncTFPrePrice finish")

def check():

    global tfprice

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)
    fordate = util.GetPreTradingDate(predate)

    # 自己根据规则生成合约代码，和IssueMasterTable 以及 HistoricalPriceTable_CFETS对比
    contract_ref, codelist = calc.generateCodeList(tfprice, datetime.datetime.today(), type='TF')

    # IssueMasterTable 和标准对比
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE ProductCode = '37' AND ExpirationDate >= '{curdate}'" \
        .format(db=context.mysql_db, table='IssueMasterTable', curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTFPrePrice')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in IssueMasterTable", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in IssueMasterTable", list(set_extra), ['IssueCode'])
        util.WriteErrorLog(msg)

    # HistoricalPriceTable_CFETS 和标准对比
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE DataDate='{predate}' AND IssueCode IN {codelist}" \
        .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', predate=predate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTFPrePrice')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in HistoricalPriceTable_CFETS", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # 基准价 HistoricalPriceTable_CFETS 和自己历史数据对比
    tfprice.setFieldCheck(['BasicPrice'])
    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = tfprice.generateCheckSQL()

    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond} " \
               " WHERE a.IssueCode IN {codelist} AND a.DataDate='{predate}'" \
               " AND b.IssueCode IN {codelist} AND b.DataDate='{fordate}'" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, fordate=fordate, codelist=codelist,
                db=context.mysql_db, table1='HistoricalPriceTable_CFETS', table2='HistoricalPriceTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTFPrePrice')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in HistoricalPriceTable_CFETS", diff, fields_double)
        util.WriteErrorLog(msg)

    # 结算价 HistoricalPriceTable 和 HistoricalPriceTable_CFETS 对比
    tfprice.setFieldCheck(['ClearingPrice'])
    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = tfprice.generateCheckSQL()

    # 筛掉上市首日的
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE IssueCode IN {codelist} AND ListedDate < '{curdate}'"\
        .format(db=context.mysql_db, table='IssueMarketTable', codelist=codelist, curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTFPrePrice')

    contract_filt = context.mysql.fetchall()
    codelist_filt = ['\'' + item[0] + '\'' for item in contract_filt]
    codelist_filt = '(' + ','.join(codelist_filt) + ')'

    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond} " \
               " WHERE a.IssueCode IN {codelist} AND a.DataDate={predate}" \
               " AND b.IssueCode IN {codelist} AND b.DataDate={predate}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, codelist=codelist_filt,
                db=context.mysql_db, table1='HistoricalPriceTable', table2='HistoricalPriceTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTFPrePrice')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in HistoricalPriceTable and HistoricalPriceTable_CFETS", diff, fields_double)
        util.WriteErrorLog(msg)

    print("CheckTFPrePrice finish")

