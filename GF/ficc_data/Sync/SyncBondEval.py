# coding=utf-8
from Global import context
import Utils as util
from Model.BondEval import BondEval

bondeval = BondEval()

# 估值表
def sync():

    global bondeval

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    beg_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]
    fields = ','.join(bondeval.fieldSource)

    # 同步上清所估值
    sql = "select {fields} from {db}.{table} " \
          "where beg_date ='{beg_date}' and M_TYPE in ('X_CNBD','XSHE','XSHG') order by i_code" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table='TQSS_BOND_EVAL', beg_date=beg_date)

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步上清所估值0条")
            break
        else:
            count += 1
            util.WriteLog('fetch BondEval ' + str(len(res)) + ' row')

            bondeval.setData(res, args={'PreTradingDate': predate})
            bondeval.setDefaultValue()

            fields, placeholder, values = bondeval.generateDataSQL(bondeval.dataBondEval,
                                                                   bondeval.fieldBondEval)
            bondEvalSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondEvalTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondEvalSQL, values):
                for row in values:
                    util.WriteLog(bondEvalSQL % row)
            else:
                util.WriteErrorLog("ERRData-bondEvalSQL")

    # 同步中债估值
    fields = ','.join(bondeval.fieldSource)
    sql = "select {fields} from {db}.{table} " \
          "where beg_date ='{beg_date}' and M_TYPE in ('X_CNBD','XSHE','XSHG') order by i_code" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table='TCB_BOND_EVAL', beg_date=beg_date)

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步中债估值0条")
            break
        else:
            count += 1
            util.WriteLog('fetch BondEval ' + str(len(res)) + ' row')

            bondeval.setData(res, args={'PreTradingDate': predate})
            bondeval.setDefaultValue()

            fields, placeholder, values = bondeval.generateDataSQL(bondeval.dataBondEval,
                                                                   bondeval.fieldBondEval)
            bondEvalSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondEvalTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondEvalSQL, values):
                for row in values:
                    util.WriteLog(bondEvalSQL % row)
            else:
                util.WriteErrorLog("ERRData-bondEvalSQL")

    print("SyncBondEval finish")

def check(threshold):

    global bondeval

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)
    fordate = util.GetPreTradingDate(predate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondeval.generateCheckSQL()

    # BondEvalTable_CFETS 和 历史数据 对比 IssueCode
    checkSQL = "SELECT a.IssueCode FROM {db}.{table1} a INNER JOIN {db}.{table3} c" \
               " ON a.IssueCode = c.IssueCode" \
               " WHERE a.DataDate='{predate}' AND c.ListingDate < '{predate}'" \
        .format(keys=keys_str, db=context.mysql_db, table1='BondEvalTable_CFETS', table3='BondInfoTable_CFETS', predate=predate, fordate=fordate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondEval')

    bond_test = context.mysql.fetchall()

    # BondEvalTable_CFETS
    checkSQL = "SELECT b.IssueCode FROM {db}.{table2} b INNER JOIN {db}.{table3} c" \
               " ON b.IssueCode = c.IssueCode" \
               " WHERE b.DataDate='{fordate}' AND c.ListingDate < '{predate}'" \
        .format(keys=keys_str, db=context.mysql_db, table2='BondEvalTable_CFETS', table3='BondInfoTable_CFETS', predate=predate, fordate=fordate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondEval')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondEvalTable_CFETS", list(set_miss), bondeval.fieldKeys)
        # msg = "ERRData-miss in BondEvalTable_CFETS" + ": (" + str(len(list(set_miss))) + ")\n"
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondEvalTable_CFETS", list(set_extra), bondeval.fieldKeys)
        util.WriteErrorLog(msg)

    # 先取出不同的，再比较偏离度
    checkSQL = "SELECT {fields_double}" \
               " FROM {db}.{table1} a INNER JOIN {db}.{table2} b ON {join_cond}" \
               " WHERE a.DataDate='{predate}' AND b.DataDate='{fordate}'" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, fordate=fordate,
                db=context.mysql_db, table1='BondEvalTable_CFETS', table2='BondEvalTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondEval')

    diff = context.mysql.fetchall()

    abnormal = []
    if diff:
        for row in list(diff):
            if row[2] == 0 or abs(row[1] - row[2]) / row[2] > threshold:
                abnormal.append(row)
        if len(abnormal) > 0:
            msg = util.getErrorMessage("ERRData-abnormal in BondEvalTable_CFETS", abnormal, fields_double)
            util.WriteErrorLog(msg)

    print("CheckBondEval finish")
