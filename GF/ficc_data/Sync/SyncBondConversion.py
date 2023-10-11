# coding=utf-8
from Global import context
import Utils as util
from Model.BondConversion import BondConversion

bondconv = BondConversion()

# 私募可交换债与正股关系
def sync():

    global bondconv

    curdate = context.gCurrentDate

    fields = ','.join(bondconv.fieldSource)
    sql = "SELECT {fields} " \
          "FROM {db1}.{table1} TB JOIN {db2}.{table2} BC " \
          "ON TB.I_CODE = BC.I_CODE AND TB.M_TYPE = BC.M_TYPE AND BC.END_DATE = '2050-12-31' " \
          "WHERE TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND TB.P_CLASS IN ('可交换债','可转换债')" \
        .format(fields=fields, db1=context.GlobalSettingTable['Oracle_XIR_MD'], table1='TBND',
                db2=context.GlobalSettingTable['Oracle_XIR_MD'], table2='GF_BONDCONVERSION')

    util.WriteLog(sql)
    context.oracle.query(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步私募可交换债与正股关系0条")
            break
        else:
            count += 1
            util.WriteLog('fetch BondConversion ' + str(len(res)) + ' row')

            bondconv.setData(res, args={'CurrentDate': curdate})
            bondconv.setDefaultValue()

            fields, placeholder, values = bondconv.generateDataSQL(bondconv.dataBondConversion,
                                                                   bondconv.fieldBondConversion)
            bondConversionSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondCoversionTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(bondConversionSQL, values):
                for row in values:
                    util.WriteLog(bondConversionSQL % row)
            else:
                util.WriteErrorLog("ERRData-bondConversionSQL")

            # 插入历史表
            fields, placeholder, values = bondconv.generateDataSQL(bondconv.dataHisBondConversion,
                                                                   bondconv.fieldHisBondConversion)
            hisBondConversionSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HisBondCoversionTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(hisBondConversionSQL, values):
                for row in values:
                    util.WriteLog(hisBondConversionSQL % row)
            else:
                util.WriteErrorLog("ERRData-hisBondConversionSQL")

    print("SyncBondConversion finish")

def check():

    global bondconv

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondconv.generateCheckSQL()

    # BondCoversionTable_CFETS 和 HisBondCoversionTable_CFETS 对比
    checkSQL = "SELECT {keys} FROM {db}.{table}"\
        .format(keys=keys_str, db=context.mysql_db, table='BondCoversionTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondConversion')

    bond_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE DataDate='{predate}'"\
        .format(keys=keys_str, db=context.mysql_db, table='HisBondCoversionTable_CFETS', predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondConversion')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondCoversionTable_CFETS", list(set_miss), bondconv.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondCoversionTable_CFETS", list(set_extra), bondconv.fieldKeys)
        util.WriteErrorLog(msg)

    # 不同的
    checkSQL = "SELECT {fields_double}" \
               " FROM {db}.{table1} a INNER JOIN {db}.{table2} b ON {join_cond}" \
               " WHERE b.DataDate='{predate}'" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, db=context.mysql_db,
                table1='BondCoversionTable_CFETS', table2='HisBondCoversionTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncBondConversion')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in BondCoversionTable_CFETS", diff, fields_double,
                                   flag=True, pos=(len(bondconv.fieldKeys), len(bondconv.fieldCheck)))
        util.WriteErrorLog(msg)

    print("CheckBondConversion finish")
