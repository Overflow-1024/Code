# coding=utf-8
from Global import context
import Utils as util
from Model.BondInfoXBond import BondInfoXBond

bondinfo_xbond = BondInfoXBond()

# X-bond可交易债券
def sync():

    global bondinfo_xbond

    curdate = context.gCurrentDate

    fields = ','.join(bondinfo_xbond.fieldSource)
    sql = "SELECT {fields} FROM {db}.{table} " \
          "where substr(updatetime,1,10) = to_char(sysdate,'yyyy-MM-dd')" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='ttrd_cfets_b_xbondinfo')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步XBOND可交易债0条")
            break
        else:
            count += 1
            util.WriteLog('fetch BondInfo_XBond ' + str(len(res)) + ' row')

            bondinfo_xbond.setData(res, args={'CurrentDate': curdate})
            bondinfo_xbond.setDefaultValue()

            fields, placeholder, values = bondinfo_xbond.generateDataSQL(bondinfo_xbond.dataBondInfoXBond,
                                                                         bondinfo_xbond.fieldBondInfoXBond)
            avlTradeBondSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='BondInfo_XBond_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(avlTradeBondSQL, values):
                for row in values:
                    util.WriteLog(avlTradeBondSQL % row)
            else:
                util.WriteErrorLog("ERRData-avlTradeBondSQL")

    print("SyncAvlTradeBondInfo finish")

def check():

    global bondinfo_xbond

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondinfo_xbond.generateCheckSQL()

    # BondInfo_XBond_CFETS 和历史数据对比
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE DataDate='{curdate}'"\
        .format(db=context.mysql_db, keys=keys_str, table='BondInfo_XBond_CFETS', curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncAvlTradeBondInfo')

    bond_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE DataDate={predate}"\
        .format(db=context.mysql_db, keys=keys_str, table='BondInfo_XBond_CFETS', predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncAvlTradeBondInfo')

    bond_ref = context.mysql.fetchall()

    set_test = set(bond_test)
    set_ref = set(bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in BondInfo_XBond_CFETS", list(set_miss), bondinfo_xbond.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in BondInfo_XBond_CFETS", list(set_extra), bondinfo_xbond.fieldKeys)
        util.WriteErrorLog(msg)

    print("CheckAvlTradeBondInfo finish")