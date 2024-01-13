# coding=utf-8
from Global import context
import Utils as util
from Model.TradeMember import TradeMember


member = TradeMember()

def sync():

    global member

    curdate = context.gCurrentDate

    fields = ','.join(member.fieldSource)
    sql = "SELECT {fields} from {db1}.{table1} t" \
          " LEFT JOIN" \
          " (SELECT ti.T_CODE,ti.T_NAME,tc.BANKCODE,tc.ORGCODE" \
          " from {db2}.{table2} tc" \
          " LEFT JOIN {db3}.{table3} ti" \
          " ON tc.CLIENTKIND = ti.T_CODE WHERE tc.PARTY_STATUS = '1' and tc.BANKCODE IS NOT NULL) t1" \
          " ON (t1.ORGCODE = t.ORGCODE)"\
        .format(fields=fields, db1=context.GlobalSettingTable['Oracle_XIR_TRD'], table1='TTRD_CFETS_B_TRADE_MEMBER',
                db2=context.GlobalSettingTable['Oracle_XIR_TRD'], table2='TTRD_OTC_COUNTERPARTY',
                db3=context.GlobalSettingTable['Oracle_XIR_TRD'], table3='TTRD_OTC_INSTITUTIONTYPE')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步交易成员信息0条")
            break
        else:
            count += 1

            member.setData(res, args={'CurrentDate': curdate})
            member.setDefaultValue()

            fields, placeholder, values = member.generateDataSQL(member.dataTradeMember,
                                                                 member.fieldTradeMember)
            memberSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
                .format(db=context.mysql_db, table='TradeMemberTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(memberSQL, values):
                for row in values:
                    util.WriteLog(memberSQL % row)
            else:
                util.WriteErrorLog("ERRData-memberSQL")

            # 插入历史表
            fields, placeholder, values = member.generateDataSQL(member.dataHisTradeMember,
                                                                 member.fieldHisTradeMember)
            hisMemberSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HisTradeMemberTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(hisMemberSQL, values):
                for row in values:
                    util.WriteLog(hisMemberSQL % row)
            else:
                util.WriteErrorLog("ERRData-hisMemberSQL")

    print("SyncTradeMember finish")


def check():

    global member

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = member.generateCheckSQL()

    # TradeMemberTable_CFETS 和 HisTradeMemberTable_CFETS 对比
    checkSQL = "SELECT {keys} FROM {db}.{table}"\
        .format(db=context.mysql_db, keys=keys_str, table='TradeMemberTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTradeMember')

    member_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE DataDate='{predate}'"\
        .format(db=context.mysql_db, keys=keys_str, table='HisTradeMemberTable_CFETS', predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTradeMember')

    member_ref = context.mysql.fetchall()

    set_test = set(member_test)
    set_ref = set(member_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in TradeMemberTable_CFETS", list(set_miss), member.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in TradeMemberTable_CFETS", list(set_extra), member.fieldKeys)
        util.WriteErrorLog(msg)

    # 不同的
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond}" \
               " WHERE b.DataDate='{predate}'" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, db=context.mysql_db, table1='TradeMemberTable_CFETS', table2='HisTradeMemberTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTradeMember')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in TradeMemberTable_CFETS", diff, fields_double,
                                   flag=True, pos=(len(member.fieldKeys), len(member.fieldCheck)))
        util.WriteErrorLog(msg)

    print("CheckTradeMember finish")