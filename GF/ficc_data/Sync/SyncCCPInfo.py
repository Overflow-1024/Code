# coding=utf-8
import datetime

from Global import context
import Utils as util
import Calculate as calc

from Model.CCPInfo import CCPInfo
from Model.CCPDelivery import CCPDelivery
from Model.CCPFactor import CCPFactor


ccpinfo = CCPInfo()
ccpdl = CCPDelivery()
ccpcf = CCPFactor()

def sync():

    global ccpinfo
    global ccpdl
    global ccpcf

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    # 1.同步CCP基础信息
    fields = ','.join(ccpinfo.fieldSource)
    sql = "select {fields} from {db}.{table}"\
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CFETS_B_SBF')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步标债远期数据, CCP基础信息0条")
            break

        count += 1

        ccpinfo.setData(res, {'CurrentDate': curdate,
                              'PreTradingDate': predate})
        ccpinfo.setDefaultValue()

        # 插入IssueMasterTable_CFETS
        fields, placeholder, values = ccpinfo.generateDataSQL(ccpinfo.dataIssueMasterCFETS,
                                                              ccpinfo.fieldIssueMasterCFETS)
        ccpSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='IssueMasterTable_CFETS', fields=fields, placeholder=placeholder)

        if context.mysql.update(ccpSQL, values):
            for row in values:
                util.WriteLog(ccpSQL % row)
        else:
            util.WriteErrorLog("ERRData-ccpSQL")

        # 插入HisIssueMasterTable_CFETS
        fields, placeholder, values = ccpinfo.generateDataSQL(ccpinfo.dataHisIssueMasterCFETS,
                                                              ccpinfo.fieldHisIssueMasterCFETS)
        ccpSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='HisIssueMasterTable_CFETS', fields=fields, placeholder=placeholder)

        if context.mysql.update(ccpSQL, values):
            for row in values:
                util.WriteLog(ccpSQL % row)
        else:
            util.WriteErrorLog("ERRData-ccpSQL")

        # 插入IssueMasterTable
        fields, placeholder, values = ccpinfo.generateDataSQL(ccpinfo.dataIssueMaster,
                                                              ccpinfo.fieldIssueMaster)
        issueMasterSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='IssueMasterTable', fields=fields, placeholder=placeholder)

        if context.mysql.update(issueMasterSQL, values):
            for row in values:
                util.WriteLog(issueMasterSQL % row)
        else:
            util.WriteErrorLog("ERRData-issueMasterSQL")

        # 插入IssueMarketTable
        fields, placeholder, values = ccpinfo.generateDataSQL(ccpinfo.dataIssueMarket,
                                                              ccpinfo.fieldIssueMarket)
        issueMarketSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
            .format(db=context.mysql_db, table='IssueMarketTable', fields=fields, placeholder=placeholder)

        if context.mysql.update(issueMarketSQL, values):
            for row in values:
                util.WriteLog(issueMarketSQL % row)
        else:
            util.WriteErrorLog("ERRData-issueMarketSQL")

        # 插入HistoricalPriceTable
        fields, placeholder, values = ccpinfo.generateDataSQL(ccpinfo.dataHistoricalPrice,
                                                              ccpinfo.fieldHistoricalPrice)
        historicalPriceSQL = "INSERT IGNORE INTO {db}.{table} ({fields}) values ({placeholder})" \
            .format(db=context.mysql_db, table='HistoricalPriceTable', fields=fields, placeholder=placeholder)

        if context.mysql.update(historicalPriceSQL, values):
            for row in values:
                util.WriteLog(historicalPriceSQL % row)
        else:
            util.WriteErrorLog("ERRData-historicalPriceSQL")

    # 2.同步CCP的可交割券数据
    fields = ','.join(ccpdl.fieldSource)
    sql = "select {fields} from {db}.{table}"\
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CFETS_B_SBF_DELIVERABLE')

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步标债远期数据, CCP的可交割券数据0条")
            break

        count += 1

        ccpdl.setData(res, args={'CurrentDate': curdate})
        ccpdl.setDefaultValue()

        fields, placeholder, values = ccpdl.generateDataSQL(ccpdl.dataIssueDelivery,
                                                            ccpdl.fieldIssueDelivery)
        deliverySQL = "INSERT IGNORE INTO {db}.{table} ({fields}) values ({placeholder})"\
            .format(db=context.mysql_db, table='IssueDeliveryTable_CFETS', fields=fields, placeholder=placeholder)

        if context.mysql.update(deliverySQL, values):
            for row in values:
                util.WriteLog(deliverySQL % row)
        else:
            util.WriteErrorLog("ERRData-deliverySQL")

    # 4.更新实物交割券的转换因子
    fields = ','.join(ccpcf.fieldSource)
    sql = "select {fields} from {db}.{table} where a_type = 'FWD_BDS'"\
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table='tbnd_future_deliverbonds')

    util.WriteLog(sql)
    context.oracle.query(sql)
    count = 0

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步标债远期实物交割转换因子, 数据0条")
            break

        count += 1

        ccpcf.setData(res, args={'CurrentDate': curdate})
        ccpcf.setDefaultValue()

        values = ccpcf.getDataByField(ccpcf.dataIssueDelivery, ccpcf.fieldIssueDelivery)
        cfSQL = "update {db}.{table} set CF = %s " \
                "where UpdateDate = %s and IssueCode = %s and BondCode = %s"\
            .format(db=context.mysql_db, table='IssueDeliveryTable_CFETS')

        if context.mysql.update(cfSQL, values):
            for row in values:
                util.WriteLog(cfSQL % row)
        else:
            util.WriteErrorLog("ERRData-cfSQL")

    print("SyncCCPInfo finish")

def check():

    global ccpinfo
    global ccpdl
    global ccpcf

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    # 根据规则生成的合约代码（作为比对的标准）
    contract_ref, codelist = calc.generateCodeList(ccpinfo, datetime.datetime.today(), type='CCP')

    """
    校验标债远期合约基础信息
    """

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = ccpinfo.generateCheckSQL()

    # IssueMasterTable_CFETS 和标准合约代码列表对比
    checkSQL = "SELECT IssueCode FROM {db}.{table} WHERE ProductCode = '38' AND ExpirationDate >= '{curdate}'"\
        .format(db=context.mysql_db, table='IssueMasterTable_CFETS', curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in IssueMasterTable_CFETS", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in IssueMasterTable_CFETS", list(set_extra), ['IssueCode'])
        util.WriteErrorLog(msg)

    # IssueMasterTable_CFETS 和 HisIssueMasterTable_CFETS 对比
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond} " \
               " WHERE a.ProductCode = '38' AND a.ExpirationDate >= '{curdate}'" \
               " AND b.ProductCode = '38' AND b.ExpirationDate >= '{curdate}'" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                curdate=curdate, db=context.mysql_db, table1='IssueMasterTable_CFETS', table2='HisIssueMasterTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in IssueMasterTable_CFETS", diff, fields_double,
                                   flag=True, pos=(len(ccpinfo.fieldKeys), len(ccpinfo.fieldCheck)))
        util.WriteErrorLog(msg)

    """
    校验标债远期对应的交割券
    """
    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = ccpdl.generateCheckSQL()

    # IssueDeliveryTable_CFETS 和标准合约代码列表对比
    checkSQL = "SELECT DISTINCT IssueCode FROM {db}.{table} WHERE UpdateDate={curdate}"\
        .format(db=context.mysql_db, table='IssueDeliveryTable_CFETS', curdate=curdate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in IssueDeliveryTable_CFETS", list(set_miss), ['IssueCode'])
        util.WriteErrorLog(msg)

    # 检验 IssueDeliveryTable_CFETS 每个交割券对应的债券数量是否正确
    checkSQL = "SELECT IssueCode, COUNT(*) FROM {db}.{table}" \
               " WHERE Updatedate={curdate} AND IssueCode In {codelist} GROUP BY IssueCode"\
        .format(db=context.mysql_db, table='IssueDeliveryTable_CFETS', curdate=curdate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    contract_test = context.mysql.fetchall()

    for item in contract_test:
        # 后缀是P 实物交割合约
        if item[0][-1] == 'P':
            if item[1] < 2 or item[1] > 4:
                msg = util.getErrorMessage("ERRData-wrong count in IssueDeliveryTable_CFETS", [item], ['IssueCode', 'Count'])
                util.WriteLog(msg)
        # 否则是 现金交割合约
        else:
            if item[1] != 2:
                msg = util.getErrorMessage("ERRData-wrong count in IssueDeliveryTable_CFETS", [item], ['IssueCode', 'Count'])
                util.WriteLog(msg)

    # IssueDeliveryTable_CFETS 和自己历史数据对比
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE Updatedate={curdate} AND IssueCode IN {codelist}"\
        .format(keys=keys_str, db=context.mysql_db, table='IssueDeliveryTable_CFETS', curdate=curdate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    contract_bond_test = context.mysql.fetchall()

    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE Updatedate='{predate}' AND IssueCode IN {codelist}" \
        .format(keys=keys_str, db=context.mysql_db, table='IssueDeliveryTable_CFETS', predate=predate, codelist=codelist)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    contract_bond_ref = context.mysql.fetchall()

    set_test = set(contract_bond_test)
    set_ref = set(contract_bond_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in IssueDeliveryTable_CFETS", list(set_miss), ccpdl.fieldKeys)
        util.WriteErrorLog(msg)

    # 多出的
    set_extra = set_test - set_ref
    if set_extra:
        msg = util.getErrorMessage("ERRData-extra in IssueDeliveryTable_CFETS", list(set_extra), ccpdl.fieldKeys)
        util.WriteErrorLog(msg)

    """
    校验标债远期对应交割券的转换因子
    """
    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = ccpcf.generateCheckSQL()
    # IssueDeliveryTable_CFETS 和自己历史数据对比
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond} " \
               " WHERE a.Updatedate={curdate} AND b.Updatedate={predate}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                curdate=curdate, predate=predate,
                db=context.mysql_db, table1='IssueDeliveryTable_CFETS', table2='IssueDeliveryTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncCCPInfo')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in IssueDeliveryTable_CFETS", diff, fields_double)
        util.WriteErrorLog(msg)

    print("CheckCCPInfo finish")