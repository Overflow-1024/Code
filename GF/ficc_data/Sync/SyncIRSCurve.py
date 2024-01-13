# coding=utf-8
from Global import context
import Utils as util
from Model.IRSCurve import IRSCurveCMDS
from Model.IRSCurve import IRSCurveHT

icurve_cmds = IRSCurveCMDS()
icurve_ht = IRSCurveHT()


def sync():

    global icurve_cmds
    global icurve_ht

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    begin_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]
    beg_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]

    sql = "SELECT case a.SECURITY_ID" \
          " WHEN 'CISCD01M' THEN 'FDR001'" \
          " WHEN 'CISCSONM' THEN 'ShiborON'" \
          " WHEN 'CISCY1QM' THEN 'LPR1Y'" \
          " WHEN 'CISCY5QM' THEN 'LPR5Y'" \
          " WHEN 'CISCS3MM' THEN 'Shibor3M'" \
          " WHEN 'CISCD07M' THEN 'FDR007'" \
          " WHEN 'CISCF07M' THEN 'FR007'" \
          " else a.SECURITY_ID end," \
          "b.YIELD_RATE,a.SYMBOL," \
          "case to_number(b.YIELD_TERM)" \
          " WHEN 30 THEN '1M'" \
          " WHEN 90 THEN '3M'" \
          " WHEN 180 THEN '6M'" \
          " WHEN 270 THEN '9M'" \
          " WHEN 360 THEN '1Y'" \
          " WHEN 720 THEN '2Y'" \
          " WHEN 1080 THEN '3Y'" \
          " WHEN 1440 THEN '4Y'" \
          " WHEN 1800 THEN '5Y'" \
          " WHEN 2520 THEN '7Y'" \
          " WHEN 3600 THEN '10Y'" \
          " ELSE b.YIELD_TERM END," \
          "a.BEGIN_DATE,a.UPDATE_TIME" \
          " FROM {db1}.{table1} a JOIN {db2}.{table2} b" \
          " ON a.id = b.id WHERE a.BEGIN_DATE = '{begin_date}' AND a.SYMBOL LIKE '%%收盘%%均值%%'"\
        .format(db1=context.GlobalSettingTable['Oracle_XIR_TRD'], table1='TTRD_CMDS_IRS_STD_TM_EXP_CURVE',
                db2=context.GlobalSettingTable['Oracle_XIR_TRD'], table2='TTRD_CMDS_IRS_STD_TM_EXP_CV_ET',
                begin_date=begin_date)

    context.oracle.query(sql)
    util.WriteLog(sql)

    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("没有查到IRS利率曲线")
            break
        else:
            count += 1
            util.WriteLog('fetch IRSCurve ' + str(len(res)) + ' row')

            icurve_cmds.setData(res, args={'PreTradingDate': predate})
            icurve_cmds.setDefaultValue()

            fields, placeholder, values = icurve_cmds.generateDataSQL(icurve_cmds.dataHistoricalPriceCFETS,
                                                                      icurve_cmds.fieldHistoricalPriceCFETS)
            irsCurveSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
                .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(irsCurveSQL, values):
                for row in values:
                    util.WriteLog(irsCurveSQL % row)
            else:
                util.WriteErrorLog("ERRData-irsCurveSQL")

    # 没有查到CMDS数据,查询衡泰资讯

    fields = ','.join(icurve_ht.fieldSource)
    sql = "select {fields} from {db}.{table}" \
          " where BEG_DATE = '{beg_date}' and DP_BANK = 'CFETS' and Q_TYPE = '1' and end_date = '2050-12-31'"\
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_MD'], table=' TIRSWAP_SERIES', beg_date=beg_date)

    util.WriteLog(sql)
    context.oracle.query(sql)

    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步IRS昨日收盘曲线, 没有查到CMDS数据,查询衡泰资讯0条")
            break
        else:
            count += 1
            util.WriteLog('fetch IRSCurve ' + str(len(res)) + ' row')

            icurve_ht.setData(res, args={'PreTradingDate': predate})
            icurve_ht.setDefaultValue()

            fields, placeholder, values = icurve_ht.generateDataSQL(icurve_ht.dataHistoricalPriceCFETS,
                                                                    icurve_ht.fieldHistoricalPriceCFETS)
            irsCurveSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='HistoricalPriceTable_CFETS', fields=fields, placeholder=placeholder)

            if context.mysql.update(irsCurveSQL, values):
                for row in values:
                    util.WriteLog(irsCurveSQL % row)
            else:
                util.WriteErrorLog("ERRData-irsCurveSQL")

    print("SyncIRSCurve finish")

def check():

    global icurve_cmds
    global icurve_ht

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = icurve_cmds.generateCheckSQL()

    # HistoricalPriceTable_CFETS 和 IRSInfoTable_CFETS 比较合约代码
    # IRSInfoTable_CFETS
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE CombiFlag=0 AND UnderlyingIssueCode IN('FR007', 'Shibor3M', 'LPR1Y', 'LPR5Y')"\
        .format(keys=keys_str, db=context.mysql_db, table='IRSInfoTable_CFETS', predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncIRSCurve')

    contract_ref = context.mysql.fetchall()

    # HistoricalPriceTable_CFETS
    checkSQL = "SELECT {keys} FROM {db}.{table} WHERE DataDate={predate}"\
        .format(keys=keys_str, db=context.mysql_db, table='HistoricalPriceTable_CFETS', predate=predate)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncIRSCurve')

    contract_test = context.mysql.fetchall()

    set_test = set(contract_test)
    set_ref = set(contract_ref)

    # 缺少的
    set_miss = set_ref - set_test
    if set_miss:
        msg = util.getErrorMessage("ERRData-miss in HistoricalPriceTable_CFETS", list(set_miss), icurve_cmds.fieldKeys)
        util.WriteErrorLog(msg)

    # HistoricalPriceTable_CFETS 和 HistoricalPriceTable 对比
    fields_double = ['a.IssueCode', 'a.MarketCode', 'a.BasicPrice', 'b.ClearingPrice']
    fields_double_str = ','.join(fields_double)
    check_cond = "a.BasicPrice=b.ClearingPrice"

    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a, {db}.{table2} b, {db}.{table3} c" \
               " WHERE a.DataDate={predate} AND b.DataDate={predate}" \
               " AND c.CombiFlag=0 AND c.UnderlyingIssueCode IN('FR007', 'Shibor3M', 'LPR1Y', 'LPR5Y')" \
               " AND a.IssueCode=c.IssueCode AND a.MarketCode=c.MarketCode" \
               " AND {join_cond} " \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond, predate=predate,
                db=context.mysql_db, table1='HistoricalPriceTable_CFETS', table2='HistoricalPriceTable', table3='IRSInfoTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncIRSCurve')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in HistoricalPriceTable_CFETS and HistoricalPriceTable", diff, fields_double)
        util.WriteErrorLog(msg)

    print("CheckIRSCurve finish")