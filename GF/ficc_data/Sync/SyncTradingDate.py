# coding=utf-8
import datetime
from Global import context
import Utils as util
from Model.TradingDate import TradingDate

trdate = TradingDate()


def sync():

    global trdate
    global dateTable

    curdate = context.gCurrentDate

    # 1.获取交易日
    fields = ','.join(trdate.fieldSource)
    sql = "select {fields} from {db}.{table} where MARKET = '现券'" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CFETS_B_TRADE_HOLIDAY')

    context.oracle.query(sql)
    util.WriteLog(sql)

    count = 0
    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步交易日, 获取交易日数据0条")
            break
        else:
            count += 1
            trdate.setDateTable(res, field='BondTrade')

    # 2.获取结算日
    fields = ','.join(trdate.fieldSource)
    sql = "select {fields} from {db}.{table} where MARKET = '现券'"\
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='TTRD_CFETS_B_SETTLE_HOLIDAY')

    context.oracle.query(sql)
    util.WriteLog(sql)

    count = 0
    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步交易日, 获取结算日数据0条")
            break
        else:
            count += 1
            trdate.setDateTable(res, field='BondSettle')

    # 3.生成未来5年的日历, 插入数据库
    trdate.setData(args={'CurrentDate': curdate})
    trdate.setDefaultValue()

    # 插入CalendarTable_CFETS
    fields, placeholder, values = trdate.generateDataSQL(trdate.dataCalendarCFETS, trdate.fieldCalendarCFETS)

    calendarSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
        .format(db=context.mysql_db, table='CalendarTable_CFETS', fields=fields, placeholder=placeholder)

    if context.mysql.update(calendarSQL, values):
        for row in values:
            util.WriteLog(calendarSQL % row)
    else:
        util.WriteErrorLog("ERRData-calendarSQL")

    # 插入HisCalendarTable_CFETS
    fields, placeholder, values = trdate.generateDataSQL(trdate.dataHisCalendarCFETS, trdate.fieldHisCalendarCFETS)

    calendarSQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})"\
        .format(db=context.mysql_db, table='HisCalendarTable_CFETS', fields=fields, placeholder=placeholder)

    if context.mysql.update(calendarSQL, values):
        for row in values:
            util.WriteLog(calendarSQL % row)
    else:
        util.WriteErrorLog("ERRData-calendarSQL")

    print("SyncTradingDate finish")


def check():

    global trdate

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)

    today = datetime.datetime.today()
    nextyear = today.year + 1
    maxday = datetime.date(nextyear, 12, 31)

    today_str = today.strftime('%Y%m%d')
    maxday_str = maxday.strftime('%Y%m%d')

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = trdate.generateCheckSQL()

    # 交易日字段 和 结算日字段 对比
    fields = ['DTSDate', 'MarketCode', 'BondTrade', 'BondSettle']
    fields_str = ','.join(fields)
    checkSQL = "SELECT {fields} FROM {db}.{table}" \
               " WHERE DTSDate >= {today} AND DTSDate <= {maxday} AND BondTrade <> BondSettle"\
        .format(fields=fields_str, db=context.mysql_db, table='CalendarTable_CFETS', today=today_str, maxday=maxday_str)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTradingDate')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in CalendarTable_CFETS", diff, fields)
        util.WriteErrorLog(msg)

    # 跟历史数据对比 CalendarTable_CFETS 和 HisCalendarTable_CFETS
    checkSQL = "SELECT {fields} FROM {db}.{table1} a INNER JOIN {db}.{table2} b" \
               " ON {join_cond}" \
               " WHERE a.DTSDate >= {today} AND a.DTSDate <= {maxday}" \
               " AND b.DTSDate >= {today} AND b.DTSDate <= {maxday} AND b.DataDate='{predate}'" \
               " AND NOT ({check_cond})"\
        .format(fields=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                today=today_str, maxday=maxday_str, predate=predate,
                db=context.mysql_db, table1='CalendarTable_CFETS', table2='HisCalendarTable_CFETS',)

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in SyncTradingDate')

    diff = context.mysql.fetchall()
    if diff:
        msg = util.getErrorMessage("ERRData-diff in CalendarTable_CFETS", diff, fields)
        util.WriteErrorLog(msg)

    print("CheckTradingDate finish")



