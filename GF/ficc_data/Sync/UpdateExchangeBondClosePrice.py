# coding=utf-8
from Global import context
import Utils as util
from Model.ExchangeBondClosePrice import ExchangeBondClosePrice

bondprice = ExchangeBondClosePrice()

# 交易所债券收盘价
def sync():

    global bondprice

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDateExchange(curdate).encode('utf-8')

    beg_date = predate[0:4] + "-" + predate[4:6] + "-" + predate[6:8]
    fields = ','.join(bondprice.fieldSource)

    sql = "select {fields} from {db}.{table} p " \
          "where (p.M_TYPE='XSHG') and p.eval_source='债券交易所收盘价' and p.beg_date='{beg_date}'" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'], table='Ttrd_Otc_Instrument_Eval', beg_date=beg_date)

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0
    # M_TYPE,I_CODE,EVAL_NETPRICE
    # 0,     1,     2,

    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步债券不同市场信息0条")
            break
        else:
            count += 1
            util.WriteLog('fetch ExchangeBondClosePrice ' + str(len(res)) + ' row')

            bondprice.setData(res, args={'PreTradingDate': predate})
            bondprice.setDefaultValue()

            values = bondprice.getDataByField(bondprice.dataHistoricalPrice, bondprice.fieldHistoricalPrice)

            # 插入HistoricalPriceTable
            historicalPriceSQL = "UPDATE {db}.{table} " \
                                 "SET AdjustedClosePrice=%s " \
                                 "WHERE IssueCode=%s and MarketCode=%s and DataDate=%s"\
                .format(db=context.mysql_db, table='HistoricalPriceTable')

            if context.mysql.update(historicalPriceSQL, values):
                for row in values:
                    util.WriteLog(historicalPriceSQL % row)
            else:
                util.WriteErrorLog("ERRClosepriceData-historicalPriceSQL")

    print("UpdateExchangeBondClosePrice finish")

def check(threshold):

    global bondprice

    curdate = context.gCurrentDate
    predate = util.GetPreTradingDate(curdate)
    fordate = util.GetPreTradingDate(predate)

    keys_str, fields_str, fields_double, fields_double_str, join_cond, check_cond = bondprice.generateCheckSQL()

    # 先取出不同的，再比较偏离度
    checkSQL = "SELECT {fields_double} FROM {db}.{table1} a, {db}.{table2} b, {db}.{table3} c" \
               " WHERE a.DataDate='{predate}' AND b.DataDate='{fordate}' AND c.MarketCode IN ('1','2')" \
               " AND a.IssueCode=c.IssueCode AND a.MarketCode=c.MarketCode" \
               " AND {join_cond}" \
               " AND NOT ({check_cond})" \
        .format(fields_double=fields_double_str, join_cond=join_cond, check_cond=check_cond,
                predate=predate, fordate=fordate,
                db=context.mysql_db, table1='HistoricalPriceTable', table2='HistoricalPriceTable', table3='BondInfoTable_CFETS')

    if not context.mysql.query(checkSQL):
        util.WriteErrorLog('ERR-CheckSQL in UpdateExchangeBondClosePrice')

    diff = context.mysql.fetchall()

    abnormal = []
    if diff:
        for row in list(diff):
            if row[3] == 0 or abs(row[2] - row[3]) / row[3] > threshold:
                abnormal.append(row)
        if len(abnormal) > 0:
            msg = util.getErrorMessage("ERRData-abnormal in HistoricalPriceTable", abnormal, fields_double)
            util.WriteErrorLog(msg)

    print("CheckExchangeBondClosePrice finish")