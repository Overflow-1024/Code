# coding=utf-8
from Global import context
import Utils as util
from Model.TradeParty import TradeParty


party = TradeParty()

def sync():

    global party

    curdate = context.gCurrentDate

    fields = ','.join(party.fieldSource)
    sql = "SELECT {fields} FROM {db}.TTRD_OTC_COUNTERPARTY WHERE PARTY_STATUS='1'" \
        .format(fields=fields, db=context.GlobalSettingTable['Oracle_XIR_TRD'])

    context.oracle.query(sql)
    util.WriteLog(sql)
    count = 0
    while True:

        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步交易对手信息0条")
            break
        else:
            count += 1

            party.setData(res, args={'CurrentDate': curdate})
            party.setDefaultValue()

            fields, placeholder, values = party.generateDataSQL(party.dataTradeParty,
                                                                party.fieldTradeParty)
            partySQL = "REPLACE INTO {db}.{table} ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, table='TradeMemberTable', fields=fields, placeholder=placeholder)

            if context.mysql.update(partySQL, values):
                for row in values:
                    util.WriteLog(partySQL % row)
            else:
                util.WriteErrorLog("ERRData-partySQL")
