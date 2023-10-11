# coding=utf-8
from Global import context
import Utils as util


def sync():
    curdate = context.gCurrentDate

    sql = "SELECT I_CODE FROM" \
          " (SELECT I_CODE FROM {db1}.TTRD_CFETS_B_BOND" \
          " WHERE (BOND_TYPE = '国债' OR BOND_TYPE = '政策性金融债') AND COUPON_TYPE = '固定利率' AND LENGTH(I_CODE) = 6" \
          " AND (TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) / 365 >=0.5" \
          " AND (TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) / 365 <=10" \
          " AND I_CODE IN" \
          " (SELECT K.SECURITY_ID FROM {db2}.TTRD_CFETS_B_XBONDINFO K" \
          " WHERE SUBSTR(K.UPDATETIME, 1, 10) = TO_CHAR(SYSDATE, 'YYYY-MM-DD'))" \
          " ORDER BY SUBSTR(ISSUE_DATE, 1, 4) DESC,TO_NUMBER(TO_DATE(MATURITY_DATE, 'YYYY-MM-DD') - SYSDATE) DESC)" \
          " WHERE ROWNUM <= 200"\
          .format(db1=context.GlobalSettingTable['Oracle_XIR_TRD'], db2=context.GlobalSettingTable['Oracle_XIR_TRD'])

    context.oracle.query(sql)
    count = 0
    while 1:
        res = context.oracle.fetchone()
        if not res:
            break

        issueCode = "IB" + res[0]
        count += 1

        autoSubscribeSQL = "Update {db}.BondInfo_XBond_CFETS set AutoSubscribe = '1' " \
                           "where DataDate ='{curdate}' and IssueCode = '{issueCode}'"\
            .format(db=context.mysql_db, issueCode=issueCode, curdate=curdate)
        util.WriteLog(autoSubscribeSQL)

        if not context.mysql.updateone(autoSubscribeSQL):
            util.WriteLog("ERRData-autoSubscribeSQL")

    if count == 0:
        util.WriteErrorLog("获取自动订阅的XBOND债券0条")

    print("SyncAutoSubscribeXBOND finish")