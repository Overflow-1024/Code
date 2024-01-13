# coding=utf-8
from Global import context
import Utils as util
from Model.BondPosition import BondPosition
from Model.BondPosition import FundPosition
from Model.BondPosition import TFPosition

bondposi = BondPosition()
fundposi = FundPosition()
tfposi = TFPosition()


def sync():
    global bondposi
    global fundposi
    global tfposi

    curdate = context.gCurrentDate

    # 债券持仓
    sql = "SELECT T.I_CODE,T.B_NAME,T.A_TYPE,T.M_TYPE,A.ACCID,A.PS_L_AMOUNT,T.B_EXTEND_TYPE,T.PENETRATEISSUER" \
          " FROM %s.TBND T" \
          " LEFT JOIN" \
          " (SELECT TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT) PS_L_AMOUNT" \
          " FROM %s.TBND TB LEFT JOIN %s.TTRD_ACC_BALANCE_SECU SE" \
          " ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE" \
          " WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD')" \
          " AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN" \
          " (SELECT COMPONENT ACCID FROM" \
          " (SELECT * FROM %s.TTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部')" \
          " CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0)" \
          " GROUP BY TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID)A" \
          " ON T.I_CODE = A.I_CODE AND T.A_TYPE = A.A_TYPE AND T.M_TYPE = A.M_TYPE WHERE T.PENETRATEISSUER IN" \
          " (SELECT DISTINCT TB.PENETRATEISSUER" \
          " FROM %s.TBND TB LEFT JOIN %s.TTRD_ACC_BALANCE_SECU SE" \
          " ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE" \
          " WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD')" \
          " AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN" \
          " (SELECT COMPONENT ACCID FROM" \
          " (SELECT * FROM %s.TTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部')" \
          " CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0))" \
          " AND T.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD') AND T.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD')" \
          " AND T.I_CODE NOT LIKE 'UL%%'" \
          " UNION" \
          " SELECT T.I_CODE,T.B_NAME,T.A_TYPE,T.M_TYPE,A.ACCID,A.PS_L_AMOUNT,T.B_EXTEND_TYPE,T.PENETRATEISSUER" \
          " FROM %s.TBND T JOIN" \
          " (SELECT TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT) PS_L_AMOUNT" \
          " FROM %s.TBND TB LEFT JOIN %s.TTRD_ACC_BALANCE_SECU SE" \
          " ON TB.I_CODE = SE.I_CODE AND TB.A_TYPE = SE.A_TYPE AND TB.M_TYPE = SE.M_TYPE" \
          " WHERE TB.A_TYPE IN ('SPT_BD', 'SPT_ABS') AND TB.M_TYPE IN ('XSHG', 'XSHE', 'X_CNBD')" \
          " AND TB.B_MTR_DATE > TO_CHAR(SYSDATE, 'YYYY-MM-DD') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN" \
          " (SELECT COMPONENT ACCID FROM" \
          " (SELECT * FROM %s.TTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部')" \
          " CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0) AND TB.PENETRATEISSUER IS NULL" \
          " GROUP BY TB.I_CODE,TB.B_NAME,TB.A_TYPE,TB.M_TYPE,TB.B_EXTEND_TYPE,SE.ACCID) A" \
          " ON T.I_CODE = A.I_CODE AND T.A_TYPE = A.A_TYPE AND T.M_TYPE = A.M_TYPE" % (
    context.GlobalSettingTable['Oracle_XIR_MD'], context.GlobalSettingTable['Oracle_XIR_MD'], context.GlobalSettingTable['Oracle_XIR_TRD'],
    context.GlobalSettingTable['Oracle_XIR_TRD'], context.GlobalSettingTable['Oracle_XIR_MD'], context.GlobalSettingTable['Oracle_XIR_TRD'],
    context.GlobalSettingTable['Oracle_XIR_TRD'], context.GlobalSettingTable['Oracle_XIR_MD'], context.GlobalSettingTable['Oracle_XIR_MD'],
    context.GlobalSettingTable['Oracle_XIR_TRD'], context.GlobalSettingTable['Oracle_XIR_TRD'])

    util.WriteLog(sql)
    context.oracle.query(sql)
    count = 0

    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步衡泰持仓, 债券持仓0条")
            break
        else:
            count += 1
            bondposi.setData(res, args={'CurrentDate': curdate})
            bondposi.setDefaultValue()

            fields, placeholder, values = bondposi.generateDataSQL(bondposi.dataHTPosition,
                                                                   bondposi.fieldHTPosition)

            # WriteLog(
            #     "SyncBondPosition,issueCode:%s,issueName:%s,assetType:%s,marketCode:%s,accountCode:%s,amount:%s,bondExtendType:%s,penetrateIssuer:%s" % (
            #     issueCode, issueName, assetType, marketCode, accountCode, amount, bondExtendType, penetrateIssuer))

            bondPositionSQL = "REPLACE INTO {db}.HT_PositionTable_CFETS ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, fields=fields, placeholder=placeholder)

            if context.mysql.update(bondPositionSQL, values):
                for row in values:
                    util.WriteLog(bondPositionSQL % row)
            else:
                util.WriteLog("ERRData-bondPositionSQL")

    # 基金持仓
    sql = "SELECT SE.I_CODE,TF.F_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID,SUM(SE.PS_L_AMOUNT)" \
          " FROM {db1}.TFND TF JOIN {db2}.TTRD_ACC_BALANCE_SECU SE" \
          " ON TF.I_CODE = SE.I_CODE AND TF.A_TYPE = SE.A_TYPE AND TF.M_TYPE = SE.M_TYPE" \
          " WHERE TF.M_TYPE IN ('XSHG','XSHE') AND SE.PS_L_AMOUNT <> 0 AND SE.ACCID IN" \
          " (SELECT COMPONENT ACCID FROM" \
          " (SELECT * FROM {db3}.TTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('固定收益投资部')" \
          " CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0)" \
          " GROUP BY SE.I_CODE,TF.F_NAME,SE.A_TYPE,SE.M_TYPE,SE.ACCID" \
        .format(db1=context.GlobalSettingTable['Oracle_XIR_MD'], db2=context.GlobalSettingTable['Oracle_XIR_TRD'],
                db3=context.GlobalSettingTable['Oracle_XIR_TRD'])

    util.WriteLog(sql)
    context.oracle.query(sql)
    count = 0

    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步衡泰持仓, 基金持仓0条")
            break
        else:
            count += 1

            fundposi.setData(res, args={'CurrentDate': curdate})
            fundposi.setDefaultValue()

            fields, placeholder, values = fundposi.generateDataSQL(fundposi.dataHTPosition,
                                                                   fundposi.fieldHTPosition)

            # util.WriteLog("SyncBondPosition,issueCode:%s,issueName:%s,assetType:%s,marketCode:%s,accountCode:%s,amount:%s" % (
            # issueCode, issueName, assetType, marketCode, accountCode, amount))

            bondPositionSQL = "REPLACE INTO {db}.HT_PositionTable_CFETS({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, fields=fields, placeholder=placeholder)

            if context.mysql.update(bondPositionSQL, values):
                for row in values:
                    util.WriteLog(bondPositionSQL % row)
            else:
                util.WriteLog("ERRData-bondPositionSQL")

    # 国债期货持仓
    sql = "SELECT SE.I_CODE,SE.ACCID,SE.RT_L_AVAAMOUNT,SE.LS,SE.SECU_EXT_ACCID,SE.PS_L_COST,SE.BEG_DATE,SE.A_TYPE,SE.M_TYPE" \
          " FROM {db1}.TTRD_ACC_BALANCE_SECU SE WHERE SE.A_TYPE = 'FUT_BD' AND SE.ACCID IN" \
          " (SELECT COMPONENT ACCID FROM" \
          " (SELECT * FROM {db2}.TTRD_ACC_PACKAGE_COMPONENT K START WITH K.PACKAGEID IN ('证券投资总部')" \
          " CONNECT BY PRIOR K.COMPONENT = K.PACKAGEID) P WHERE P.COMPONENTTYPE = 0)" \
        .format(db1=context.GlobalSettingTable['Oracle_XIR_TRD'], db2=context.GlobalSettingTable['Oracle_XIR_TRD'])

    util.WriteLog(sql)
    context.oracle.query(sql)
    count = 0

    while True:
        res = context.oracle.fetchmany(int(context.GlobalSettingTable['NumberofLines_inserted']))
        if not res:
            if count == 0:
                util.WriteErrorLog("同步衡泰持仓, 国债期货持仓0条")
            break
        else:
            count += 1

            tfposi.setData(res, args={'CurrentDate': curdate})
            tfposi.setDefaultValue()

            fields, placeholder, values = tfposi.generateDataSQL(tfposi.dataHTPosition,
                                                                 tfposi.fieldHTPosition)

            # util.WriteLog("SyncTFPosition,issueCode:%s,accountCode:%s,quantity:%s,baSubID:%s,investorID:%s,amount:%s" % (
            # issueCode, accountCode, quantity, baSubID, investorID, amount))

            tfPositionSQL = "REPLACE INTO {db}.HT_PositionTable_CFETS ({fields}) VALUES ({placeholder})" \
                .format(db=context.mysql_db, fields=fields, placeholder=placeholder)

            if context.mysql.update(tfPositionSQL, values):
                for row in values:
                    util.WriteLog(tfPositionSQL % row)
            else:
                util.WriteLog("ERRData-tfPositionSQL")

    print("SyncBondPosition finish")
