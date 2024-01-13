# -*- coding: utf-8 -*-
from __future__ import division
import unicodedata
import time
import datetime
import sys

from Global import context

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def WriteLog(log):

    nowTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    context.file.write(nowTime+"    "+log+"\n")


def WriteErrorLog(log):

    nowTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    context.errorFile.write(nowTime+"    "+log+"\n")


def getErrorMessage(title, values, fields, flag=False, pos=None):
    # values：list(tuple) 或者 tuple(tuple)
    # pos[0]:keys数量  pos[1]:fields数量  len(fields) = pos[0] + 2 * pos[1]
    time_space = "                 "
    errMessage = title + ": (" + str(len(values)) + ")\n"

    if flag:
        # 验证字段数量能否对上号，不然下面逻辑会出错
        if len(fields) != pos[0] + 2 * pos[1]:
            print("invalid fields length in getErrorMessage()")
            return errMessage

        for value in values:

            strList = []
            # 处理keys
            for i in range(0, pos[0]):
                strList.append(fields[i] + '[' + str(value[i]).encode('latin1').decode('utf8') + ']')
            # 处理比较字段
            a_start = pos[0]
            b_start = a_start + pos[1]
            for j in range(0, pos[1]):
                if value[a_start + j] != value[b_start + j]:
                    strList.append(fields[a_start + j] + '[' + str(value[a_start + j]).encode('latin1').decode('utf8') + ']')
                    strList.append(fields[b_start + j] + '[' + str(value[b_start + j]).encode('latin1').decode('utf8') + ']')

            errMessage = errMessage + time_space + title + ": " + ', '.join(strList) + '\n'
    else:
        for value in values:
            strList = []
            for i in range(len(fields)):
                strList.append(fields[i] + '[' + str(value[i]).encode('latin1').decode('utf8') + ']')
            errMessage = errMessage + time_space + title + ": " + ', '.join(strList) + '\n'

    return errMessage


# -------------------------------- 时间类函数 --------------------------------


def GetPreTradingDate(current_date):
    preTradingDate = current_date
    # 获取上一交易日
    sql = "select max(DTSDate) from {db}.CalendarTable_CFETS where DTSDate < '{curdate}' AND BondTrade = '0'"\
        .format(db=context.mysql_db, curdate=current_date)

    context.mysql.query(sql)
    results = context.mysql.fetchall()

    for row in results:
        preTradingDate = row[0]
    WriteLog("GetPreTradingDate,preTradingDate:" + preTradingDate)

    return preTradingDate


# 获取下一个结算日
def GetNextSettleDate(obj):
    nextSettleDate = obj.strftime('%Y%m%d')
    sql = "select min(DTSDate) from {db}.CalendarTable_CFETS where DTSDate >= '{nextdate}' AND BondTrade = '0'"\
        .format(db=context.mysql_db, nextdate=nextSettleDate)
    WriteLog('GetNextSettleDate,sql:' + sql)

    context.mysql.query(sql)
    results = context.mysql.fetchall()

    for row in results:
        nextSettleDate = row[0]

    WriteLog("GetNextSettleDate,nextSettleDate:" + nextSettleDate)
    return datetime.date(int(nextSettleDate[0:4]), int(nextSettleDate[4:6]), int(nextSettleDate[6:8]))


def GetPreTradingDateExchange(current_date):
    preTradingDate = current_date
    # 获取上一交易日
    sql = "select max(DTSDate) from {db}.CalendarTable where DTSDate < '{curdate}' AND DayOffFlag = '0'"\
        .format(db=context.mysql_db, curdate=current_date)

    context.mysql.query(sql)
    results = context.mysql.fetchall()

    for row in results:
        preTradingDate = row[0]
    WriteLog("GetPreTradingDate,preTradingDate:" + preTradingDate)
    return preTradingDate






