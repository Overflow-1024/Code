# coding=utf-8
import datetime
import copy
from Model.Base import DataModel
from Global import context


class TradingDate(DataModel):

    def __init__(self):
        super(TradingDate, self).__init__()

        self.dataCalendarCFETS = []
        self.dataHisCalendarCFETS = []

        self.dateTable = {}
        self.maxDate = context.gCurrentDate

        self.fieldSource = [
            'MARKET',
            'HOLIDAY',
            'UPDATETIME'
        ]
        self.fieldCalendarCFETS = [
            'DTSDate',
            'MarketCode',
            'BondTrade',
            'BondSettle',
            'CreateTime',
            'TimeStamp'
        ]
        self.fieldHisCalendarCFETS = [
            'DTSDate',
            'MarketCode',
            'BondTrade',
            'BondSettle',
            'CreateTime',
            'TimeStamp',
            'DataDate'
        ]

        self.fieldCheck = [
            'BondTrade',
            'BondSettle',
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'DTSDate',
            'MarketCode'
        ]

    def setDateTable(self, res, field):

        # field两种取值：BondTrade, BondSettle 用来区分交易日和计算日
        def setOneDate(row):

            market = row[0]
            holiday = row[1][0:4] + row[1][5:7] + row[1][8:10]
            updateTime = row[2]

            if not self.dateTable.has_key(holiday):
                self.dateTable[holiday] = {}

            self.dateTable[holiday][field] = 1
            if self.maxDate < holiday:
                self.maxDate = holiday

        map(setOneDate, res)

    def setData(self, args=None):

        res = []
        for i in range(0, 5 * 365):
            date_tmp = str(datetime.date.today() - datetime.timedelta(days=-i))
            date = date_tmp[0:4] + date_tmp[5:7] + date_tmp[8:10]

            trade = 0
            settle = 0

            # 如果有日程表数据，取日程表数据值
            if self.dateTable.has_key(date):
                if self.dateTable[date].has_key('BondTrade'):
                    trade = 1

                if self.dateTable[date].has_key('BondSettle'):
                    settle = 1

            # 如果大于获取到的最大假日,根据周六日赋值默认
            elif date > self.maxDate:
                w = datetime.datetime(int(date_tmp[0:4]), int(date_tmp[5:7]), int(date_tmp[8:10])).strftime("%w")
                if w == '6' or w == '0':
                    trade = 1
                    settle = 1

            res.append((date, trade, settle))

        self.data = []
        self.dataCalendarCFETS = []
        self.dataHisCalendarCFETS = []

        def setDataElement(row):
            data_em = dict()

            data_em['DTSDate'] = row[0]
            data_em['BondTrade'] = row[1]
            data_em['BondSettle'] = row[2]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataCalendarCFETS = copy.deepcopy(self.data)
        self.dataHisCalendarCFETS = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueCalendarCFETS, self.dataCalendarCFETS)
        map(self.setValueHisCalendarCFETS, self.dataHisCalendarCFETS)


    def setValueCalendarCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def setValueHisCalendarCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DataDate'] = self.time['CurrentDate']

