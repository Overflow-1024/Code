# coding=utf-8
from Model.Base import DataModel
import copy
import datetime


class CCPClearPrice(DataModel):

    def __init__(self):
        super(CCPClearPrice, self).__init__()

        self.dataHistoricalPrice = []
        self.dataHistoricalPriceCFETS = []

        self.fieldSource = [

        ]
        self.fieldHistoricalPrice = [
            'ClearingPrice',
            'TimeStamp',
            'DataDate',
            'IssueCode'
        ]
        self.fieldHistoricalPriceCFETS = [
            'IssueCode',
            'MarketCode',
            'DataDate',
            'ClearingPrice'
        ]

        self.fieldCheck = [
            'ClearingPrice'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode'
        ]
        # 合约列表（时间占四个字符，先用0000作为占位符）
        self.contractTemplate = [
            'IBCDB2_0000P',
            'IBADBC7_0000P',
            'IBADBC2_0000P',
            'IBCDB5_0000',
            'IBCDB3_0000',
            'IBCDB10_0000',
            'IBADBC5_0000',
            'IBADBC10_0000'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataHistoricalPrice = []
        self.dataHistoricalPriceCFETS = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = "IB" + row[0]
            data_em['ClearingPrice'] = row[1]

            # 浮点数的插入转成字符串类型（不然会丢失精度）
            data_em['ClearingPrice'] = str(data_em['ClearingPrice'])

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataHistoricalPrice = copy.deepcopy(self.data)
        self.dataHistoricalPriceCFETS = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHistoricalPrice, self.dataHistoricalPrice)
        map(self.setValueHistoricalPriceCFETS, self.dataHistoricalPriceCFETS)

    def setValueHistoricalPrice(self, data_em):
        data_em['TimeStamp'] = datetime.datetime.now()
        data_em['DataDate'] = self.time['PreTradingDate']

    def setValueHistoricalPriceCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['DataDate'] = self.time['PreTradingDate']

