# coding=utf-8
from Model.Base import DataModel
import copy


# 可交换债/可转换债
class ExchangeBondClosePrice(DataModel):

    def __init__(self):
        super(ExchangeBondClosePrice, self).__init__()

        self.dataHistoricalPrice = []

        self.fieldSource = [
            'M_TYPE',
            'I_CODE',
            'EVAL_NETPRICE'
        ]
        self.fieldHistoricalPrice = [
            'AdjustedClosePrice',
            'IssueCode',
            'MarketCode',
            'DataDate'
        ]
        self.fieldCheck = [
            'AdjustedClosePrice'    # 收盘价
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',    # 债券代码
            'MarketCode'    # 市场代码
        ]


    def setData(self, res, args=None):

        self.data = []
        self.dataHistoricalPrice = []

        def setDataElement(row):
            data_em = dict()
            data_em['IssueCode'] = row[1]
            data_em['MarketCode'] = '0'
            if row[0] == 'XSHG':
                data_em['MarketCode'] = '1'
                data_em['IssueCode'] = 'SH' + row[1]
            elif row[0] == 'XSHE':
                data_em['MarketCode'] = '2'
                data_em['IssueCode'] = 'SZ' + row[1]
            data_em['AdjustedClosePrice'] = row[2]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataHistoricalPrice = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHistoricalPrice, self.dataHistoricalPrice)

    def setValueHistoricalPrice(self, data_em):
        data_em['DataDate'] = self.time['PreTradingDate']
