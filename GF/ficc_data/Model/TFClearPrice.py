# coding=utf-8
from Model.Base import DataModel
import copy


class TFClearPrice(DataModel):

    def __init__(self):
        super(TFClearPrice, self).__init__()

        self.dataHistoricalPriceCFETS = []

        self.fieldSource = [
            'p.i_code',
            'p.list_set_price',
            'q.dp_set'
        ]
        self.fieldHistoricalPriceCFETS = [
            'IssueCode',
            'MarketCode',
            'DataDate',
            'ClearingPrice',
            'BasicPrice'
        ]
        self.fieldCheck = [
            'BasicPrice',
            'ClearingPrice'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode'
        ]
        # 合约列表（时间占四个字符，先用0000作为占位符）
        self.contractTemplate = [
            'T0000',
            'TF0000',
            'TS0000'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataHistoricalPriceCFETS = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = row[0]
            data_em['BasicPrice'] = row[1]
            data_em['ClearingPrice'] = row[2]

            # 浮点数的插入转成字符串类型（不然会丢失精度）
            data_em['BasicPrice'] = str(data_em['BasicPrice'])
            data_em['ClearingPrice'] = str(data_em['ClearingPrice'])

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataHistoricalPriceCFETS = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHistoricalPriceCFETS, self.dataHistoricalPriceCFETS)

    def setValueHistoricalPriceCFETS(self, data_em):
        data_em['MarketCode'] = '3'
        data_em['DataDate'] = self.time['PreTradingDate']

    def setFieldCheck(self, field):
        self.fieldCheck = copy.copy(field)

