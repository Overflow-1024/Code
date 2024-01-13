# coding=utf-8
from Model.Base import DataModel
import copy


# 可交换债/可转换债
class IRSCurveCMDS(DataModel):

    def __init__(self):
        super(IRSCurveCMDS, self).__init__()

        self.dataHistoricalPriceCFETS = []

        self.fieldSource = [

        ]
        self.fieldHistoricalPriceCFETS = [
            'IssueCode',
            'MarketCode',
            'DataDate',
            'ClearingPrice',
            'BasicPrice'
        ]
        self.fieldCheck = [
            'BasicPrice'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',    # 债券代码
            'MarketCode'    # 市场代码
        ]


    def setData(self, res, args=None):

        self.data = []
        self.dataHistoricalPriceCFETS = []

        def setDataElement(row):
            data_em = dict()
            data_em['IssueCode'] = 'IB' + row[0] + '_' + row[3]
            data_em['BasicPrice'] = row[1]
            data_em['Comment'] = row[2]
            data_em['Term'] = row[3]
            data_em['BeginDate'] = row[4]
            data_em['UpdateTime'] = row[5]

            # 时间
            data_em['DataDate'] = args.get('PreTradingDate')
            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataHistoricalPriceCFETS = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHistoricalPriceCFETS, self.dataHistoricalPriceCFETS)

    def setValueHistoricalPriceCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['ClearingPrice'] = 0
        data_em['DataDate'] = self.time['PreTradingDate']


class IRSCurveHT(DataModel):

    def __init__(self):
        super(IRSCurveHT, self).__init__()

        self.dataHistoricalPriceCFETS = []

        self.fieldSource = [
            'I_CODE',
            'DP_CLOSE',
            'BEG_DATE'
        ]
        self.fieldHistoricalPriceCFETS = [
            'IssueCode',
            'MarketCode',
            'DataDate',
            'ClearingPrice',
            'BasicPrice'
        ]
        self.fieldCheck = [
            'BasicPrice'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',    # 债券代码
            'MarketCode'    # 市场代码
        ]


    def setData(self, res, args=None):

        self.data = []
        self.dataHistoricalPriceCFETS = []

        def setDataElement(row):
            data_em = dict()

            issueCode = row[0]
            data_em['BasicPrice'] = row[1]
            data_em['BeginDate'] = row[2]

            if issueCode[:6] == 'LPR_1Y':
                issueCode = 'LPR1Y' + issueCode[6:]
            elif issueCode[:6] == 'LPR_5Y':
                issueCode = 'LPR5Y' + issueCode[6:]
            elif issueCode[:9] == 'SHIBOR-1D':
                issueCode = 'ShiborO/N' + issueCode[9:]
            elif issueCode[:9] == 'SHIBOR-3M':
                issueCode = 'Shibor3M' + issueCode[9:]
            else:
                pass

            data_em['IssueCode'] = 'IB' + issueCode

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataHistoricalPriceCFETS = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHistoricalPriceCFETS, self.dataHistoricalPriceCFETS)

    def setValueHistoricalPriceCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['ClearingPrice'] = 0
        data_em['DataDate'] = self.time['PreTradingDate']