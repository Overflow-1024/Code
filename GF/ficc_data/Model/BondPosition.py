# coding=utf-8
from Model.Base import DataModel
import copy


class BondPosition(DataModel):

    def __init__(self):
        super(BondPosition, self).__init__()

        self.dataHTPosition = []

        self.fieldSource = [

        ]
        self.fieldHTPosition = [
            'IssueCode',
            'IssueName',
            'MarketCode',
            'AssetType',
            'HTAccountCode',
            'PositionAmount',
            'BondExtendType',
            'PenetrateIssuer',
            'DataDate'
        ]
        self.fieldCheck = []
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode',
            'HTAccountCode',
            'LS'
        ]

    def setData(self, res, args=None):
        self.data = []
        self.dataHTPosition = []

        def setDataElement(row):
            data_em = dict()
            data_em['IssueCode'] = row[0]  # 债券代码
            data_em['IssueName'] = row[1]  # 债券名称
            data_em['AssetType'] = row[2]  # 资产类型
            data_em['MarketCode'] = row[3]  # 市场代码
            data_em['HTAccountCode'] = row[4]  # 内证
            data_em['PositionAmount'] = row[5]  # 持仓金额
            data_em['BondExtendType'] = row[6]  # 债券扩展类型
            data_em['PenetrateIssuer'] = row[7]  # 债券风控主体

            if data_em['MarketCode'] == 'XSHG':
                data_em['MarketCode'] = '1'
            elif data_em['MarketCode'] == 'XSHE':
                data_em['MarketCode'] = '2'
            elif data_em['MarketCode'] == 'X_CNBD':
                data_em['MarketCode'] = '9'

            if data_em['HTAccountCode'] is None:
                data_em['HTAccountCode'] = ''

            if data_em['PositionAmount'] is None:
                data_em['PositionAmount'] = 0

            if data_em['PenetrateIssuer'] is None:
                data_em['PenetrateIssuer'] = ''

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataHTPosition = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHTPosition, self.dataHTPosition)

    def setValueHTPosition(self, data_em):
        data_em['DataDate'] = self.time['CurrentDate']


class FundPosition(DataModel):

    def __init__(self):
        super(FundPosition, self).__init__()

        self.dataHTPosition = []

        self.fieldSource = [

        ]
        self.fieldHTPosition = [
            'IssueCode',
            'IssueName',
            'MarketCode',
            'AssetType',
            'HTAccountCode',
            'PositionAmount',
            'DataDate'
        ]
        self.fieldCheck = []
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode',
            'HTAccountCode',
            'LS'
        ]

    def setData(self, res, args=None):
        self.data = []
        self.dataHTPosition = []

        def setDataElement(row):
            data_em = dict()
            data_em['IssueCode'] = row[0]  # 债券代码
            data_em['IssueName'] = row[1]  # 债券名称
            data_em['AssetType'] = row[2]  # 资产类型
            data_em['MarketCode'] = row[3]  # 市场代码
            data_em['HTAccountCode'] = row[4]  # 内证
            data_em['PositionAmount'] = row[5]  # 持仓金额

            if data_em['MarketCode'] == 'XSHG':
                data_em['MarketCode'] = '1'
            elif data_em['MarketCode'] == 'XSHE':
                data_em['MarketCode'] = '2'
            elif data_em['MarketCode'] == 'X_CNBD':
                data_em['MarketCode'] = '9'

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataHTPosition = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHTPosition, self.dataHTPosition)

    def setValueHTPosition(self, data_em):
        data_em['DataDate'] = self.time['CurrentDate']


class TFPosition(DataModel):

    def __init__(self):
        super(TFPosition, self).__init__()

        self.dataHTPosition = []

        self.fieldSource = [

        ]
        self.fieldHTPosition = [
            'IssueCode',
            'IssueName',
            'MarketCode',
            'AssetType',
            'HTAccountCode',
            'PositionAmount',
            'LS',
            'Amount',
            'DataDate'
        ]
        self.fieldCheck = []
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode',
            'HTAccountCode',
            'LS'
        ]

    def setData(self, res, args=None):
        self.data = []
        self.dataHTPosition = []

        def setDataElement(row):
            data_em = dict()
            data_em['IssueCode'] = row[0]  # 合约代码
            data_em['IssueName'] = row[0]  # 合约代码
            data_em['HTAccountCode'] = row[1]  # 内证
            data_em['PositionAmount'] = row[2]  # 多仓数量
            data_em['LS'] = row[3]          # 多空方向
            data_em['InvestorID'] = row[4]  # 资金账号
            data_em['Amount'] = row[5]  # 持仓成本

            if data_em['LS'] == 'S':
                data_em['LS'] = '1'
            else:
                data_em['LS'] = '3'

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataHTPosition = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHTPosition, self.dataHTPosition)

    def setValueHTPosition(self, data_em):
        data_em['MarketCode'] = '3'
        data_em['AssetType'] = 'FUT_BD'
        data_em['DataDate'] = self.time['CurrentDate']
