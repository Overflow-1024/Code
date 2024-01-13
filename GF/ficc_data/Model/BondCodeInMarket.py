# coding=utf-8
from Model.Base import DataModel
import copy


# 债券估值
class BondCodeInMarket(DataModel):

    def __init__(self):
        super(BondCodeInMarket, self).__init__()

        self.dataBondCodeInMarket = []
        self.dataHisBondCodeInMarket = []

        self.fieldSource = [
            'I_CODE',
            'A_TYPE',
            'M_TYPE',
            'SH_CODE',
            'SZ_CODE',
            'YH_CODE'
        ]
        self.fieldBondCodeInMarket = [
            'IssueCode',
            'MarketCode',
            'SH_Code',
            'SZ_Code',
            'YH_Code',
            'AssetType'
        ]
        self.fieldHisBondCodeInMarket = [
            'IssueCode',
            'MarketCode',
            'SH_Code',
            'SZ_Code',
            'YH_Code',
            'AssetType',
            'DataDate'
        ]
        self.fieldCheck = [
            'SH_Code',
            'SZ_Code',
            'YH_Code'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataBondCodeInMarket = []
        self.dataHisBondCodeInMarket = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = str(row[0])
            data_em['MarketCode'] = 0
            if row[2] == 'XSHG':
                data_em['IssueCode'] = 'SH' + str(row[0])
                data_em['MarketCode'] = 1
            elif row[2] == 'XSHE':
                data_em['IssueCode'] = 'SZ' + str(row[0])
                data_em['MarketCode'] = 2
            elif row[2] == 'X_CNBD':
                data_em['IssueCode'] = 'IB' + str(row[0])
                data_em['MarketCode'] = 9
            data_em['SH_Code'] = ''
            data_em['SZ_Code'] = ''
            data_em['YH_Code'] = ''
            if str(row[3]) != 'None':
                data_em['SH_Code'] = 'SH' + str(row[3])
            if str(row[4]) != 'None':
                data_em['SZ_Code'] = 'SZ' + str(row[4])
            if str(row[5]) != 'None':
                data_em['YH_Code'] = 'IB' + str(row[5])
            data_em['AssetType'] = row[1]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataBondCodeInMarket = copy.deepcopy(self.data)
        self.dataHisBondCodeInMarket = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHisBondCodeInMarket, self.dataHisBondCodeInMarket)

    def setValueHisBondCodeInMarket(self, data_em):
        data_em['DataDate'] = self.time['CurrentDate']


