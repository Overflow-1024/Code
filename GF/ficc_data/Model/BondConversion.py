# coding=utf-8
from Model.Base import DataModel
import copy

# 可交换债/可转换债
class BondConversion(DataModel):

    def __init__(self):
        super(BondConversion, self).__init__()

        self.dataBondConversion = []
        self.dataHisBondConversion = []

        self.fieldSource = [
            'TB.I_CODE',
            'TB.A_TYPE',
            'TB.M_TYPE',
            'TB.B_NAME',
            'BC.BEG_DATE',
            'BC.CONV_CODE',
            'BC.CONV_PRICE',
            'TB.P_CLASS',
            'TB.B_EXTEND_TYPE'
        ]
        self.fieldBondConversion = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ConvDate',
            'ConvCode',
            'ConvPrice',
            'BondType',
            'BondExtredType'
        ]
        self.fieldHisBondConversion = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ConvDate',
            'ConvCode',
            'ConvPrice',
            'BondType',
            'BondExtredType',
            'DataDate'
        ]
        self.fieldCheck = [
            'IssueName',    # 债券名称
            'ConvCode',     # 转股代码
            'ConvPrice',    # 转股价格
            'BondType',     # 债券类型
            'BondExtredType'  # 债券扩展类型
        ]

        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',    # 债券代码
            'MarketCode'    # 市场代码
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataBondConversion = []
        self.dataHisBondConversion = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = row[0]  # 债券代码
            data_em['AssetType'] = row[1]  # 资产类型
            data_em['MarketCode'] = row[2]  # 市场代码
            data_em['IssueName'] = row[3]  # 债券名称
            data_em['ConvDate'] = row[4]  # 转股日期
            data_em['ConvCode'] = row[5]  # 转股代码
            data_em['ConvPrice'] = row[6]  # 转股价格
            data_em['BondType'] = row[7]  # 债券类型
            data_em['BondExtredType'] = row[8]  # 债券扩展类型

            if data_em['MarketCode'] == 'XSHG':
                data_em['MarketCode'] = '1'
            elif data_em['MarketCode'] == 'XSHE':
                data_em['MarketCode'] = '2'

            convdate = data_em['ConvDate']
            data_em['ConvDate'] = convdate.replace('-', '')

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataBondConversion = copy.deepcopy(self.data)
        self.dataHisBondConversion = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueHisBondConversion, self.dataHisBondConversion)

    def setValueHisBondConversion(self, data_em):
        data_em['DataDate'] = self.time['CurrentDate']

