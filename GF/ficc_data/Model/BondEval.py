# coding=utf-8
from Model.Base import DataModel
import copy


# 债券估值
class BondEval(DataModel):

    def __init__(self):
        super(BondEval, self).__init__()

        self.dataBondEval = []

        self.fieldSource = [
            'beg_date',
            'i_code',
            'yield',
            'netprice',
            'fullprice',
            'm_type'
        ]
        self.fieldBondEval = [
            'IssueCode',
            'DataDate',
            'Yield',
            'NetPrice',
            'FullPrice'
        ]
        self.fieldCheck = [
            'Yield',     # 估值收益率
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',  # 债券代码
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataBondEval = []
        def setDataElement(row):
            data_em = dict()

            if row[5] == 'X_CNBD':
                data_em['IssueCode'] = "IB" + row[1]
            elif row[5] == 'XSHE':
                data_em['IssueCode'] = "SZ" + row[1]
            elif row[5] == 'XSHG':
                data_em['IssueCode'] = "SH" + row[1]
            else:
                data_em['IssueCode'] = row[1]

            data_em['Yield'] = row[2]
            data_em['NetPrice'] = row[3]
            data_em['FullPrice'] = row[4]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataBondEval = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueBondEval, self.dataBondEval)

    def setValueBondEval(self, data_em):
        data_em['DataDate'] = self.time['PreTradingDate']


