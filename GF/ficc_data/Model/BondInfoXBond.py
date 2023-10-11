# coding=utf-8
from Model.Base import DataModel
import copy

# 债券基础信息
class BondInfoXBond(DataModel):

    def __init__(self):
        super(BondInfoXBond, self).__init__()

        self.dataBondInfoXBond = []

        self.fieldSource = [
            'security_id',
            'symbol',
            'security_desc',
            'centraquotebondindic',
            'securitytypeid',
            'issuershortpartyid',
            'termtomaturitystring'
        ]
        self.fieldBondInfoXBond = [
            'IssueCode',
            'IssueName',
            'BondType',
            'BondTypeID',
            'CenterQuote',
            'IssuerShortPartyID',
            'DurationString',
            'AutoSubscribe',
            'DataDate'
        ]
        self.fieldCheck = []
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode'    # 债券代码
        ]


    def setData(self, res, args=None):
        self.data = []
        self.dataBondInfoXBond = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = "IB" + row[0]
            data_em['IssueName'] = row[1]
            data_em['BondType'] = row[2]
            data_em['CenterQuote'] = row[3]
            data_em['BondTypeID'] = row[4]
            data_em['IssuerShortPartyID'] = row[5]
            data_em['DurationString'] = row[6]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataBondInfoXBond = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueBondInfoXBond, self.dataBondInfoXBond)

    def setValueBondInfoXBond(self, data_em):
        data_em['AutoSubscribe'] = 0
        data_em['DataDate'] = self.time['CurrentDate']


