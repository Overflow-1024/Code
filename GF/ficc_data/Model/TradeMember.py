# coding=utf-8
import datetime
from Model.Base import DataModel
import copy


class TradeMember(DataModel):

    def __init__(self):
        super(TradeMember, self).__init__()

        self.dataTradeMember = []
        self.dataHisTradeMember = []

        self.fieldSource = [
            't.MEMBER_ID',
            't.ORGCODE',
            't.CH_NAME',
            't.CH_SHORT_NAME',
            't.EN_NAME',
            'EN_SHORT_NAME',
            't1.T_CODE',
            't1.T_NAME'
        ]
        self.fieldTradeMember = [
            'MEMBER_ID',
            'ORGCODE',
            'CH_NAME',
            'CH_SHORT_NAME',
            'EN_NAME',
            'EN_SHORT_NAME',
            'KIND_CODE',
            'KIND_NAME',
            'TimeStamp'
        ]
        self.fieldHisTradeMember = [
            'MEMBER_ID',
            'ORGCODE',
            'CH_NAME',
            'CH_SHORT_NAME',
            'EN_NAME',
            'EN_SHORT_NAME',
            'KIND_CODE',
            'KIND_NAME',
            'TimeStamp',
            'DataDate'
        ]

        self.fieldCheck = [
            'CH_SHORT_NAME',
            'KIND_CODE'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'MEMBER_ID'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataTradeMember = []
        self.dataHisTradeMember = []

        def setDataElement(row):
            data_em = dict()

            data_em['MEMBER_ID'] = row[0]
            data_em['ORGCODE'] = row[1]
            data_em['CH_NAME'] = row[2]
            data_em['CH_SHORT_NAME'] = row[3]
            data_em['EN_NAME'] = row[4]
            data_em['EN_SHORT_NAME'] = row[5]
            data_em['KIND_CODE'] = row[6]
            data_em['KIND_NAME'] = row[7]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataTradeMember = copy.deepcopy(self.data)
        self.dataHisTradeMember = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueTradeMember, self.dataTradeMember)
        map(self.setValueHisTradeMember, self.dataHisTradeMember)


    def setValueTradeMember(self, data_em):
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def setValueHisTradeMember(self, data_em):

        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DataDate'] = self.time['CurrentDate']




