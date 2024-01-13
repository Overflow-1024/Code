# coding=utf-8
import datetime
from Model.Base import DataModel
import copy


class TradeParty(DataModel):

    def __init__(self):
        super(TradeParty, self).__init__()

        self.dataTradeParty = []

        self.fieldSource = [
            'PARTYID',
            'PARTYNAME',
            'PARTYNAME_SHORT'
        ]
        self.fieldTradeParty = [
            'PARTY_ID',
            'CH_NAME',
            'CH_SHORT_NAME',
            'TimeStamp'
        ]
        self.fieldCheck = [

        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'PARTY_ID'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataTradeParty = []

        def setDataElement(row):
            data_em = dict()

            data_em['PARTY_ID'] = row[0]
            data_em['CH_NAME'] = row[1]
            data_em['CH_SHORT_NAME'] = row[2]

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataTradeParty = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueTradeParty, self.dataTradeParty)

    def setValueTradeParty(self, data_em):
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")