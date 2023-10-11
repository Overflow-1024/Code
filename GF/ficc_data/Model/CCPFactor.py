# coding=utf-8
from Model.Base import DataModel
import datetime
import copy
import re


class CCPFactor(DataModel):

    def __init__(self):
        super(CCPFactor, self).__init__()

        self.dataIssueDelivery = []

        self.fieldSource = [
            'i_code',
            'd_i_code',
            'ctd_cf',
            'beg_date'
        ]
        self.fieldIssueDelivery = [
            'CF',
            'UpdateDate',
            'IssueCode',
            'BondCode'
        ]
        self.fieldCheck = [
            'CF'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'BondCode'
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataIssueDelivery = []

        def setDataElement(row):
            data_em = dict()

            data_em['IssueCode'] = "IB" + row[0]  # CCP代码
            data_em['BondCode'] = "IB" + row[1]  # CBT代码
            data_em['CF'] = row[2]              # 转换因子

            data_em['CF'] = format(data_em['CF'], '.6f')

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataIssueDelivery = copy.deepcopy(self.data)


    def setDefaultValue(self):
        map(self.setValueIssueDelivery, self.dataIssueDelivery)


    def setValueIssueDelivery(self, data_em):
        data_em['UpdateDate'] = self.time['CurrentDate']



