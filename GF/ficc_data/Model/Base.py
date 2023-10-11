# coding=utf-8
from abc import abstractmethod


class DataModel(object):

    def __init__(self):

        self.data = []
        self.time = {}
        self.fieldKeys = []
        self.fieldCheck = []

    # 通过字段列表取值，返回值的列表
    def getDataByField(self, data, fields):
        def getDataElement(data_em):
            values_em = []
            for field in fields:
                values_em.append(data_em.get(field))
            return tuple(values_em)

        return list(map(getDataElement, data))

    def generateDataSQL(self, data, fields):
        fields_str = ','.join(fields)
        placeholder = ','.join(['%s'] * len(fields))
        values = self.getDataByField(data, fields)

        return fields_str, placeholder, values

    def generateCheckSQL(self):
        # 给字段加上前缀 a b （用于两表联查）
        # a 作为当天 b 作为昨天
        keys_a = ["a." + item for item in self.fieldKeys]
        keys_b = ["b." + item for item in self.fieldKeys]
        fields_a = ["a." + item for item in self.fieldCheck]
        fields_b = ["b." + item for item in self.fieldCheck]
        fields_double = keys_a + fields_a + fields_b

        keys_str = ','.join(self.fieldKeys)
        fields_str = ','.join(self.fieldCheck)
        fields_double_str = ','.join(fields_double)

        # 生成sql字符串
        join_cond_list = []
        for i in range(len(self.fieldKeys)):
            cond = keys_a[i] + '=' + keys_b[i]
            join_cond_list.append(cond)
        join_condition = ' AND '.join(join_cond_list)

        check_cond_list = []
        for i in range(len(self.fieldCheck)):
            cond = fields_a[i] + '=' + fields_b[i]
            check_cond_list.append(cond)
        check_condition = ' AND '.join(check_cond_list)

        return keys_str, fields_str, fields_double, fields_double_str, join_condition, check_condition





