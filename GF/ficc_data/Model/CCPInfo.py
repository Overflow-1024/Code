# coding=utf-8
from Model.Base import DataModel
import datetime
import copy
import re


class CCPInfo(DataModel):

    def __init__(self):
        super(CCPInfo, self).__init__()

        self.dataIssueMasterCFETS = []
        self.dataHisIssueMasterCFETS = []
        self.dataIssueMaster = []
        self.dataIssueMarket = []
        self.dataHistoricalPrice = []

        self.fieldSource = [
            'BENCH_MARK_PRICE',
            'ISSUE_DATE',
            'I_CODE',
            'I_NAME',
            'LAST_TRD_DATE',
            'CLEARING_METHOD',
            'TRAD_SES_END_TIME',
            'SETTLDATE',
            'FACEVALUE',
            'SECURITY_STATUS',
            'MINQTY',
            'MAXQTY'
        ]
        self.fieldIssueMasterCFETS = [
            'IssueCode',
            'MarketCode',
            'ReportCode',
            'IssueName',
            'ProductCode',
            'UnderlyingIssueCode',
            'FaceValue',
            'MinQuantity',
            'MaxQuantity',
            'ListDate',
            'ExpirationDate',
            'ExpirationTime',
            'SettleDate',
            'Tick',
            'ContractSize',
            'Status',
            'ClearingMethod',
            'BenchMarkPrice',
            'CreateTime',
            'UpdateTime'
        ]
        self.fieldHisIssueMasterCFETS = [
            'IssueCode',
            'MarketCode',
            'ReportCode',
            'IssueName',
            'ProductCode',
            'UnderlyingIssueCode',
            'FaceValue',
            'MinQuantity',
            'MaxQuantity',
            'ListDate',
            'ExpirationDate',
            'ExpirationTime',
            'SettleDate',
            'Tick',
            'ContractSize',
            'Status',
            'ClearingMethod',
            'BenchMarkPrice',
            'CreateTime',
            'UpdateTime',
            'DataDate'
        ]
        self.fieldIssueMaster = [
            'IssueCode',
            'IssueShortName',
            'IssueShortLocalName',
            'IssueLongName',
            'IssueLongLocalName',

            'ProductCode',
            'Currency',
            'MarketSectorCode',
            'PriorMarket',
            'UnderlyingAssetCode',

            'PutCall',
            'ContractMonth',
            'OtherContractMonth',
            'StrikePrice',
            'ExpirationDate',

            'FaceValue',
            'EstimateFaceValue',
            'NearMonthIssueCode',
            'OtherMonthIssueCode',
            'GrantedRatio',

            'UnderlyingIssueCode',
            'ExRightsDate',
            'ReserveString',
            'CreateTime',
            'TimeStamp',

            'DTSTimeStamp',
            'Shares',
            'Status',
            'UnitAmount',
            'AmountLeast',

            'AmountMost',
            'BalanceMost',
            'Tick',
            'RaisingLimitRate',
            'DecliningLimitRate',

            'FareRule',
            'PromptRule',
            'TradingMonth',
            'OpenDropFareRatio',
            'OpenDropFareBalance',

            'DropCuFareRatio',
            'DropCuFareBalance',
            'DeliverFareRatio',
            'DeliverFareBalance',
            'SpeculationBailRatio',

            'SpeculationBailBalance',
            'HedgeBailRatio',
            'HedgeBailBalance',
            'ContractSize'
        ]
        self.fieldIssueMarket = [
            'IssueCode',
            'MarketCode',
            'MarketSystemCode',
            'EvenLot',
            'CashDeliveryFlag',
            'LoanableIssueFlag',
            'MarginIssueFlag',
            'ListedDate',
            'MarketSectionCode',
            'SessionPatternID',
            'ReserveString',
            'CreateTime',
            'TimeStamp',
            'DTSTimeStamp',
            'ApplicationStopFlag'
        ]
        self.fieldHistoricalPrice = [
            'IssueCode',
            'MarketCode',
            'DataDate',
            'MarkPrice',
            'ClosePrice',
            'AdjustedClosePrice',
            'OpenPrice',
            'HighPrice',
            'LowPrice',
            'Volume',
            'UpperLimitPrice',
            'LowerLimitPrice',
            'MMLNBestBid',
            'ReserveString',
            'CreateTime',
            'TimeStamp',
            'DTSTimeStamp',
            'WeeklyHighPrice',
            'WeeklyLowPrice',
            'MonthlyHighPrice',
            'MonthlyLowPrice',
            'QuarterHighPrice',
            'QuarterLowPrice',
            'Psychological',
            'WeightGiftCounter',
            'WeightSellCounter',
            'WeightSellPrice',
            'WeightDividend',
            'WeightIncCounter',
            'WeightOwnerShip',
            'WeightFreeCounter',
            'ClearingPrice',
            'MMBestBid1',
        ]
        self.fieldCheck = [
            'IssueName',
            'ListDate',
            'ExpirationDate',
            'ExpirationTime',
            'SettleDate',
            'FaceValue',
            'Status'
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',
            'MarketCode'
        ]
        # 合约列表（时间占四个字符，先用0000作为占位符）
        self.contractTemplate = [
            'IBCDB2_0000P',
            'IBADBC7_0000P',
            'IBADBC2_0000P',
            'IBCDB5_0000',
            'IBCDB3_0000',
            'IBCDB10_0000',
            'IBADBC5_0000',
            'IBADBC10_0000'
        ]


    def setData(self, res, args=None):

        self.data = []
        self.dataIssueMasterCFETS = []
        self.dataHisIssueMasterCFETS = []
        self.dataIssueMaster = []
        self.dataIssueMarket = []
        self.dataHistoricalPrice = []

        def setDataElement(row):
            data_em = dict()

            data_em['BenchMarkPrice'] = row[0]  # 挂牌基准价
            data_em['ListDate'] = row[1][0:4] + row[1][5:7] + row[1][8:10]  # 上市日期
            data_em['ReportCode'] = row[2]  # 代码
            data_em['IssueCode'] = "IB" + str(data_em['ReportCode'])
            data_em['IssueName'] = row[3]  # 名称
            data_em['ExpirationDate'] = row[4][0:4] + row[4][5:7] + row[4][8:10]  # 到期日
            data_em['ClearingMethod'] = row[5]  # 清算方式
            data_em['ExpirationTime'] = row[6][9:11] + row[6][12:14] + row[6][15:17]  # 到期时间
            data_em['SettleDate'] = row[7][0:4] + row[7][5:7] + row[7][8:10]  # 交割日
            data_em['FaceValue'] = int(row[8]) * 10000  # 合约面值（万元）
            data_em['Status'] = row[9]  # 合约状态
            data_em['MinQuantity'] = int(row[10])  # 最小单笔报价量（手）
            data_em['MaxQuantity'] = int(row[11])  # 最大单笔报价量（手）
            data_em['UnderlyingIssueCode'] = (re.split('[_]+', str(data_em['ReportCode'])))[0]
            if data_em['ReportCode'][-1:] == 'P':
                data_em['UnderlyingIssueCode'] = str(data_em['UnderlyingIssueCode']) + 'P'

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['PreTradingDate'] = args.get('PreTradingDate')
        self.time['CurrentDate'] = args.get('CurrentDate')

        # 复制给各个数据表对应的字典
        self.dataIssueMasterCFETS = copy.deepcopy(self.data)
        self.dataHisIssueMasterCFETS = copy.deepcopy(self.data)
        self.dataIssueMaster = copy.deepcopy(self.data)
        self.dataIssueMarket = copy.deepcopy(self.data)
        self.dataHistoricalPrice = copy.deepcopy(self.data)

    def setDefaultValue(self):
        map(self.setValueIssueMasterCFETS, self.dataIssueMasterCFETS)
        map(self.setValueHisIssueMasterCFETS, self.dataHisIssueMasterCFETS)
        map(self.setValueIssueMaster, self.dataIssueMaster)
        map(self.setValueIssueMarket, self.dataIssueMarket)
        map(self.setValueHistoricalPrice, self.dataHistoricalPrice)

    def setValueIssueMasterCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['ProductCode'] = '38'
        data_em['Tick'] = 0.0001
        data_em['ContractSize'] = 1
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['UpdateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def setValueHisIssueMasterCFETS(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['ProductCode'] = '38'
        data_em['Tick'] = 0.0001
        data_em['ContractSize'] = 1
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['UpdateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DataDate'] = self.time['CurrentDate']

    def setValueIssueMaster(self, data_em):
        data_em['ProductCode'] = '38'
        data_em['Currency'] = 'RMB'
        data_em['MarketSectorCode'] = None
        data_em['PriorMarket'] = '9'
        data_em['UnderlyingAssetCode'] = 'CD'
        data_em['PutCall'] = None
        data_em['ContractMonth'] = ''
        data_em['OtherContractMonth'] = None
        data_em['StrikePrice'] = 0

        data_em['FaceValue'] = 0
        data_em['EstimateFaceValue'] = 0
        data_em['NearMonthIssueCode'] = None
        data_em['OtherMonthIssueCode'] = None
        data_em['GrantedRatio'] = 0

        data_em['UnderlyingIssueCode'] = ''
        data_em['ExRightsDate'] = None
        data_em['ReserveString'] = None
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data_em['DTSTimeStamp'] = 0
        data_em['Shares'] = 0
        data_em['Status'] = None
        data_em['UnitAmount'] = 0
        data_em['AmountLeast'] = 0

        data_em['AmountMost'] = 0
        data_em['BalanceMost'] = 0
        data_em['Tick'] = 0
        data_em['RaisingLimitRate'] = 0
        data_em['DecliningLimitRate'] = 0

        data_em['FareRule'] = None
        data_em['PromptRule'] = None
        data_em['TradingMonth'] = None
        data_em['OpenDropFareRatio'] = 0
        data_em['OpenDropFareBalance'] = 0

        data_em['DropCuFareRatio'] = 0
        data_em['DropCuFareBalance'] = 0
        data_em['DeliverFareRatio'] = 0
        data_em['DeliverFareBalance'] = 0
        data_em['SpeculationBailRatio'] = 0

        data_em['SpeculationBailBalance'] = 0
        data_em['HedgeBailRatio'] = 0
        data_em['HedgeBailBalance'] = 0
        data_em['ContractSize'] = 1

    def setValueIssueMarket(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['MarketSystemCode'] = ''
        data_em['EvenLot'] = 1
        data_em['CashDeliveryFlag'] = None
        data_em['LoanableIssueFlag'] = None
        data_em['MarginIssueFlag'] = None

        data_em['MarketSectionCode'] = '0'
        data_em['SessionPatternID'] = None
        data_em['ReserveString'] = '38'
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DTSTimeStamp'] = 0
        data_em['ApplicationStopFlag'] = None

        data_em['ListedDate'] = data_em['ListDate']

    def setValueHistoricalPrice(self, data_em):
        data_em['MarketCode'] = '9'
        data_em['MarkPrice'] = 0
        data_em['ClosePrice'] = 0

        data_em['AdjustedClosePrice'] = 0
        data_em['OpenPrice'] = 0
        data_em['HighPrice'] = 0
        data_em['LowPrice'] = 0
        data_em['Volume'] = 0

        data_em['UpperLimitPrice'] = 0
        data_em['LowerLimitPrice'] = 0
        data_em['MMLNBestBid'] = 0
        data_em['ReserveString'] = None
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DTSTimeStamp'] = 0
        data_em['WeeklyHighPrice'] = 0
        data_em['WeeklyLowPrice'] = 0
        data_em['MonthlyHighPrice'] = 0

        data_em['MonthlyLowPrice'] = 0
        data_em['QuarterHighPrice'] = 0
        data_em['QuarterLowPrice'] = 0
        data_em['Psychological'] = 0
        data_em['WeightGiftCounter'] = 0

        data_em['WeightSellCounter'] = 0
        data_em['WeightSellPrice'] = 0
        data_em['WeightDividend'] = 0
        data_em['WeightIncCounter'] = 0
        data_em['WeightOwnerShip'] = 0

        data_em['WeightFreeCounter'] = 0
        data_em['ClearingPrice'] = 0
        data_em['MMBestBid1'] = 0

        data_em['DataDate'] = self.time['PreTradingDate']



