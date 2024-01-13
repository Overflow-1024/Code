# coding=utf-8
from Model.Base import DataModel
import datetime
import re
import copy

class ExchangeBondInfo(DataModel):

    def __init__(self):
        super(ExchangeBondInfo, self).__init__()

        self.dataBondInfo = []
        self.dataHisBondInfo = []
        self.dataIssueMaster = []
        self.dataIssueMarket = []
        self.dataHistoricalPrice = []

        self.fieldSource = [
            'I_CODE',
            'M_TYPE',
            'B_NAME',
            'WIND_CLASS1',
            'B_PAR_VALUE',
            'B_MTR_DATE',
            'B_TERM',
            'B_COUPON_TYPE',
            'B_CASH_TIMES',
            'B_DAYCOUNT',
            'B_START_DATE',
            'B_COUPON',
            'CURRENCY',
            'ISSUER_CODE',
            'B_ISSUE_PRICE',
            'B_LIST_DATE',
            'B_DELIST_DATE',
            'HOST_MARKET'
        ]
        self.fieldBondInfo = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FixedCouponRate',
            'SettlCurrency',
            'IssuerShortPartyID',
            'IssuerPrice',
            'CustodianName'
        ]
        self.fieldHisBondInfo = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FixedCouponRate',
            'SettlCurrency',
            'IssuerShortPartyID',
            'IssuerPrice',
            'CustodianName',
            'DataDate'
        ]
        self.fieldIssueMaster = [
            'IssueCode',
            'IssueShortName',
            'ProductCode',
            'Currency',
            'ExpirationDate',
            'FaceValue',
            'PriorMarket',
            'UnderlyingAssetCode',
            'ContractSize'
        ]
        self.fieldIssueMarket = [
            'IssueCode',
            'MarketCode',
            'ListedDate'
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
            'MMBestBid1'
        ]
        self.fieldCheck = [
            'IssueName',    # 债券名称
            'BondType',     # 债券类型
            'ExpirationDate',   # 最后交易日期
            'TradeLimitDays',   # 债券期限
            'AI',               # 百元应计利息
            'DurationString',   # 代偿期（下单标准格式）
            'CouponType',   # 息票类型
            'CouponFrequency',  # 付息频率
            'FirstValueDate',   # 起息日
            'FixedCouponRate',  # 固定利率
            'CustodianName',    # 托管机构名称
            'IssuerShortPartyID',   # 发行人6位机构码
            'AccrualBasis',         # 计息基准
            'IssuerPrice',      # 发行价
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',    # 债券代码
            'MarketCode'    # 市场代码
        ]

    def setData(self, res, args=None):

        self.data = []
        self.dataBondInfo = []
        self.dataHisBondInfo = []
        self.dataIssueMaster = []
        self.dataIssueMarket = []
        self.dataHistoricalPrice = []

        def setDataElement(row):
            data_em = dict()

            data_em['ReportCode'] = row[0]
            zhmodel = re.compile(u'[\u4e00-\u9fa5]')
            if not zhmodel.search(data_em['ReportCode']):
                data_em['MarketCode'] = '0'
                data_em['IssueCode'] = row[0]
                if row[1] == 'XSHG':
                    data_em['MarketCode'] = '1'
                    data_em['IssueCode'] = 'SH' + row[0]
                elif row[1] == 'XSHE':
                    data_em['MarketCode'] = '2'
                    data_em['IssueCode'] = 'SZ' + row[0]
                data_em['IssueName'] = row[2]
                # productCode='11'
                data_em['BondType'] = row[3]
                data_em['FaceValue'] = row[4]
                data_em['ExpirationDate'] = row[5][0:4] + row[5][5:7] + row[5][8:10]
                data_em['FirstValueDate'] = row[10][0:4] + row[10][5:7] + row[10][8:10]
                data_em['TradeLimitDays'] = row[6]
                couponTypedict = {'1': '固定利率', '2': '浮动利率', '3': '零息票利率'}
                data_em['CouponType'] = couponTypedict[row[7]]
                data_em['CouponFrequency'] = row[8]
                data_em['AccrualBasis'] = row[9]
                data_em['FixedCouponRate'] = row[11] * 100
                data_em['SettlCurrency'] = row[12]
                data_em['IssueShortPartyID'] = row[13]
                data_em['IssuerPrice'] = row[14]
                try:
                    data_em['ListingDate'] = row[15][0:4] + row[15][5:7] + row[15][8:10]  # 上市日期
                except:
                    data_em['ListingDate'] = None
                try:
                    data_em['DelistingDate'] = row[16][0:4] + row[16][5:7] + row[16][8:10]  # 摘牌日期
                except:
                    data_em['DelistingDate'] = None
                data_em['CustodianName'] = row[17]

                today = datetime.date.today()
                d2 = datetime.date(int(row[5][0:4]), int(row[5][5:7]), int(row[5][8:10]))
                data_em['Duration'] = (d2 - today).days

            return data_em

        self.data = list(map(setDataElement, res))

        # 时间
        self.time['CurrentDate'] = args.get('CurrentDate')
        self.time['PreTradingDate'] = args.get('PreTradingDate')

        # 复制给各个数据表对应的字典
        self.dataBondInfo = copy.deepcopy(self.data)
        self.dataHisBondInfo = copy.deepcopy(self.data)
        self.dataIssueMaster = copy.deepcopy(self.data)
        self.dataIssueMarket = copy.deepcopy(self.data)
        self.dataHistoricalPrice = copy.deepcopy(self.data)

    def setDefaultValue(self):

        map(self.setValueBondInfo, self.dataBondInfo)
        map(self.setValueHisBondInfo, self.dataHisBondInfo)
        map(self.setValueIssueMaster, self.dataIssueMaster)
        map(self.setValueIssueMarket, self.dataIssueMarket)
        map(self.setValueHistoricalPrice, self.dataHistoricalPrice)

    def setValueBondInfo(self, data_em):

        data_em['ProductCode'] = 11

    def setValueHisBondInfo(self, data_em):

        data_em['ProductCode'] = 11
        data_em['DataDate'] = self.time['CurrentDate']

    def setValueIssueMaster(self, data_em):

        data_em['ProductCode'] = 11
        data_em['UnderlyingAssetCode'] = 0
        data_em['ContractSize'] = 1

    def setValueIssueMarket(self, data_em):

        data_em['ListedDate'] = data_em['ListingDate']

    def setValueHistoricalPrice(self, data_em):

        data_em['DataDate'] = self.time['PreTradingDate']
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
