# coding=utf-8
from Model.Base import DataModel
import datetime
import re
import copy


# 债券基础信息
class BondInfo(DataModel):

    def __init__(self):
        super(BondInfo, self).__init__()

        self.dataBondInfo = []
        self.dataHisBondInfo = []
        self.dataIssueMaster = []
        self.dataIssueMarket = []
        self.dataHistoricalPrice = []

        self.fieldSource = [
            'I_CODE',
            'I_NAME',
            'BOND_TYPE',
            'FACE_VALUE',
            'MATURITY_DATE',
            'BOND_TERM',
            'DURATION',
            'AI',
            'CR_1',
            'COUPON_TYPE',
            'COUPON_FREQUENCY',
            'FIRST_VALUE_DATE',
            'FIRST_PAYMENT_DATE',
            'FIXED_COUPON_RATE',
            'EXERCISE_TYPE_1',
            'SETTLCURRENCY',
            'CUSTODIAN_NAME',
            'CURRENT_COUPON_RATE',
            'DELISTING_DATE',
            'SECURITYTYPEID',
            'TERMTOMATURITYSTRING',
            'ISSUERSHORTPARTYID',
            'ACCRUAL_BASIS',
            'ISSUE_PRICE',
            'LISTING_DATE'
        ]
        self.fieldBondInfo = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'BondTypeID',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'DurationString',
            'AI',
            'CR',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FirstPaymentDate',
            'FixedCouponRate',
            'ExerciseType1',
            'SettlCurrency',
            'CustodianName',
            'IssuerShortPartyID',
            'IssuerPrice',
            'TheoryPrice',
            'ClearingSpeed'
        ]
        self.fieldHisBondInfo = [
            'IssueCode',
            'MarketCode',
            'IssueName',
            'ReportCode',
            'ProductCode',
            'BondType',
            'BondTypeID',
            'FaceValue',
            'ExpirationDate',
            'ListingDate',
            'DelistingDate',
            'TradeLimitDays',
            'Duration',
            'DurationString',
            'AI',
            'CR',
            'CouponType',
            'CouponFrequency',
            'AccrualBasis',
            'FirstValueDate',
            'FirstPaymentDate',
            'FixedCouponRate',
            'ExerciseType1',
            'SettlCurrency',
            'CustodianName',
            'IssuerShortPartyID',
            'IssuerPrice',
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
            'IssueName',  # 债券名称
            'BondType',  # 债券类型
            'BondTypeID',  # 债券类型ID
            'ExpirationDate',  # 最后交易日期
            'CouponType',  # 息票类型
            'CouponFrequency',  # 付息频率
            'FirstValueDate',  # 起息日
            'FixedCouponRate',  # 固定利率
            'CustodianName',  # 托管机构名称
            'IssuerShortPartyID',  # 发行人6位机构码
            'IssuerPrice',  # 发行价
        ]
        # 两表比对时用于联合的主键
        self.fieldKeys = [
            'IssueCode',  # 债券代码
            'MarketCode'  # 市场代码
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

                data_em['IssueCode'] = 'IB' + row[0]
                data_em['IssueName'] = row[1]
                data_em['BondType'] = row[2]
                data_em['BondTypeID'] = row[19]
                data_em['FaceValue'] = row[3]
                data_em['ExpirationDate'] = row[4][0:4] + row[4][5:7] + row[4][8:10]
                data_em['DelistingDate'] = row[18][0:4] + row[18][5:7] + row[18][8:10]
                data_em['TradeLimitDays'] = row[5]
                data_em['Duration'] = row[6]
                data_em['DurationString'] = str(row[20]).replace("365D", "1Y")
                data_em['AI'] = str(row[7])
                data_em['CR'] = row[8]
                data_em['CouponType'] = row[9]
                data_em['CouponFrequency'] = row[10]
                data_em['FirstValueDate'] = row[11][0:4] + row[11][5:7] + row[11][8:10]
                data_em['FirstPaymentDate'] = row[12][0:4] + row[12][5:7] + row[12][8:10]
                data_em['FixedCouponRate'] = row[13]
                data_em['ExerciseType1'] = row[14]
                data_em['SettlCurrency'] = row[15]
                data_em['CustodianName'] = row[16]
                data_em['IssuerShortPartyID'] = row[21]
                data_em['AccrualBasis'] = row[22]
                data_em['IssuerPrice'] = row[23]
                data_em['ListingDate'] = row[24][0:4] + row[24][5:7] + row[24][8:10]

                currentCouponRate = row[17]

                if str(data_em['ExerciseType1']) == 'None':
                    data_em['ExerciseType1'] = ''
                if str(data_em['AI']) == 'None':
                    data_em['AI'] = 0.00
                if str(data_em['CR']) == 'None':
                    data_em['CR'] = ''
                if str(data_em['FixedCouponRate']) == 'None':
                    data_em['FixedCouponRate'] = 0.00
                if str(currentCouponRate) == 'None':
                    currentCouponRate = 0.00

                if data_em['CouponType'] == '浮动利率':
                    data_em['FixedCouponRate'] = currentCouponRate

                today = datetime.date.today()
                d2 = datetime.date(int(row[4][0:4]), int(row[4][5:7]), int(row[4][8:10]))
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

        data_em['MarketCode'] = '9'
        data_em['ProductCode'] = '40'
        data_em['TheoryPrice'] = 0
        data_em['ClearingSpeed'] = 1

    def setValueHisBondInfo(self, data_em):

        data_em['MarketCode'] = '9'
        data_em['ProductCode'] = '40'
        data_em['DataDate'] = self.time['CurrentDate']

    def setValueIssueMaster(self, data_em):

        data_em['IssueShortName'] = data_em['ReportCode']
        data_em['IssueShortLocalName'] = data_em['IssueName']
        data_em['IssueLongName'] = data_em['ReportCode']
        data_em['IssueLongLocalName'] = data_em['IssueName']

        data_em['ProductCode'] = '40'
        data_em['Currency'] = data_em['SettlCurrency']
        data_em['MarketSectorCode'] = None
        data_em['PriorMarket'] = '9'
        data_em['UnderlyingAssetCode'] = '0'
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
        data_em['ListedDate'] = data_em['ListingDate']
        data_em['MarketSectionCode'] = '0'
        data_em['SessionPatternID'] = None
        data_em['ReserveString'] = '40'
        data_em['CreateTime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['TimeStamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_em['DTSTimeStamp'] = 0
        data_em['ApplicationStopFlag'] = None

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