CREATE TABLE `HisBondCoversionTable_CFETS` (
	`IssueCode` VARCHAR(32) NOT NULL COMMENT '债券代码' COLLATE 'latin1_general_cs',
	`MarketCode` VARCHAR(2) NOT NULL COMMENT '市场代码' COLLATE 'latin1_general_cs',
	`IssueName` VARCHAR(200) NOT NULL COMMENT '债券名称' COLLATE 'latin1_general_cs',
	`ConvDate` VARCHAR(20) NOT NULL COMMENT '转股日期' COLLATE 'latin1_general_cs',
	`ConvCode` VARCHAR(50) NOT NULL DEFAULT '0' COMMENT '转股代码' COLLATE 'latin1_general_cs',
	`ConvPrice` DOUBLE(22,4) NOT NULL COMMENT '转股价格',
	`BondType` VARCHAR(50) NOT NULL COMMENT '债券分类' COLLATE 'latin1_general_cs',
	`BondExtredType` VARCHAR(50) NOT NULL COMMENT '债券扩展分类' COLLATE 'latin1_general_cs',
	`UpdateTime` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
	`DataDate` VARCHAR(10) NOT NULL COMMENT '日期' COLLATE 'latin1_general_cs',
	PRIMARY KEY (`IssueCode`, `MarketCode`, `DataDate`) USING BTREE
)
COLLATE='latin1_general_cs'
ENGINE=MyISAM
ROW_FORMAT=DYNAMIC
;

CREATE TABLE `HisBondCodeInMarket_CFETS` (
	`IssueCode` VARCHAR(32) NOT NULL COMMENT '债券代码' COLLATE 'latin1_general_cs',
	`MarketCode` VARCHAR(2) NOT NULL COMMENT '市场代码' COLLATE 'latin1_general_cs',
	`SH_Code` VARCHAR(32) NOT NULL COMMENT '上海代码' COLLATE 'latin1_general_cs',
	`SZ_Code` VARCHAR(32) NOT NULL COMMENT '深圳代码' COLLATE 'latin1_general_cs',
	`YH_Code` VARCHAR(32) NOT NULL COMMENT '银行间代码' COLLATE 'latin1_general_cs',
	`AssetType` VARCHAR(50) NOT NULL COMMENT '资产类型' COLLATE 'latin1_general_cs',
	`DataDate` VARCHAR(10) NOT NULL COLLATE 'latin1_general_cs',
	PRIMARY KEY (`IssueCode`, `MarketCode`, `DataDate`) USING BTREE
)
COLLATE='latin1_general_cs'
ENGINE=MyISAM
ROW_FORMAT=DYNAMIC
;

CREATE TABLE `HisIssueMasterTable_CFETS` (
	`IssueCode` VARCHAR(32) NOT NULL COMMENT '合约内码,IB+申报代码' COLLATE 'latin1_general_cs',
	`MarketCode` VARCHAR(2) NOT NULL COMMENT '市场代码,9:银行间' COLLATE 'latin1_general_cs',
	`ReportCode` VARCHAR(30) NOT NULL COMMENT '申报代码' COLLATE 'latin1_general_cs',
	`IssueName` VARCHAR(200) NOT NULL COMMENT '合约名称' COLLATE 'latin1_general_cs',
	`ProductCode` VARCHAR(2) NOT NULL COMMENT '产品代码' COLLATE 'latin1_general_cs',
	`UnderlyingIssueCode` VARCHAR(20) NOT NULL COMMENT '元资产代码' COLLATE 'latin1_general_cs',
	`FaceValue` DOUBLE NULL DEFAULT NULL COMMENT '面值,单位元',
	`MinQuantity` DOUBLE NULL DEFAULT NULL COMMENT '最小单笔报价量(手)',
	`MaxQuantity` DOUBLE NULL DEFAULT NULL COMMENT '最大单笔报价量(手)',
	`ListDate` VARCHAR(8) NULL DEFAULT NULL COMMENT '上市日期' COLLATE 'latin1_general_cs',
	`ExpirationDate` VARCHAR(8) NULL DEFAULT NULL COMMENT '最后交易日' COLLATE 'latin1_general_cs',
	`ExpirationTime` VARCHAR(20) NULL DEFAULT NULL COMMENT '最后交易时间' COLLATE 'latin1_general_cs',
	`SettleDate` VARCHAR(8) NULL DEFAULT NULL COMMENT '交割日' COLLATE 'latin1_general_cs',
	`Tick` DOUBLE NULL DEFAULT NULL COMMENT '最小变动单位',
	`ContractSize` INT(11) NULL DEFAULT NULL COMMENT '合约乘数',
	`Status` VARCHAR(20) NULL DEFAULT NULL COMMENT '合约状态' COLLATE 'latin1_general_cs',
	`ClearingMethod` VARCHAR(50) NULL DEFAULT NULL COMMENT '清算方式' COLLATE 'latin1_general_cs',
	`BenchMarkPrice` DOUBLE NULL DEFAULT NULL COMMENT '挂牌基准价',
	`CreateTime` DATETIME NULL DEFAULT NULL COMMENT '创建时间',
	`UpdateTime` DATETIME NULL DEFAULT NULL COMMENT '更新时间',
	`DataDate` VARCHAR(10) NOT NULL COLLATE 'latin1_general_cs',
	PRIMARY KEY (`IssueCode`, `MarketCode`, `DataDate`) USING BTREE,
	UNIQUE INDEX `IssueMasterTable_CFETS_idx01` (`IssueCode`, `MarketCode`, `DataDate`) USING BTREE
)
COLLATE='latin1_general_cs'
ENGINE=MyISAM
ROW_FORMAT=DYNAMIC
;

CREATE TABLE `HisTradeMemberTable_CFETS` (
	`MEMBER_ID` VARCHAR(50) NOT NULL COLLATE 'latin1_general_cs',
	`ORGCODE` VARCHAR(50) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`CH_NAME` VARCHAR(600) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`CH_SHORT_NAME` VARCHAR(200) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`EN_NAME` VARCHAR(600) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`EN_SHORT_NAME` VARCHAR(200) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`KIND_CODE` VARCHAR(100) NULL DEFAULT '' COLLATE 'latin1_general_cs',
	`KIND_NAME` VARCHAR(100) NULL DEFAULT '' COLLATE 'latin1_general_cs',
	`PARTYID` VARCHAR(100) NULL DEFAULT '' COLLATE 'latin1_general_cs',
	`PARENT_PARTYID` VARCHAR(100) NULL DEFAULT '' COLLATE 'latin1_general_cs',
	`TimeStamp` DATETIME NULL DEFAULT NULL,
	`DataDate` VARCHAR(10) NOT NULL COLLATE 'latin1_general_cs',
	PRIMARY KEY (`MEMBER_ID`, `DataDate`) USING BTREE
)
COLLATE='latin1_general_cs'
ENGINE=MyISAM
ROW_FORMAT=DYNAMIC
;

CREATE TABLE `HisCalendarTable_CFETS` (
	`DTSDate` VARCHAR(8) NOT NULL COMMENT '日期' COLLATE 'latin1_general_cs',
	`MarketCode` VARCHAR(2) NOT NULL COMMENT '市场号' COLLATE 'latin1_general_cs',
	`BondTrade` VARCHAR(1) NULL DEFAULT NULL COMMENT '债券交易日标志' COLLATE 'latin1_general_cs',
	`BondSettle` VARCHAR(1) NULL DEFAULT NULL COMMENT '债券结算日标志' COLLATE 'latin1_general_cs',
	`IRSTrade` VARCHAR(1) NULL DEFAULT NULL COMMENT 'IRS交易日标志' COLLATE 'latin1_general_cs',
	`CreateTime` DATETIME NULL DEFAULT NULL COMMENT '创建时间',
	`TimeStamp` DATETIME NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
	`DayOffFlag` VARCHAR(1) NULL DEFAULT NULL COLLATE 'latin1_general_cs',
	`DataDate` VARCHAR(10) NOT NULL COMMENT '日期' COLLATE 'latin1_general_cs',
	PRIMARY KEY (`DTSDate`, `MarketCode`, `DataDate`) USING BTREE
)
COLLATE='latin1_general_cs'
ENGINE=MyISAM
ROW_FORMAT=DYNAMIC
;
