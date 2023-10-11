# -*- coding: utf-8 -*-
import sys
import os
import time
import re
import Database


class GlobalManager(object):

    def __init__(self):
        # 先读取配置文件
        configFile = open(sys.path[0] + '\Sync_CFETS.config', 'r')
        configText = configFile.readlines()

        self.gCurrentDate = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.GlobalSettingTable = {}

        for i in range(0, len(configText)):
            text = (configText[i].rstrip('\n')).strip()
            if len(text) > 0 and text[0:1] != "#":
                key = ((re.split('[=]+', text))[0]).strip()
                value = ((re.split('[=]+', text))[1]).strip()
                self.GlobalSettingTable[key] = value

        path = self.GlobalSettingTable['Log_Path']
        if path[-1:] == '\\':
            path = path[:-1]
        filePath = "%s\Sync_CFETS_%s.log" % (path, self.gCurrentDate)
        errorFilePath = "%s\Sync_CFETS_Error_%s.log" % (path, self.gCurrentDate)

        self.file = open(filePath, 'a+')
        self.errorFile = open(errorFilePath, 'a+')

        nowTime = time.strftime('%H:%M:%S', time.localtime(time.time()))
        self.file.write(nowTime + "    " + 'gCurrentDate:' + self.gCurrentDate + "\n")

        # 设置Oracle的字符集
        os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
        # 连接Oracle数据库
        oracle_args = {
            'host': self.GlobalSettingTable['Oracle_IP'],
            'port': self.GlobalSettingTable['Oracle_Port'],
            'user': self.GlobalSettingTable['Oracle_User'],
            'password': self.GlobalSettingTable['Oracle_PassWord'],
            'db': self.GlobalSettingTable['Oracle_DB']
        }
        self.oracle = Database.OracleManager(oracle_args)
        self.oracle.connect()

        # 连接MySQL数据库
        mysql_args = {
            'host': self.GlobalSettingTable['MySQL_IP'],
            'user': self.GlobalSettingTable['MySQL_User'],
            'password': self.GlobalSettingTable['MySQL_PassWord'],
            'db': self.GlobalSettingTable['MySQL_DB'],
            'charset': 'latin1'
        }
        self.mysql = Database.MysqlManager(mysql_args)
        self.mysql.connect()

        self.mysql_db = self.GlobalSettingTable['MySQL_DB']

    def __del__(self):
        # 关闭连接
        self.oracle.close()
        self.mysql.close()

        self.file.close()
        self.errorFile.close()


context = GlobalManager()

