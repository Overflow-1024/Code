# -*- coding: utf-8 -*-
import MySQLdb
import cx_Oracle


class MysqlManager:

    def __init__(self, args):
        self.host = args.get('host')
        self.user = args.get('user')
        self.password = args.get('password')
        self.db = args.get('db')
        self.charset = args.get('charset')

        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = MySQLdb.connect(self.host, self.user, self.password, self.db, charset=self.charset)
            self.cursor = self.conn.cursor()
            return True
        except MySQLdb.Error as e:
            print(e)
            return False

    def query(self, sql):
        try:
            self.cursor.execute(sql)
            return True
        except MySQLdb.Error as e:
            print(e)
            return False

    def update(self, sql, args=None):

        try:
            if args:
                self.cursor.executemany(sql, args=args)
            else:
                self.cursor.executemany(sql)
            self.conn.commit()
            return True
        except MySQLdb.Error as e:
            self.conn.rollback()
            print(e)
            return False

    def updateone(self, sql, args=None):

        try:
            if args:
                self.cursor.execute(sql, args=args)
            else:
                self.cursor.execute(sql)
            self.conn.commit()
            return True
        except MySQLdb.Error as e:
            self.conn.rollback()
            print(e)
            return False

    def delete(self, sql):

        try:
            self.cursor.execute(sql)
            return True
        except MySQLdb.Error as e:
            print(e)
            return False

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, n):
        return self.cursor.fetchmany(n)

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):

        self.cursor.close()
        self.conn.close()


class OracleManager:

    def __init__(self, args):
        self.host = args.get('host')
        self.port = args.get('port')
        self.user = args.get('user')
        self.password = args.get('password')
        self.db = args.get('db')

        self.conn = None
        self.cursor = None

    def connect(self):

        self.conn = cx_Oracle.connect('%s/%s@%s:%s/%s' % (self.user, self.password, self.host, self.port, self.db))
        self.cursor = self.conn.cursor()

    def query(self, sql):

        self.cursor.execute(sql)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, n):
        return self.cursor.fetchmany(n)

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):

        self.cursor.close()
        self.conn.close()
