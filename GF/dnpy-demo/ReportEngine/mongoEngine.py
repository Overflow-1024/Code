"""
Created on Thu Aug 08 09:22:12 2019
mongo数据库引擎
@author: Administrator
"""
# %%导入相关库
# 第三方库
import functools
import logging
import time
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure, \
    AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError
import pandas as pd  # 面板数据处理
from datetime import datetime, timedelta
from typing import Union

# 通用模块


DB_BACKUP = "DB_BACK_UP"
INIT_SOCKET_TIMEOUT = 90000


logger = logging.getLogger(__name__)


# %%装饰器
def auto_reconnect(mongo_method):
    """
    连接失败自动重连3次
    NetworkTimeout -> 由socketTimeout引发,重设并延长timeout值
    AutoReconnect -> 可能由网络问题引发,尝试重连
    OperationFailure -> 用户名密码验证失败
    """
    ATTEMPTS = 5
    @functools.wraps(mongo_method)
    def wrapper(*args, **kwargs):
        err = None
        instance: MongoManager = args[0]
        for attempt in range(1, ATTEMPTS+1):
            try:
                # if instance.API:
                #     instance._ping()
                return mongo_method(*args, **kwargs)
            except NetworkTimeout as e:
                print(f"网络错误：{str(e)}")
                print(f"{2 ** attempt}秒后自动尝试重连，剩余{ATTEMPTS - attempt}次")
                time.sleep(2 ** attempt)
                instance.__init__(instance.global_setting, socket_timeout=INIT_SOCKET_TIMEOUT * attempt)
                err = e
            except (AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"数据库连接失败:{str(e)}")
                print(f"{2 ** attempt}秒后自动尝试重连，剩余{ATTEMPTS - attempt}次")
                time.sleep(2 ** attempt)
                err = e
            except OperationFailure as e:
                if "Authentication failed" in str(e):
                    raise OperationFailure("数据库认证失败，请检查用户名及密码！")
                else:
                    raise
        raise ConnectionFailure(f"{str(err)}\n数据库连接不可用，请检查网络、IP地址或端口正确性！")
    return wrapper


# %%引擎类
class MongoManager:
    """
    传入mongoDB设置，获取连接实例。
    参数字典格式：
    host: str IP地址
    port: int 端口号
    username: str 用户名
    password: str 密码
    authSource: str 验证数据库，默认admin

    不传入参数则默认连接本地27017端口无验证的数据库
    """

    def __init__(self, settings=None, socket_timeout=INIT_SOCKET_TIMEOUT):
        self.socket_timeout = socket_timeout
        self.API: Union[MongoClient, None] = None  # PyMongo的MongoClient对象
        self.global_setting = settings or {}
        self._connect()

    def _ping(self):
        """检查连接情况及权限验证情况"""
        # 调用command查询服务器状态，防止服务器异常并未连接成功
        self.API.admin.command("ping")

    @auto_reconnect
    def _connect(self):
        """连接MongoDB数据库"""
        self.API = MongoClient(
            self.global_setting.get('host', "127.0.0.1"),
            self.global_setting.get('port', 27017),
            username=self.global_setting.get('username', None),
            password=self.global_setting.get('password', None),
            authSource=self.global_setting.get('authSource', 'admin'),
            connectTimeoutMS=5000, serverSelectionTimeoutMS=1000,
            compressors='zstd',
            replicaSet=self.global_setting.get('replicaSet'),
        )
        self._ping()

    @auto_reconnect
    def insert_one(self, db, collection, document):
        """向MongoDB中插入数据，document是具体一条数据"""
        if self._docs_is_empty(document):
            logger.warning("docs is empty, insert_one will return directly")
            return
        self.API[db][collection].insert_one(document)

    @auto_reconnect
    def insert_many(self, db, collection, documents):
        """向MongoDB中插入数据，documents是dataframe对象或可迭代mapping对象"""

        if self._docs_is_empty(documents):
            logger.warning("docs is empty, insert_many will return directly")
            return

        if isinstance(documents, pd.DataFrame):
            documents = documents.to_dict(orient='records')

        self.API[db][collection].insert_many(documents)

    @auto_reconnect
    def query(self, db, collection, flt=None, return_as_df=True, projection=None,
              sort_key=None, sort_direction=None, **kwargs):
        """
        mongo通用查询函数

        Parameters
        ----------
        db: str, 查询目标数据库
        collection: str, 查询目标集合
        flt: dict, 查询条件
        return_as_df: bool, 是否将返回值转为pd.DataFrame格式，或直接返回cursor
        projection: dict
            使用掩码的方式确定返回值中应包含的字段，
            如{"_id": 0, "remark": 0}, 则返回除_id和remark之外的字段
            注意: 提供的字典中，除_id字段外，其他字段只能是全0或全1，
            即要么选择一些字段，要么排除一些字段，掩码不可混用
            默认值: {"_id": 0}
        sort_key: str, list(tuple)
            如需排序，则提供一个排序字段，
            多字段排序可提供包含排序字段与排序方向数据对的列表，如：
            [("date", -1), ("time", 1)]
        sort_direction: 排序方向，默认正向(ASCENDING or 1)。如已提供排序字典，则可以忽略。
        可选的其他参数：
        limit: int, 限制返回值最大记录条数
        skip: int, 跳过返回值的前面若干条记录
        其他pymonogo官方文档提供的参数，按关键字方式传参。

        Returns
        -------
        默认返回pd.DataFrame格式的数据，或者直接返回cursor
        """

        if sort_key and isinstance(sort_key, str):
            sort_direction = sort_direction or ASCENDING
            sort = [(sort_key, sort_direction)]
        else:
            sort = sort_key
        projection = projection or {"_id": 0}

        cursor = self.API[db][collection].find(filter=flt, projection=projection,
                                               sort=sort, **kwargs)
        if return_as_df:
            df = pd.DataFrame(tuple(cursor))
            return df
        else:
            return cursor

    @auto_reconnect
    def distinct(self, db, collection, field, flt=None):
        """
        获取field字段在flt条件下去重后的列表

        Parameters
        ----------
        db: str, 数据库名
        collection: str, 集合名
        flt: dict, 筛选条件
        field: str, 字段名

        Returns
        -------
        List
        """
        return self.API[db][collection].distinct(field, flt)

    @auto_reconnect
    def replace_one(self, db, collection, flt, replacement, upsert=True):
        """
        替换一条数据库中的记录，如果匹配得到多条，只会替换第一条
        Parameters
        ----------
        db: str, 数据库名
        collection: str, 集合名
        flt: dict, 查找被替换文档的筛选条件
        replacement: dict, 用于替换的内容
        upsert: 未找到匹配项则插入，默认True

        """
        if self._docs_is_empty(replacement):
            logger.warning("docs is empty, replace_one will return directly")
            return
        self.API[db][collection].replace_one(flt, replacement, upsert=upsert)

    @auto_reconnect
    def replace_all(self, db, collection, flt, replacement, backup=False):
        """
        **谨慎使用**：务必检查好flt条件

        **删除**根据flt筛选出的所有记录，然后批量插入replacement内容
        注意跟原生replace的区别，原生replace不会变更_id值
        删除再插入会变更_id值

        Parameters
        ----------
        db: str, 数据库名
        collection: 集合名
        flt: 筛选条件
        replacement: 用于替换的记录
        backup: 删除的数据是否进行备份，默认不备份（慢）

        Returns
        -------

        """
        self.delete(db, collection, flt, backup=backup)
        self.insert_many(db, collection, replacement)

    def _backup(self, db, collection, flt):
        """
        对将要删除的数据进行备份，以当时时间进行命名
        参数与delete方法相同
        """
        col_backup = f"FROM_{db}_{collection}_{datetime.today().strftime('%y%m%d_%H%M%S_')}{datetime.now().microsecond}"
        documents = self.query(db, collection, flt, projection={})
        if documents.empty:
            return
        self.insert_many(DB_BACKUP, col_backup, documents=documents)

    def _clean_backup(self, delta: timedelta = None):
        """
        清理备份的集合，默认清理60天前的
        """
        delta = delta or timedelta(60)
        collections = self.list_collection_names(DB_BACKUP)
        for col in collections:
            if datetime.strptime(col.split("_")[-3], "%y%m%d") + delta < datetime.today():
                logger.warning(f"2s后删除备份集合{col}，有误请及时Ctrl+C取消")
                time.sleep(2)
                self.drop_collection(DB_BACKUP, col)
                logger.info(f"集合{col}已成功删除\n")

    @staticmethod
    def _docs_is_empty(documents):
        """空文档的插入会报错，进行一下检测"""
        if isinstance(documents, pd.DataFrame):
            return documents.empty
        return not documents

    @auto_reconnect
    def delete(self, db, collection, flt, backup=False):
        """从数据库中删除数据，flt是过滤条件

        Parameters
        ----------
        backup: 删除的数据是否进行备份，默认不备份（慢）
        """
        if backup:
            self._backup(db, collection, flt)
        self.API[db][collection].delete_many(filter=flt)

    def drop_collection(self, db, collection):
        """从数据库中删除集合"""
        if self.API:
            db = self.API[db]
            db[collection].drop()

    def remove_collections(self, db, to_delete, reverse=False):
        """
        db:mongo对象，数据库对象
        collection_names:mongo对象，所有集合
        reverse:True的时候删除不在给定的集合名的集合，False的时候删除给定的集合名
        """
        collections = self.list_collection_names(db)
        for collection in list(collections):
            if reverse and collection not in to_delete:
                logger.warning(f'2s后删除集合：{collection}，后悔请按Ctrl+C')
                time.sleep(2)
                self.drop_collection(db, collection)
            elif not reverse and collection in to_delete:
                logger.warning(f'2s后删除集合：{collection}，后悔请按Ctrl+C')
                time.sleep(2)
                self.drop_collection(db, collection)

    def list_collection_names(self, db):
        """从MongoDB中读取所有集合名称"""
        return self.API[db].list_collection_names()

    def close(self):
        if self.API:
            self.API.close()
            self.API = None
    
    # utility
    def get_last_value_of(self, db, collection, fields, flt, direction=-1):
        """
        获取数据库中某一个字段，在某条件下，排序后的第一个值，默认降序（-1）
        
        Parameters
        ----------
        db: 数据库
        collection: 数据表
        fields: 字段
        flt: 查询条件
        direction: 顺序，1升序，-1降序

        Returns
        -------
        返回一个值，空则返回None
        """

        cur = self.API[db][collection].find(filter=flt, sort=[(fields, direction)],
                                            projection={fields: 1}, limit=1)
        res_list = list(cur)

        if res_list:
            return res_list[0].get(fields)
        else:
            return None

    # -----------------------------------------legacy
    def loadHistoryData(self, dbName, symbol, start="20000101", end="",
                        fields=['datetime', 'lastPrice'], pdformat=True, batch_size=1000, upper=True):

        """
        加载历史区间数据
        db:str,数据库名称
        symbol:str,合约名称，程序会自动处理成大写，数据都是大写，入参无严格大小写区分
        start：str，开始时间，格式为%Y%m%d或%Y%m%d %H:%M:%S
        end：str，结束时间，格式为%Y%m%d或%Y%m%d %H:%M:%S
        fields:list,需要获取的字段
        pdformat:bool,是否需要转换成dataFrame对象，否的话为数据表指针对象dbCursor
        batch_size:int,获取行大小，默认为1000，如果为-1则获取所有
        """
        if not end:
            end = ''
        if not start:
            start = "20000101"

        if upper:
            collection = self.API[dbName][symbol.upper()]
        else:
            collection = self.API[dbName][symbol]

        # 数据查询条件
        dayFmt = '%Y%m%d'
        minFmt = '%Y%m%d %H:%M:%S'
        dataStartDate = datetime.strptime(start, dayFmt) if len(start) == 8 else datetime.strptime(start, minFmt)
        dataEndDate = datetime.strptime(end, dayFmt) if len(end) == 8 else datetime.strptime(end, minFmt) if len(
            end) > 0 else None
        flt = {'datetime': {'$gte': dataStartDate}} if not dataEndDate else {
            'datetime': {'$gte': dataStartDate, '$lte': dataEndDate}}

        # 执行查询
        if batch_size > 0:
            dbCursor = collection.find(flt).batch_size(batch_size)
        else:
            dbCursor = collection.find(flt)

        if not pdformat:
            return dbCursor

        if 'datetime' not in fields:
            fields.insert(0, 'datetime')

        allDatas = [data for data in dbCursor]

        if len(fields) == 1:
            fields = set([])
            for d in allDatas:
                fields = fields | set(d.keys())
            fields.remove('_id')
            fields = [str(f) for f in fields]
        datas = pd.DataFrame(allDatas, columns=fields, index=range(0, dbCursor.count()))
        datas = datas.set_index('datetime')
        return datas

    # -----------------------------------------加载最新数据
    def loadLastData(self, dbName, symbol, limit=1):
        collection = self.API[dbName][symbol.upper()]
        cursor = collection.find({}, sort=[("datetime", DESCENDING)]).limit(limit)
        data = [data for data in cursor]
        return data
