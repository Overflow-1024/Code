import datetime as dt


class BaseModel:

    def __init__(self, date, db_manager):
        self.date = date
        self.datetime = dt.datetime.strptime(str(date), '%Y-%m-%d')
        self.db_manager = db_manager

        self.data = None
        self._load_data()

    def _load_data(self):
        raise NotImplementedError

    def get_methods(self):
        def filt_fun(item):
            if (item.startswith('txt_') or item.startswith('tbl_') or item.startswith('fig_')) and callable(getattr(self, item)):
                return True
            else:
                return False
        return list(filter(filt_fun, dir(self)))
