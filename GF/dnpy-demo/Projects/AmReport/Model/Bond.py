from Projects.AmReport.Model.BaseModel import BaseModel
import Projects.AmReport.util as util


class Bond(BaseModel):

    def __init__(self, date, db_manager):
        super().__init__(date, db_manager)

    def _load_data(self):
        self.data = util.query_table(db_connect=self.db_manager, cltName='bonds', sort_k='date', datetime=self.datetime)
