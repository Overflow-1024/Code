import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

from Projects.AmReport.Model.BaseModel import BaseModel
import Projects.AmReport.util as util


class MonetaryPolicy(BaseModel):

    def __init__(self, date, db_manager):
        super().__init__(date, db_manager)
        self.df = self.data.copy().sort_index(ascending=False).replace({0: None})

        # 推导逆回购操作列表
        self.r_repo = pd.DataFrame()
        for day in ['7', '14', '28', '63']:
            a_r_repo = self.df[[u'逆回购%sD数量' % day, u'逆回购%sD利率' % day]].copy()
            a_r_repo.columns = ['qty', 'rate']
            a_r_repo.dropna(inplace=True)
            a_r_repo.loc[:, 'expiration'] = [x + dt.timedelta(int(day)) for x in a_r_repo.index]
            a_r_repo.loc[:, 'amt'] = a_r_repo['qty'] * a_r_repo['rate']
            self.r_repo = pd.concat([self.r_repo, a_r_repo])

        # 推导12个月MLF操作列表
        self.r_MLF = pd.DataFrame()
        for mon in ['6', '12']:
            a_MLF = self.df[[u'%s个月MLF数量' % mon, u'%s个月MLF利率' % mon]].copy()
            a_MLF.columns = ['qty', 'rate']
            a_MLF.dropna(inplace=True)
            a_MLF.loc[:, 'expiration'] = [x + relativedelta(months=int(mon)) for x in a_MLF.index]
            a_MLF.loc[:, 'amt'] = a_MLF['qty'] * a_MLF['rate']
            self.r_MLF = pd.concat([self.r_MLF, a_MLF])


    def _load_data(self):
        self.data = util.query_table(db_connect=self.db_manager, cltName='centerbankOP', sort_k='date', datetime=self.datetime)

    def _process(self, df):

        df = df.resample('w-sun', closed='right', label='right').sum()

        df['rate'] = df.apply(lambda x: x.amt / x.qty if x.qty != 0.0 else None, axis=1)
        df['chg'] = (df['rate'].diff()) * 100

        df['rate'] = df['rate'].apply(lambda x: '%.2f%%' % x if pd.notnull(x) else None)
        df['rate'] = df['rate'].fillna('/')
        df['chg'] = df['chg'].apply(lambda x: round(x) if pd.notnull(x) else None)
        df['chg'] = df['chg'].fillna('/').astype('str')

        df['qty'] = df['qty'].fillna(0).apply(lambda x: '%.d' % x)

        return df

    # ---准备表格数据
    def tbl_1(self):
        lw, tw, nw = pd.date_range(start=str(self.date), periods=3, freq='W-sun', closed='right').shift(-1)
        lw, tw, nw = str(lw.date()), str(tw.date()), str(nw.date())

        # -----表格1
        df_repo_issue = self._process(self.r_repo[['qty', 'amt']])
        df_repo_exp = self._process(self.r_repo[['qty', 'amt', 'expiration']].set_index('expiration'))
        df_mlf_issue = self._process(self.r_MLF[['qty', 'amt']])
        df_mlf_exp = self._process(self.r_MLF[['qty', 'amt', 'expiration']].set_index('expiration'))

        res = pd.DataFrame(columns=['qty', 'rate', 'chg'])
        res.loc['1', :] = df_repo_issue.loc[tw, :] if tw in df_repo_issue.index else ['0', '/', '/']
        res.loc['2', :] = df_repo_exp.loc[tw, :] if tw in df_repo_exp.index else ['0', '/', '/']
        res.loc['3', :] = df_mlf_issue.loc[tw, :] if tw in df_mlf_issue.index else ['0', '/', '/']
        res.loc['4', :] = df_mlf_exp.loc[tw, :] if tw in df_mlf_exp.index else ['0', '/', '/']
        res.loc['5', :] = df_repo_exp.loc[nw, :] if nw in df_repo_exp.index else ['0', '/', '/']
        res.loc['6', :] = df_mlf_exp.loc[nw, :] if nw in df_mlf_exp.index else ['0', '/', '/']
        res = res.replace({'': '0', '//': '/', -0.0: '0'})

        return res

    def fig_12(self):
        # ----图12
        data = pd.DataFrame()
        data['WE'] = pd.date_range(end=self.date, periods=122, freq='w-fri')[::-1]
        data['WS'] = data['WE'].apply(lambda x: x - dt.timedelta(6))

        for _, r in data.iterrows():
            data.loc[_, u'央行公开市场投放余量'] = self.r_repo[(r['WE'] < self.r_repo['expiration']) & (self.r_repo.index <= r['WE'])][
                'qty'].sum()
            data.loc[_, u'央行加权投放成本'] = self.r_repo[(r['WS'] <= self.r_repo.index) & (self.r_repo.index <= r['WE'])]['amt'].sum() / \
                                       self.r_repo[(r['WS'] <= self.r_repo.index) & (self.r_repo.index <= r['WE'])]['qty'].sum() if \
                self.r_repo[(r['WS'] <= self.r_repo.index) & (self.r_repo.index <= r['WE'])]['amt'].sum() != 0 else None
        data.set_index('WE', inplace=True)

        return [data[u'央行公开市场投放余量'], data[u'央行加权投放成本']]

    def fig_13(self):
        # ----图13
        data = pd.DataFrame()
        data['WE'] = pd.date_range(end=self.date, periods=122, freq='w-fri')[::-1]
        data['WS'] = data['WE'].apply(lambda x: x - dt.timedelta(6))

        for _, r in data.iterrows():
            data.loc[_, u'MLF余额'] = self.r_MLF[(r['WE'] < self.r_MLF['expiration']) & (self.r_MLF.index <= r['WE'])]['qty'].sum()
            data.loc[_, u'MLF投放成本'] = self.r_MLF[(r['WS'] <= self.r_MLF.index) & (self.r_MLF.index <= r['WE'])]['amt'].sum() / \
                                      self.r_MLF[(r['WS'] <= self.r_MLF.index) & (self.r_MLF.index <= r['WE'])]['qty'].sum() if \
                self.r_MLF[(r['WS'] <= self.r_MLF.index) & (self.r_MLF.index <= r['WE'])]['amt'].sum() != 0 else None
        data.set_index('WE', inplace=True)

        return [data[u'MLF余额'], data[u'MLF投放成本']]

    def fig_14(self):
        # ----图14
        data = pd.DataFrame()
        data['WE'] = pd.date_range(end=self.date, periods=121, freq='w-sun')[::-1]
        data['WS'] = data['WE'].apply(lambda x: x - dt.timedelta(6))

        for op in [u'逆回购7D', u'逆回购14D', u'逆回购28D', u'逆回购63D', u'国库定存3个月', u'6个月MLF', u'12个月MLF', u'TMLF365D']:
            amt = self.df['%s数量' % op] * self.df['%s利率' % op]
            qty = self.df['%s数量' % op]
            for _, r in data.iterrows():
                data.loc[_, u'%s(投放)' % op] = amt[(r['WS'] <= amt.index) & (amt.index <= r['WE'])].sum() / qty[
                    (r['WS'] <= qty.index) & (qty.index <= r['WE'])].sum() if amt[(r['WS'] <= amt.index) & (
                        amt.index <= r['WE'])].sum() != 0 else None
        data.set_index('WE', inplace=True)

        return [data[u'逆回购7D(投放)'], data[u'逆回购14D(投放)'], data[u'逆回购28D(投放)'], data[u'逆回购63D(投放)'],
                data[u'国库定存3个月(投放)'], data[u'6个月MLF(投放)'], data[u'12个月MLF(投放)'], data[u'TMLF365D(投放)']]

