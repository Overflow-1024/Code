import numpy as np
import pandas as pd
import openpyxl.cell._writer
from datetime import datetime

import os
import warnings

warnings.filterwarnings('ignore')

config = {
    'input': {
        'voucher_path': "./input/凭证.xlsx",
        'sequence_book_path': "./blank/序时簿-空.xlsx",

        'detail_template_path': "./blank/明细账-空.xlsx",
        'detail_account_path': "./input/明细账",

        'summary_template_path': "./blank/科目汇总-空.xlsx",
        'subject_summary_path': "./input/科目汇总.xlsx",

        'trial_balance_path': "./blank/科目余额及试算平衡-空.xlsx",
        'profit_statement_path': "./blank/利润表-空.xlsx",
        'balance_sheet_path': "./blank/资产负债表-空.xlsx",

        'last_year_path': "./input/上年结转.xlsx"

    },
    'output': {
        'output_path': "./output",
        'sequence_book_path': "./output/序时簿.xlsx",

        'detail_account_path': "./output/明细账",

        'subject_summary_path': "./output/科目汇总.xlsx",

        'trial_balance_path': "./output/科目余额及试算平衡.xlsx",
        'profit_statement_path': "./output/利润表.xlsx",
        'balance_sheet_path': "./output/资产负债表.xlsx",
    }
}


def parse_date(date_str):
    time = datetime.strptime(date_str.replace(' ', ''), "%Y年%m月%d日")

    return time.date()


def encode(text):
    if text == '借':
        return 1
    else:
        return -1


def decode(value):
    if value == 1:
        return '借'
    else:
        return '贷'


class AccountProcessor:

    def __init__(self):

        self.voucher = None
        self.voucher_inter = None
        self.sequence_book = None

        self.detail_template = None
        self.detail_account = None

        self.summary_template = None
        self.subject_summary = None

        self.trial_balance = None
        self.profit_statement = None
        self.balance_sheet = None

        self.year_begin = True
        self.year = -1
        self.month = -1
        self.error_list = {
            '总账科目': set(),
            '明细科目': set()
        }
        # self.sub_property = {
        #     '银行存款': 1,
        #     '应收账款': 1,
        #     '预付账款': 1,
        #     '其他应收款': 1,
        #     '应付账款': -1,
        #     '其他应付款': -1,
        #     '本年利润': -1,
        #     '利润分配': -1,
        #     '销售收入': -1,
        #     '销售成本': 1,
        #     '销售费用': 1,
        #     '管理费用': 1,
        #     '财务费用': 1,
        #     '营业外收入': -1,
        #     '营业外支出': 1,
        #     '短期借款': -1,
        # }

        # 明细账 列坐标
        self.borrow_col = 5
        self.loan_col = 6
        self.prop_col = 7
        self.balance_col = 8

    def print_errmsg(self, scope):

        if len(self.error_list['总账科目']) > 0:
            print(scope + " 找不到总账科目： " + '，'.join(list(self.error_list['总账科目'])))

        if len(self.error_list['明细科目']) > 0:
            print(scope + " 找不到明细科目： " + '，'.join(list(self.error_list['明细科目'])))

        self.error_list['总账科目'].clear()
        self.error_list['明细科目'].clear()

    # 读取数据
    def read_data(self):

        self.voucher = pd.read_excel(io=config['input']['voucher_path'], header=None)
        self.voucher_inter = pd.DataFrame(columns=['日期', '凭证号', '摘要', '总账科目', '明细科目', '对方科目', '借方金额', '贷方金额', '借贷标记'])

        self.sequence_book = pd.read_excel(io=config['input']['sequence_book_path'], header=None)

        self.detail_template = pd.read_excel(io=config['input']['detail_template_path'], header=None)

        self.summary_template = pd.read_excel(io=config['input']['summary_template_path'], header=None)

        self.profit_statement = pd.read_excel(io=config['input']['profit_statement_path'], header=None)
        self.trial_balance = pd.read_excel(io=config['input']['trial_balance_path'], header=None)
        self.balance_sheet = pd.read_excel(io=config['input']['balance_sheet_path'], header=None)

        # 年初（1月份）运行读取上年结转数据
        if self.year_begin:
            last_year_data = pd.read_excel(io=config['input']['last_year_path'])

            last_year_data.fillna(method='ffill', inplace=True)
            # 数据清洗 总账科目/明细科目去空格
            last_year_data['总账科目'] = last_year_data['总账科目'].str.strip()
            last_year_data['明细科目'] = last_year_data['明细科目'].str.strip()

            # 处理明细账
            self.detail_account = dict()
            for index, data in last_year_data.iterrows():

                table_name = data['总账科目'] + '（总）'  # 总表名字

                if data['总账科目'] not in self.detail_account.keys():
                    self.detail_account[data['总账科目']] = dict()

                general_book = self.detail_account[data['总账科目']]
                general_book[data['明细科目']] = self.detail_template.copy(deep=True)

                detail_book = general_book[data['明细科目']]
                detail_book.iloc[1, 0] = '账户名称：' + data['明细科目']
                detail_book.iloc[-1, self.balance_col] = data['上年结转']
                if data['明细科目'] == table_name:
                    detail_book.iloc[-1, self.prop_col] = data['借或贷']
                else:
                    detail_book.iloc[-1, self.prop_col] = general_book[table_name].iloc[4, self.prop_col]

            # 处理科目汇总
            self.subject_summary = dict()
            for index, data in last_year_data.iterrows():

                table_name = data['总账科目'] + '（总）'  # 总表名字

                if data['总账科目'] not in self.subject_summary.keys():
                    self.subject_summary[data['总账科目']] = self.summary_template.copy(deep=True)

                general_sheet = self.subject_summary[data['总账科目']]

                general_sheet.iloc[0, 0] = data['总账科目']

                if data['明细科目'] == table_name:
                    general_sheet.iloc[-1, 1] = table_name
                    general_sheet.iloc[-1, 2] = data['上年结转']
                    general_sheet.iloc[-1, -1] = 0
                else:
                    general_sheet.loc[len(general_sheet)] = general_sheet.iloc[-1]
                    new_row = [len(general_sheet) - 3, data['明细科目'], data['上年结转'],
                               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                               0]
                    general_sheet.iloc[-2] = new_row

        # 不是年初直接读取本年数据
        else:
            # 明细账
            self.detail_account = dict()
            for root, dirs, files in os.walk(config['input']['detail_account_path']):
                for file in files:
                    subject_name = file.split('.')[0]
                    path = os.path.join(root, file)
                    self.detail_account[subject_name] = pd.read_excel(io=path, sheet_name=None, header=None)
            # 科目汇总
            self.subject_summary = pd.read_excel(io=config['input']['subject_summary_path'], sheet_name=None, header=None)

    # 处理一张凭证的内容
    def process_one_voucher(self, r):

        group = pd.DataFrame(columns=['日期', '凭证号', '摘要', '总账科目', '明细科目', '对方科目', '借方金额', '贷方金额', '借贷标记'])
        borrow_subject = set()  # 借方科目
        loan_subject = set()  # 贷方科目

        borrow_sum = 0
        loan_sum = 0

        i = r
        # 第0行: 记账凭证

        i = i + 1
        # 第1行: 单位名称  时间  凭证号
        row = self.voucher.iloc[i, :]
        date = parse_date(row[2])  # 时间 第3列
        number = row[5].replace(' ', '')  # 凭证号 第6列
        abstract = None

        # 第2行: 摘要  会计科目  人民币
        # 第3行: 总账科目  明细科目  币种  借方金额  贷方金额

        i = i + 3
        # 第4行 开始内容
        row = self.voucher.iloc[i, :]
        while not pd.isna(row[1]) and not row[1] == '合计':  # 总账科目不为空且值不是 合计 ，则认定为有内容

            # 借贷金额相等，说明一个group结束了，处理上一个group
            if not group.empty and borrow_sum == loan_sum:
                # 处理 对方科目
                # group.loc[group['借贷标记'] == '借', '对方科目'] = ','.join(list(loan_subject))
                # group.loc[group['借贷标记'] == '贷', '对方科目'] = ','.join(list(borrow_subject))
                group.loc[(group['借贷标记'] == '借') & (group['借方金额'] >= 0), '对方科目'] = ','.join(list(loan_subject))
                group.loc[(group['借贷标记'] == '借') & (group['借方金额'] < 0), '对方科目'] = ','.join(list(borrow_subject))
                group.loc[(group['借贷标记'] == '贷') & (group['贷方金额'] >= 0), '对方科目'] = ','.join(list(borrow_subject))
                group.loc[(group['借贷标记'] == '贷') & (group['贷方金额'] < 0), '对方科目'] = ','.join(list(loan_subject))

                # 搞完把当前group的内容放到voucher_inter，然后清空相关变量
                self.voucher_inter = self.voucher_inter.append(group, ignore_index=True)
                # self.voucher_inter = pd.concat([self.voucher_inter, group], ignore_index=True)

                group.drop(group.index, inplace=True)
                borrow_subject.clear()
                loan_subject.clear()
                borrow_sum = 0
                loan_sum = 0

            # 摘要
            if not pd.isna(row[0]):
                abstract = row[0]

            general_subject = row[1]  # 总账科目
            detail_subject = row[2]  # 明细科目
            borrow_amount = row[4]  # 借方金额
            loan_amount = row[5]  # 贷方金额
            flag = '借' if pd.isna(loan_amount) else '贷'  # 借贷标记

            if flag == '借':
                borrow_sum += borrow_amount
                if borrow_amount < 0:
                    loan_subject.add(general_subject)
                else:
                    borrow_subject.add(general_subject)
            else:
                loan_sum += loan_amount
                if loan_amount < 0:
                    borrow_subject.add(general_subject)
                else:
                    loan_subject.add(general_subject)

            # 一条记录
            record = {
                '日期': date,
                '凭证号': number,
                '摘要': abstract,
                '总账科目': general_subject,
                '明细科目': detail_subject,
                '借方金额': borrow_amount,
                '贷方金额': loan_amount,
                '借贷标记': flag,
                '对方科目': np.nan
            }
            group.loc[len(group)] = record

            i = i + 1
            row = self.voucher.iloc[i, :]

        # 处理 对方科目
        # group.loc[group['借贷标记'] == '借', '对方科目'] = ','.join(list(loan_subject))
        # group.loc[group['借贷标记'] == '贷', '对方科目'] = ','.join(list(borrow_subject))
        group.loc[(group['借贷标记'] == '借') & (group['借方金额'] >= 0), '对方科目'] = ','.join(list(loan_subject))
        group.loc[(group['借贷标记'] == '借') & (group['借方金额'] < 0), '对方科目'] = ','.join(list(borrow_subject))
        group.loc[(group['借贷标记'] == '贷') & (group['贷方金额'] >= 0), '对方科目'] = ','.join(list(borrow_subject))
        group.loc[(group['借贷标记'] == '贷') & (group['贷方金额'] < 0), '对方科目'] = ','.join(list(loan_subject))

        # 搞完把当前group的内容放到voucher_inter，然后清空相关变量
        self.voucher_inter = self.voucher_inter.append(group, ignore_index=True)
        # self.voucher_inter = pd.concat([self.voucher_inter, group], ignore_index=True)

        group.drop(group.index, inplace=True)
        borrow_subject.clear()
        loan_subject.clear()
        borrow_sum = 0
        loan_sum = 0

        while not self.voucher.iloc[i, 0] == '会计主管':  # 第一格是 会计主管 则认为该凭证结束
            i = i + 1

        return i

    # 从凭证表格提取内容到voucher_inter
    def extract_info(self):
        # 数据清洗 统一空值
        self.voucher.replace(' ', np.nan, inplace=True)

        row_ptr = 0

        while row_ptr < self.voucher.shape[0]:

            if self.voucher.iloc[row_ptr, 2] == '记账凭证':
                row_ptr = self.process_one_voucher(row_ptr)

            row_ptr = row_ptr + 1

        # 数据清洗 总账科目/明细科目去空格
        self.voucher_inter['总账科目'] = self.voucher_inter['总账科目'].str.strip()
        self.voucher_inter['明细科目'] = self.voucher_inter['明细科目'].str.strip()

        # 数据清洗 明细科目缺失值
        self.voucher_inter.loc[pd.isna(self.voucher_inter['明细科目']), '明细科目'] = self.voucher_inter.loc[
            pd.isna(self.voucher_inter['明细科目']), '总账科目']

        # 检查，保证'借贷标记'一列的值正常
        if len(self.voucher_inter['借贷标记'].unique()) != 2:
            print("借贷标记一列有异常值")

        # 获取年份月份，取首行和尾行数据
        if self.voucher_inter.iloc[0, 0].year == self.voucher_inter.iloc[-1, 0].year:
            self.year = self.voucher_inter.iloc[-1, 0].year
        else:
            print("无法确定当前年份")
        if self.voucher_inter.iloc[0, 0].month == self.voucher_inter.iloc[-1, 0].month:
            self.month = self.voucher_inter.iloc[-1, 0].month
        else:
            print("无法确定当前月份")

    # 填写序时簿
    def fill_sequence_book(self):

        sequence_book_content = self.voucher_inter.drop(columns=['对方科目', '借贷标记'])
        sequence_book_content['记账'] = np.nan

        # 合计行
        sum_row = {
            '日期': np.nan,
            '凭证号': np.nan,
            '摘要': '合计',
            '总账科目': np.nan,
            '明细科目': np.nan,
            '借方金额': round(sequence_book_content['借方金额'].sum(), 2),
            '贷方金额': round(sequence_book_content['贷方金额'].sum(), 2),
            '记账': np.nan
        }

        if sum_row['借方金额'] != sum_row['贷方金额']:
            print('序时簿借贷金额总数不相等')

        sequence_book_content.loc[len(sequence_book_content)] = sum_row

        # 获取dataframe的列名
        self.sequence_book.columns = self.sequence_book.iloc[3, :]
        self.sequence_book = self.sequence_book.append(sequence_book_content, ignore_index=True)
        # self.sequence_book = pd.concat([self.sequence_book, sequence_book_content], ignore_index=True)

    # 按照科目生成所需表格
    def update_subject_sheet(self):

        for index, data in self.voucher_inter.iterrows():

            table_name = data['总账科目'] + '（总）'   # 总表名字

            # 修改 明细账 和 科目汇总 两个地方
            if data['总账科目'] not in self.detail_account.keys():
                # 明细账
                self.detail_account[data['总账科目']] = dict()

                general_book = self.detail_account[data['总账科目']]
                general_book[table_name] = self.detail_template.copy(deep=True)
                general_book[table_name].iloc[1, 0] = '账户名称：' + table_name
                general_book[table_name].iloc[-1, self.prop_col] = data['借贷标记']

                # 科目汇总
                self.subject_summary[data['总账科目']] = self.summary_template.copy(deep=True)
                general_sheet = self.subject_summary[data['总账科目']]

                general_sheet.iloc[0, 0] = data['总账科目']
                general_sheet.iloc[-1, 1] = table_name
                general_sheet.iloc[-1, -1] = 0

            else:
                general_book = self.detail_account[data['总账科目']]
                general_sheet = self.subject_summary[data['总账科目']]

            if data['明细科目'] not in general_book.keys():
                # 明细账
                general_book[data['明细科目']] = self.detail_template.copy(deep=True)
                general_book[data['明细科目']].iloc[1, 0] = '账户名称：' + data['明细科目']
                general_book[data['明细科目']].iloc[4, self.prop_col] = general_book[table_name].iloc[4, self.prop_col]

                # 科目汇总
                general_sheet.loc[len(general_sheet)] = general_sheet.iloc[-1]
                new_row = [len(general_sheet) - 3, data['明细科目'], np.nan,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           0]
                general_sheet.iloc[-2] = new_row

    # 填写明细账
    def fill_detail_account(self):

        data_row = -3
        month_row = -2
        year_row = -1

        # 给每个表格先添加 本月合计 本年累计 2行
        for general_subject, excel_file in self.detail_account.items():
            for detail_subject, sheet in excel_file.items():
                # 本月合计行
                sheet.loc[len(sheet)] = [self.month, np.nan, np.nan, '本月合计', np.nan,
                                         0, 0, sheet.iloc[4, self.prop_col],
                                         sheet.iloc[-1, self.balance_col], np.nan]
                # 本年累计行
                if sheet.iloc[-2, 3] == '本年累计':
                    sheet.loc[len(sheet)] = [self.month, np.nan, np.nan, '本年累计', np.nan,
                                             sheet.iloc[-2, self.borrow_col], sheet.iloc[-2, self.loan_col], sheet.iloc[4, self.prop_col],
                                             sheet.iloc[-2, self.balance_col], np.nan]
                elif sheet.iloc[-2, 3] == '上年结转':
                    sheet.loc[len(sheet)] = [self.month, np.nan, np.nan, '本年累计', np.nan,
                                             0, 0, sheet.iloc[4, self.prop_col],
                                             sheet.iloc[-2, self.balance_col], np.nan]
                else:
                    print("明细账表格最后一行数据有误")

        # 录入数据
        for index, data in self.voucher_inter.iterrows():

            detail_book = self.detail_account[data['总账科目']][data['明细科目']]

            # 计算余额
            balance = detail_book.iloc[-3, self.balance_col]
            if data['借贷标记'] == '借':
                balance = round(balance + encode(detail_book.iloc[4, self.prop_col]) * data['借方金额'], 2)
            else:
                balance = round(balance - encode(detail_book.iloc[4, self.prop_col]) * data['贷方金额'], 2)

            value = [
                data['日期'].month, data['日期'].day, data['凭证号'], data['摘要'], data['对方科目'],
                data['借方金额'], data['贷方金额'], detail_book.iloc[4, self.prop_col],
                balance, np.nan]

            # 插入新数据行
            detail_book.loc[len(detail_book)] = detail_book.iloc[-1]
            detail_book.iloc[-2] = detail_book.iloc[-3]
            detail_book.iloc[-3] = value

            # 更新 本月合计 本年累计 2行
            if data['借贷标记'] == '借':
                detail_book.iloc[month_row, self.borrow_col] += round(detail_book.iloc[data_row, self.borrow_col], 2)
                detail_book.iloc[year_row, self.borrow_col] += round(detail_book.iloc[data_row, self.borrow_col], 2)
            else:
                detail_book.iloc[month_row, self.loan_col] += round(detail_book.iloc[data_row, self.loan_col], 2)
                detail_book.iloc[year_row, self.loan_col] += round(detail_book.iloc[data_row, self.loan_col], 2)

            detail_book.iloc[month_row, self.balance_col] = detail_book.iloc[data_row, self.balance_col]
            detail_book.iloc[year_row, self.balance_col] = detail_book.iloc[data_row, self.balance_col]

        # 计算各总账科目汇总数据
        for general_subject, excel_file in self.detail_account.items():
            sum_sheet = excel_file[general_subject + '（总）']

            for detail_subject, sheet in excel_file.items():

                if detail_subject != general_subject + '（总）':
                    sum_sheet.iloc[month_row, self.borrow_col] += sheet.iloc[month_row, self.borrow_col]
                    sum_sheet.iloc[month_row, self.loan_col] += sheet.iloc[month_row, self.loan_col]
                    sum_sheet.iloc[month_row, self.balance_col] += sheet.iloc[month_row, self.balance_col]

            last = sum_sheet.iloc[-3, self.balance_col]
            borrow = sum_sheet.iloc[month_row, self.borrow_col]
            loan = sum_sheet.iloc[month_row, self.loan_col]
            sign = encode(sum_sheet.iloc[4, self.prop_col])

            sum_sheet.iloc[month_row, self.balance_col] = last + sign * (borrow - loan)
            if sum_sheet.iloc[-3, 3] == '本年累计':
                sum_sheet.iloc[year_row, self.borrow_col] = sum_sheet.iloc[-3, self.borrow_col] + borrow
                sum_sheet.iloc[year_row, self.loan_col] = sum_sheet.iloc[-3, self.loan_col] + loan
            elif sum_sheet.iloc[-3, 3] == '上年结转':
                sum_sheet.iloc[year_row, self.borrow_col] = borrow
                sum_sheet.iloc[year_row, self.loan_col] = loan
            else:
                print("明细账表格上一期最后一行数据有误")

            sum_sheet.iloc[year_row, self.balance_col] = sum_sheet.iloc[month_row, self.balance_col]

    # 从明细账汇总表获取利润表需要的数据
    def get_data_from_detail(self, mode, row, col, general_subject, detail_subject=None):

        """
        mode:  0:取汇总表数据，此时detail_subject参数没用
               1:取明细表数据

        row:  -1:本年累计
              -2:本月合计
              -3:上月的本年累计/上年结转

        col:  5:借方金额
              6:贷方金额
              8:余额

        """

        if general_subject in self.detail_account.keys():
            if mode == 0:
                return self.detail_account[general_subject][general_subject + '（总）'].iloc[row, col]
            else:
                if detail_subject in self.detail_account[general_subject].keys():
                    return self.detail_account[general_subject][detail_subject].iloc[row, col]
                else:
                    self.error_list['明细科目'].add(detail_subject)
                    return np.nan
        else:
            self.error_list['总账科目'].add(general_subject)
            return np.nan

    # 填写科目汇总表
    def fill_account_summary(self):

        name_col = 1
        start_col = 2

        for general_subject, sheet in self.subject_summary.items():

            start_row = 2
            end_row = len(sheet) - 1

            for i in range(start_row, end_row):
                detail_subject = sheet.iloc[i, name_col]
                sheet.iloc[i, start_col + self.month] = self.get_data_from_detail(1, -2, self.balance_col, general_subject, detail_subject)

                # 右 合计列
                sheet.iloc[i, -1] += sheet.iloc[i, start_col + self.month]

            # 下 合计行
            sheet.iloc[-1, start_col + self.month] = sheet.iloc[start_row: end_row, start_col + self.month].sum(axis=0)
            sheet.iloc[-1, -1] = sheet.iloc[start_row: end_row, -1].sum(axis=0)

        self.print_errmsg("填写科目汇总表时")

    # 填写科目余额及试算平衡表
    def fill_trial_balance(self):

        # 生成时间
        self.trial_balance.iloc[2, 1] = str(self.year) + "年" + str(self.month) + "月"

        start_row = 4
        end_row = 19
        sum_row = -1

        subject_col = 1
        initial_borrow = 2
        initial_loan = 3
        current_borrow = 4
        current_loan = 5
        year_borrow = 6
        year_loan = 7
        ending_borrow = 8
        ending_loan = 9

        for r in range(start_row, end_row + 1):
            sub = self.trial_balance.iloc[r, subject_col]

            self.trial_balance.iloc[r, current_borrow] = self.get_data_from_detail(0, -2, self.borrow_col, sub)
            self.trial_balance.iloc[r, current_loan] = self.get_data_from_detail(0, -2, self.loan_col, sub)
            self.trial_balance.iloc[r, year_borrow] = self.get_data_from_detail(0, -1, self.borrow_col, sub)
            self.trial_balance.iloc[r, year_loan] = self.get_data_from_detail(0, -1, self.loan_col, sub)

            if sub in self.detail_account.keys():
                if self.detail_account[sub][sub + '（总）'].iloc[4, self.prop_col] == '借':
                    if self.year_begin:
                        self.trial_balance.iloc[r, initial_borrow] = self.get_data_from_detail(0, -3, self.balance_col, sub)
                    else:
                        self.trial_balance.iloc[r, initial_borrow] = self.get_data_from_detail(0, -4, self.balance_col, sub)
                    self.trial_balance.iloc[r, ending_borrow] = self.get_data_from_detail(0, -2, self.balance_col, sub)

                else:
                    if self.year_begin:
                        self.trial_balance.iloc[r, initial_loan] = self.get_data_from_detail(0, -3, self.balance_col, sub)
                    else:
                        self.trial_balance.iloc[r, initial_loan] = self.get_data_from_detail(0, -4, self.balance_col, sub)
                    self.trial_balance.iloc[r, ending_loan] = self.get_data_from_detail(0, -2, self.balance_col, sub)
            else:
                print("找不到总账科目： " + sub)

        # 合计
        self.trial_balance.iloc[sum_row, current_borrow] = self.trial_balance.iloc[start_row: sum_row, current_borrow].sum(axis=0)
        self.trial_balance.iloc[sum_row, current_loan] = self.trial_balance.iloc[start_row: sum_row, current_loan].sum(axis=0)
        self.trial_balance.iloc[sum_row, year_borrow] = self.trial_balance.iloc[start_row: sum_row, year_borrow].sum(axis=0)
        self.trial_balance.iloc[sum_row, year_loan] = self.trial_balance.iloc[start_row: sum_row, year_loan].sum(axis=0)
        self.trial_balance.iloc[sum_row, initial_borrow] = self.trial_balance.iloc[start_row: sum_row, initial_borrow].sum(axis=0)
        self.trial_balance.iloc[sum_row, initial_loan] = self.trial_balance.iloc[start_row: sum_row, initial_loan].sum(axis=0)
        self.trial_balance.iloc[sum_row, ending_borrow] = self.trial_balance.iloc[start_row: sum_row, ending_borrow].sum(axis=0)
        self.trial_balance.iloc[sum_row, ending_loan] = self.trial_balance.iloc[start_row: sum_row, ending_loan].sum(axis=0)

        # 取2位小数
        for col in range(initial_borrow, ending_loan + 1):
            self.trial_balance.iloc[sum_row, col] = round(self.trial_balance.iloc[sum_row, col], 2)

        # 检验合计数
        check = [
            (self.trial_balance.iloc[sum_row, current_borrow], self.trial_balance.iloc[sum_row, current_loan]),
            (self.trial_balance.iloc[sum_row, year_borrow], self.trial_balance.iloc[sum_row, year_loan]),
            (self.trial_balance.iloc[sum_row, initial_borrow], self.trial_balance.iloc[sum_row, initial_loan]),
            (self.trial_balance.iloc[sum_row, ending_borrow], self.trial_balance.iloc[sum_row, ending_loan])
        ]
        if not (check[0][0] == check[0][1] and check[1][0] == check[1][1] and check[2][0] == check[2][1] and check[3][0] == check[3][1]):
            print("试算平衡合计数 借方不等于贷方")

        self.print_errmsg("填写科目余额及试算平衡表时")

    # 填写利润表
    def fill_profit_statement(self):

        # 生成时间
        self.profit_statement.iloc[2, 0] = "所属期：" + str(self.year) + "年" + str(self.month) + "月"

        start_row = 7

        # 总账科目  行坐标
        location = [
            ('销售收入', 0),
            ('销售成本', 1),
            ('销售费用', 2),
            # 销售利润   3
            ('管理费用', 4),
            ('财务费用', 5),
            # 营业利润   6
            ('营业外收入', 7),
            ('营业外支出', 8),
            # 利润总额   9
            # 净利润    10
        ]

        for item in location:
            self.profit_statement.iloc[start_row + item[1], 2] = self.get_data_from_detail(0, -2, self.balance_col, item[0])
            self.profit_statement.iloc[start_row + item[1], 3] = self.get_data_from_detail(0, -1, self.balance_col, item[0])

        # 销售利润
        self.profit_statement.iloc[start_row + 3, 2] = self.profit_statement.iloc[start_row, 2] \
                                                       - self.profit_statement.iloc[start_row + 1, 2] \
                                                       - self.profit_statement.iloc[start_row + 2, 2]
        self.profit_statement.iloc[start_row + 3, 3] = self.profit_statement.iloc[start_row, 3] \
                                                       - self.profit_statement.iloc[start_row + 1, 3] \
                                                       - self.profit_statement.iloc[start_row + 2, 3]

        # 营业利润
        self.profit_statement.iloc[start_row + 6, 2] = self.profit_statement.iloc[start_row + 3, 2] \
                                                       - self.profit_statement.iloc[start_row + 4, 2] \
                                                       - self.profit_statement.iloc[start_row + 5, 2]
        self.profit_statement.iloc[start_row + 6, 3] = self.profit_statement.iloc[start_row + 3, 3] \
                                                       - self.profit_statement.iloc[start_row + 4, 3] \
                                                       - self.profit_statement.iloc[start_row + 5, 3]

        # 利润总额
        self.profit_statement.iloc[start_row + 9, 2] = self.profit_statement.iloc[start_row + 6, 2] \
                                                       + self.profit_statement.iloc[start_row + 7, 2] \
                                                       - self.profit_statement.iloc[start_row + 8, 2]
        self.profit_statement.iloc[start_row + 9, 3] = self.profit_statement.iloc[start_row + 6, 3] \
                                                       + self.profit_statement.iloc[start_row + 7, 3] \
                                                       - self.profit_statement.iloc[start_row + 8, 3]

        # 净利润
        self.profit_statement.iloc[start_row + 10, 2] = self.profit_statement.iloc[start_row + 9, 2]
        self.profit_statement.iloc[start_row + 10, 3] = self.profit_statement.iloc[start_row + 9, 3]

        self.print_errmsg("填写利润表时")

    # 填写资产负债表
    def fill_balance_sheet(self):

        # 生成时间
        self.balance_sheet.iloc[2, 7] = "所属期：" + str(self.year) + "年" + str(self.month) + "月"

        start_row = 5
        asset_end_row = 8
        debt_end_row = 9
        asset_col = 0
        debt_col = 4

        for r in range(start_row, asset_end_row + 1):
            sub = self.balance_sheet.iloc[r, asset_col]

            # 取 上年结转
            self.balance_sheet.iloc[r, asset_col + 2] = self.get_data_from_detail(0, 4, self.balance_col, sub)
            # 取 本月合计
            self.balance_sheet.iloc[r, asset_col + 3] = self.get_data_from_detail(0, -2, self.balance_col, sub)

        for r in range(start_row, debt_end_row + 1):
            sub = self.balance_sheet.iloc[r, debt_col]

            # 取 上年结转
            self.balance_sheet.iloc[r, debt_col + 2] = self.get_data_from_detail(0, 4, self.balance_col, sub)
            # 取 本月合计
            self.balance_sheet.iloc[r, debt_col + 3] = self.get_data_from_detail(0, -2, self.balance_col, sub)

        # 计算 总计数
        sum_row = 11

        self.balance_sheet.iloc[sum_row, asset_col + 2] = self.balance_sheet.iloc[start_row:start_row + 4, asset_col + 2].sum(axis=0)
        self.balance_sheet.iloc[sum_row, asset_col + 3] = self.balance_sheet.iloc[start_row:start_row + 4, asset_col + 3].sum(axis=0)
        self.balance_sheet.iloc[sum_row, debt_col + 2] = self.balance_sheet.iloc[start_row:start_row + 5, asset_col + 2].sum(axis=0)
        self.balance_sheet.iloc[sum_row, debt_col + 3] = self.balance_sheet.iloc[start_row:start_row + 5, asset_col + 3].sum(axis=0)

        # 取两位小数
        self.balance_sheet.iloc[sum_row, asset_col + 2] = round(self.balance_sheet.iloc[sum_row, asset_col + 2], 2)
        self.balance_sheet.iloc[sum_row, asset_col + 3] = round(self.balance_sheet.iloc[sum_row, asset_col + 3], 2)
        self.balance_sheet.iloc[sum_row, debt_col + 2] = round(self.balance_sheet.iloc[sum_row, debt_col + 2], 2)
        self.balance_sheet.iloc[sum_row, debt_col + 3] = round(self.balance_sheet.iloc[sum_row, debt_col + 3], 2)

        # 检查 资产=负债
        check = [
            (self.balance_sheet.iloc[sum_row, asset_col + 2], self.balance_sheet.iloc[sum_row, debt_col + 2]),
            (self.balance_sheet.iloc[sum_row, asset_col + 3], self.balance_sheet.iloc[sum_row, debt_col + 3])
        ]
        if not (check[0][0] == check[0][1] and check[1][0] == check[1][1]):
            print("资产负债表 资产总计不等于负债总计")

        self.print_errmsg("填写资产负债表")

    # 将内容填写到需要的表格
    def fill_info(self):

        # 填写序时簿
        self.fill_sequence_book()

        # 按照科目生成所需表格
        self.update_subject_sheet()

        # 填写明细账
        self.fill_detail_account()

        # 填写科目汇总表
        self.fill_account_summary()

        # 填写科目余额及试算平衡表
        self.fill_trial_balance()

        # 填写利润表
        self.fill_profit_statement()

        # 填写资产负债表
        self.fill_balance_sheet()

    # 输出excel文件
    def output(self):

        headline = {
            'align': 'center',  # 水平位置设置：居中
            'valign': 'vcenter',  # 垂直位置设置，居中
            'font_size': 14,
            'bold': True
        }
        text = {
            'align': 'center',  # 水平位置设置：居中
            'valign': 'vcenter',  # 垂直位置设置，居中
        }
        border = {'border': 1}

        """
        voucher_inter

        """
        # path = "./output/voucher_inter.xlsx"
        # self.voucher_inter.to_excel(path, index=False)

        """
        序时簿

        """
        path = config['output']['sequence_book_path']
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        workbook = writer.book

        my_format = workbook.add_format(text)
        head_format = workbook.add_format(headline)
        border_format = workbook.add_format(border)

        self.sequence_book.to_excel(writer, index=False, header=False)
        worksheet = writer.sheets['Sheet1']

        worksheet.set_default_row(20)

        worksheet.set_column(0, 7, 16, my_format)

        worksheet.set_row(0, 24, head_format)
        worksheet.merge_range('A1:H1', self.sequence_book.iloc[0, 0], head_format)
        worksheet.conditional_format(3, 0, self.sequence_book.shape[0] - 1, self.sequence_book.shape[1] - 1,
                                     {'type': 'no_errors', 'format': border_format})

        writer.save()

        '''
        明细账

        '''
        os.mkdir(config['output']['detail_account_path'])

        for general_subject, excel_file in self.detail_account.items():
            file_name = general_subject + ".xlsx"
            path = os.path.join(config['output']['detail_account_path'], file_name)
            writer = pd.ExcelWriter(path, engine='xlsxwriter')
            workbook = writer.book

            my_format = workbook.add_format(text)
            head_format = workbook.add_format(headline)
            border_format = workbook.add_format(border)

            for detail_subject, sheet in excel_file.items():
                sheet.to_excel(writer, sheet_name=detail_subject, index=False, header=False)
                worksheet = writer.sheets[detail_subject]

                worksheet.set_default_row(20)

                worksheet.set_column(0, 1, 4, my_format)
                worksheet.set_column(2, 2, 16, my_format)
                worksheet.set_column(3, 3, 20, my_format)
                worksheet.set_column(4, 6, 16, my_format)
                worksheet.set_column(7, 7, 8, my_format)
                worksheet.set_column(8, 9, 16, my_format)

                worksheet.set_row(0, 24, head_format)
                worksheet.merge_range('A1:L1', sheet.iloc[0, 0], head_format)
                worksheet.merge_range('A2:D2', sheet.iloc[1, 0], my_format)
                worksheet.merge_range('A3:B3', sheet.iloc[2, 0], my_format)

                worksheet.conditional_format(2, 0, sheet.shape[0] - 1, sheet.shape[1] - 1,
                                             {'type': 'no_errors', 'format': border_format})
            writer.save()

        '''
        科目汇总

        '''
        path = config['output']['subject_summary_path']
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        workbook = writer.book

        my_format = workbook.add_format(text)
        head_format = workbook.add_format({
            'align': 'center',  # 水平位置设置：居中
            'valign': 'vcenter',  # 垂直位置设置，居中
            'bold': True
        })
        border_format = workbook.add_format(border)

        for general_subject, sheet in self.subject_summary.items():
            sheet.to_excel(writer, sheet_name=general_subject, index=False, header=False)
            worksheet = writer.sheets[general_subject]

            worksheet.set_default_row(20)

            worksheet.set_column(0, 0, 4, my_format)
            worksheet.set_column(1, 2, 20, my_format)
            worksheet.set_column(3, 15, 12, my_format)

            worksheet.merge_range('A1:C1', sheet.iloc[0, 0], head_format)
            worksheet.conditional_format(1, 0, sheet.shape[0] - 1, sheet.shape[1] - 1,
                                         {'type': 'no_errors', 'format': border_format})

        writer.save()

        """
        科目余额及试算平衡

        """
        path = config['output']['trial_balance_path']
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        workbook = writer.book

        my_format = workbook.add_format(text)
        head_format = workbook.add_format(headline)
        border_format = workbook.add_format(border)

        self.trial_balance.to_excel(writer, index=False, header=False)
        worksheet = writer.sheets['Sheet1']

        worksheet.set_default_row(20)
        worksheet.set_column(0, 9, 16, my_format)

        worksheet.set_row(0, 24, head_format)
        worksheet.merge_range('A1:J1', self.trial_balance.iloc[0, 0], head_format)
        worksheet.merge_range('C3:D3', self.trial_balance.iloc[2, 2], my_format)
        worksheet.merge_range('E3:F3', self.trial_balance.iloc[2, 4], my_format)
        worksheet.merge_range('G3:H3', self.trial_balance.iloc[2, 6], my_format)
        worksheet.merge_range('I3:J3', self.trial_balance.iloc[2, 8], my_format)

        worksheet.conditional_format('B3:J22', {'type': 'no_errors', 'format': border_format})
        writer.save()

        '''
        利润表

        '''
        path = config['output']['profit_statement_path']
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        workbook = writer.book

        my_format = workbook.add_format(text)
        first_col_format = workbook.add_format({
            'valign': 'vcenter',  # 垂直位置设置，居中
        })
        head_format = workbook.add_format(headline)
        border_format = workbook.add_format(border)

        self.profit_statement.to_excel(writer, index=False, header=False)
        worksheet = writer.sheets['Sheet1']

        worksheet.set_default_row(20)
        worksheet.set_column(0, 0, 32, first_col_format)
        worksheet.set_column(1, 4, 16, my_format)

        worksheet.set_row(4, 20, my_format)
        worksheet.set_row(6, 20, my_format)

        worksheet.set_row(0, 24, head_format)
        worksheet.merge_range('A1:D1', self.profit_statement.iloc[0, 0], head_format)
        worksheet.merge_range('A3:D3', self.profit_statement.iloc[2, 0], my_format)

        worksheet.conditional_format('A7:D18', {'type': 'no_errors', 'format': border_format})
        writer.save()

        '''
        资产负债表

        '''
        path = config['output']['balance_sheet_path']
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        workbook = writer.book

        my_format = workbook.add_format(text)
        head_format = workbook.add_format(headline)
        border_format = workbook.add_format(border)

        self.balance_sheet.to_excel(writer, index=False, header=False)
        worksheet = writer.sheets['Sheet1']

        worksheet.set_default_row(20)
        worksheet.set_column(0, 0, 16, my_format)
        worksheet.set_column(1, 1, 6, my_format)
        worksheet.set_column(2, 4, 16, my_format)
        worksheet.set_column(5, 5, 6, my_format)
        worksheet.set_column(6, 7, 16, my_format)

        worksheet.set_row(0, 24, head_format)
        worksheet.merge_range('A1:H1', self.balance_sheet.iloc[0, 0], head_format)
        worksheet.merge_range('G3:H3', self.balance_sheet.iloc[2, 7], my_format)

        worksheet.conditional_format('A5:H12', {'type': 'no_errors', 'format': border_format})
        writer.save()

    # 主函数
    def run(self):

        while True:
            print("请输入本次是否年初（1月份）运行程序： 是[Y]  否[N]")
            a = input()

            if a == 'Y' or a == 'y':
                self.year_begin = True
                break
            elif a == 'N' or a == 'n':
                self.year_begin = False
                break
            else:
                print("输入不合法")

        print("读取文件...")
        self.read_data()

        print("解析凭证...")
        self.extract_info()

        print("填写表格...")
        self.fill_info()

        print("输出文件...")
        self.output()

        print("程序运行完成，请按回车键退出...")
        a = input()


AP = AccountProcessor()
AP.run()
