import pandas as pd
import os
import sys
import csv


# 列分割
def columns_split(file_name):
    df = pd.read_csv(file_name, engine='python')

    # 读出列名并进行分割
    col_name = df.columns.values.tolist()
    col_name = col_name[0].split(';')

    # 列分割
    df = df.iloc[:, 0].str.split(';', expand=True)
    # 列重命名
    df.columns = col_name
    df.to_csv(file_name, encoding='gb2312', index=False)


# 添加标签列
def add_label(file_name, label):

    df = pd.read_csv(file_name, engine='python', encoding='gb2312')
    df.insert(loc=1, column='pod', value=label)
    df.to_csv(file_name, encoding='gb2312', index=False)


# 遍历文件夹
def traverse_folder(root_path, fun):
    files = os.listdir(root_path)  # 得到文件夹下的所有文件名称

    for file in files:  # 遍历文件夹
        child_path = os.path.join(root_path, file)
        print(child_path)

        if os.path.isdir(child_path):  # 判断是否是文件夹，是文件夹则递归遍历
            traverse_folder(child_path)

        else:   # 不是文件夹

            if fun == 1:
                columns_split(child_path)
            elif fun == 2:
                label_value = root_path[-1]
                add_label(child_path, label_value)

    print(root_path + "  finish")


# 检查文件是否匹配
def check_file_match(path1, path2, path3, path4):

    MFCC12 = os.listdir(path1)  # 得到文件夹下的所有文件名称
    PLP = os.listdir(path2)
    prosodyShs = os.listdir(path3)
    smileF0 = os.listdir(path4)

    flag = 1
    if not(len(MFCC12) == len(PLP) and len(PLP) == len(prosodyShs) and len(prosodyShs) == len(smileF0)):
        flag = 0
        print("number is not match")
    else:
        for i in range(len(MFCC12)):
            if not (MFCC12[i] == PLP[i] and PLP[i] == prosodyShs[i] and prosodyShs[i] == smileF0[i]):
                flag = 0
                print("file_name is not match")
                break

    if flag == 1:
        print("match successfully")
    else:
        print("match failed")


# 修改smileF0列名以及前三行的frameTime
def modify_smileF0(root_path):
    files = os.listdir(root_path)  # 得到文件夹下的所有文件名称

    for file in files:  # 遍历文件夹
        print(file)
        child_path = os.path.join(root_path, file)
        df = pd.read_csv(child_path, engine='python', encoding='gb2312')
        df = df.rename(columns={'F0final_sma': 'F0final_sma_0'})

        # 删掉第二行，剩下两行framTime改为0.01 0.02
        df.drop(index=1, inplace=True)
        df.iloc[0, 2] = 0.01
        df.iloc[1, 2] = 0.02
        df.to_csv(child_path, encoding='gb2312', index=False)


# 改最后一行的frameTime
def modify_prosodyShs(root_path):
    files = os.listdir(root_path)  # 得到文件夹下的所有文件名称

    for file in files:  # 遍历文件夹
        print(file)
        child_path = os.path.join(root_path, file)
        df = pd.read_csv(child_path, engine='python', encoding='gb2312')

        # 把最后一行的framTime的值改为倒数第二行+0.01
        df.iloc[-1, 2] = df.iloc[-2, 2] + 0.01
        df.to_csv(child_path, encoding='gb2312', index=False)


# 拼接表格
def merge_data(path1, path2, path3, path4, path_merge):

    MFCC12 = os.listdir(path1)
    PLP = os.listdir(path2)
    prosodyShs = os.listdir(path3)
    smileF0 = os.listdir(path4)

    for i in range(len(MFCC12)):
        child_path1 = os.path.join(path1, MFCC12[i])
        child_path2 = os.path.join(path2, PLP[i])
        child_path3 = os.path.join(path3, prosodyShs[i])
        child_path4 = os.path.join(path4, smileF0[i])
        print(MFCC12[i])

        df1 = pd.read_csv(child_path1, engine='python', encoding='gb2312')
        df2 = pd.read_csv(child_path2, engine='python', encoding='gb2312')
        df3 = pd.read_csv(child_path3, engine='python', encoding='gb2312')
        df4 = pd.read_csv(child_path4, engine='python', encoding='gb2312')

        df = pd.merge(df1, df2, how='inner', on=None)
        df = pd.merge(df, df3, how='inner', on=None)
        df = pd.merge(df, df4, how='inner', on=None)

        child_path_new = os.path.join(path_merge, MFCC12[i])
        df.to_csv(child_path_new, encoding="gb2312", index=False)


# 修改标签列pod，数值改为0和1
def modify_IS_label(file_list):
    for file in file_list:
        print(file)
        df = pd.read_csv(file, engine='python', encoding='gb2312')
        df.loc[df['pod'].notnull(), 'pod'] = 1
        df.loc[df['pod'].isnull(), 'pod'] = 0
        df.to_csv(file, encoding="gb2312", index=False)


# 修改IS12/IS13的name（去掉单引号）
def modify_IS_name(file_list):

    for file in file_list:
        df = pd.read_csv(file, engine='python', encoding='gb2312')
        for i in range(df.shape[0]):
            str_name = df.ix[i, 'name']
            # df.ix[i, 'name'] = str_name[1: -1]
            df.ix[i, 'name'] = "'" + str_name + "'"
        df.to_csv(file, encoding='gb2312', index=False)


# 比较dataframe的两列数据，返回相似度
def get_similarity(list1, list2):
    similarity = 0
    if len(list1) != len(list2):
        print("Error! length doesn't match")
    else:
        same = 0
        different = 0
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                same = same + 1
            else:
                different = different + 1
        similarity = same / (same + different)

    return similarity


# 检查不同数据文件的特征是否有重复部分，有则删除
def check_attr_intersection(file_list):

    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            file1 = file_list[i]
            file2 = file_list[j]
            df1 = pd.read_csv(file1, engine='python', encoding='gb2312')
            df2 = pd.read_csv(file2, engine='python', encoding='gb2312')

            attr_list1 = df1.columns.values.tolist()
            attr_list1.remove("name")
            attr_list1.remove("pod")
            attr_list2 = df2.columns.values.tolist()
            attr_list2.remove("name")
            attr_list2.remove("pod")

            set1 = set(attr_list1)
            set2 = set(attr_list2)
            inter_attr = set1.intersection(set2)
            print(file_list[i]+"    "+file_list[j])
            print(inter_attr)
            same_attr = []
            for attr in inter_attr:
                col1 = df1[attr].tolist()
                col2 = df2[attr].tolist()
                if get_similarity(col1, col2) > 0.95:
                    same_attr.append(attr)

            print(same_attr)

            print(file1[-8:] + " attr quantity:  " + str(len(attr_list1)))
            print(file2[-8:] + " attr quantity:  " + str(len(attr_list2)))
            print("same attr quantity:  " + str(len(same_attr)))

            if same_attr:
                for attr in same_attr:
                    df2.drop([attr], axis=1, inplace=True)
                df2.to_csv(file2, encoding='gb2312', index=False)
                print("new " + file2[-8:] + " attr quantity:  " + str(df2.shape[1] - 2))


# 对IS文件中同名列重命名
def modify_IS_colname(file_list):

    for file in file_list:
        print(file)
        df = pd.read_csv(file, engine='python', encoding='gb2312')
        attr_list = df.columns.values.tolist()
        attr_list.remove("name")
        attr_list.remove("pod")

        for attr in attr_list:
            df.rename(columns={attr: attr + '_' + file[-6:-4]}, inplace=True)
        df.to_csv(file, encoding='gb2312', index=False)


# 拼接IS数据表格
def merge_ISdata(file_list, path_merge):
    df = pd.read_csv(file_list[0], engine='python', encoding='gb2312')
    df = df.loc[:, ['name', 'pod']]
    print("init df attr quantity:  " + str(df.shape[1] - 2))

    for file in file_list:
        df1 = pd.read_csv(file, engine='python', encoding='gb2312')
        df = pd.merge(df, df1, how='inner', on=None)
        print(file[-8:] + " attr quantity:  " + str(df1.shape[1] - 2))
        print("new df attr quantity:  " + str(df.shape[1] - 2))

    df.to_csv(path_merge, encoding="gb2312", index=False)


maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


path = "D:/毕设数据/第一层"

traverse_folder(path, 1)
traverse_folder(path, 2)

path_MFCC12 = "D:/毕设数据/第一层/MFCC12"
path_PLP = "D:/毕设数据/第一层/PLP"
path_prosodyShs = "D:/毕设数据/第一层/prosodyShs"
path_smileF0 = "D:/毕设数据/第一层/smileF0"
path_new = "D:/毕设数据/第一层/MFCC12_PLP_prosodyShs_smileF0"

check_file_match(path_MFCC12, path_PLP, path_prosodyShs, path_smileF0)
modify_smileF0(path_smileF0)
modify_prosodyShs(path_prosodyShs)
merge_data(path_MFCC12, path_PLP, path_prosodyShs, path_smileF0, path_new)

path_IS09 = "D:/毕设数据/第一层/IS09.csv"
path_IS10 = "D:/毕设数据/第一层/IS10.csv"
path_IS11 = "D:/毕设数据/第一层/IS11.csv"
path_IS12 = "D:/毕设数据/第一层/IS12.csv"
path_IS13 = "D:/毕设数据/第一层/IS13.csv"
path_list = [path_IS09, path_IS10, path_IS11, path_IS12, path_IS13]
path_list0 = [path_IS12, path_IS13]
path_all = "D:/毕设数据/第一层/ISall.csv"

modify_IS_label(path_list)
modify_IS_name(path_list0)
check_attr_intersection(path_list)
modify_IS_colname(path_list)
merge_ISdata(path_list, path_all)



