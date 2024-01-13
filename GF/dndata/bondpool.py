from dndata.database import mongo


# 根据给定条件获取债券代码
def get_bond_list(year, bond_type, period):
    """
    根据
        年份：券码前两位
        类型：券码中间两位，00 国债，02 国开债，04 农发债
        期限：BOND_TERM字段指示，如"5年"，与券码后 2 位的期数无关
        券码位数：6 位整
    并
        剔除含权债,即TERMTOMATURITYSTRING字段中存在+号的，如"2.83Y+2Y"

    获取券码列表
    """

    code_pattern = r'^({})({})[0-9][0-9]$'.format(year, bond_type)

    projection = {'_id': 0, 'SECURITYID': 1, 'TERMTOMATURITYSTRING': 1}
    res = mongo.query(
        db='RDI', collection='RDI_CFETS_BOND',
        flt={'SECURITYID': {'$regex': code_pattern}, 'BOND_TERM': {'$regex': period}},
        projection=projection, return_as_df=False,
    )

    bond_list = []
    for bond in res:
        if '+' not in bond['TERMTOMATURITYSTRING']:
            bond_list.append(f"IB{bond['SECURITYID']}")
    bond_list.sort()

    return bond_list


# 测试用
if __name__ == "__main__":

    years = ['19', '20', '21', '22']
    periods = ['5年', '10年']

    for year in ['18', '19', '20', '21', '22']:
        for period in periods:
            print(f'--------{year}年-{period}---------')
            print(f'{year}年国债{period}期券列表：')
            print(get_bond_list(year, '00', period))

            print(f'{year}年国开债{period}期券列表：')
            print(get_bond_list(year, '02', period))

            print(f'{year}年农发债{period}期券列表：')
            print(get_bond_list(year, '04', period))
