import datetime as dt

global_config = {
    'template_path': './Projects/AmReport/Template.docx',
    'output_path': './Projects/AmReport/output',
    'fig_path': './Projects/AmReport/output/Figure',
    'output_filename': '大类资产表现周度跟踪报告%s.docx',
    'date': dt.date(2022, 2, 15),
}

db_config = {
    'db_host': '10.51.135.17',
    'db_port': 20017
}

plt_rc_config = {
    'font.sans-serif': 'SimHei',   # 用来正常显示中文标签
    'font.size': 10,
    'savefig.dpi': 100,  # 设置图片像素
    'legend.edgecolor': 'black',
    'legend.handletextpad': 0.1,
    'axes.unicode_minus': False,  # 显示负号
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'axes.xmargin': 0.01,
    "legend.columnspacing": 1,
    "grid.alpha": 0.6,
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,
    'ytick.minor.pad': 8
}

plt_config = {
    'FIG_SIZE': {'l': (13, 4.4), 's': (8.6, 4), 'large': (13, 4.4), 'small': (8.6, 4)},
    'COLOR_LIST': ['blue', 'brown', 'purple', 'orange', 'gray', 'gold', 'green', 'skyblue', 'red']
}

