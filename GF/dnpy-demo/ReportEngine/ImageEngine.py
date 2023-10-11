import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


class ImageEngine:

    def __init__(self, path, rc_config, config):
        self.fig_path = path
        for param in rc_config:
            plt.rcParams[param] = rc_config[param]
        plt.rc('axes', axisbelow=True)
        self.FIG_SIZE = config['FIG_SIZE']
        self.COLOR_LIST = config['COLOR_LIST']

    # 取消边框
    def delete_frame(self, ax):
        # ---2.取消边框
        for key, spine in ax.spines.items():
            if key == 'right' or key == 'left' or key == 'top':
                spine.set_visible(False)
            else:
                spine.set_color('dimgray')
                spine.set_linewidth(1)
        return

    # 绘制一条曲线
    def draw_curve(self, ax, data, config, lable_fmt='%s'):
        """
        函数功能：在图上绘制一条曲线（一组数据）
        参数：
        ax：画图的轴
        curve：dict，存要画图的数据以及画图参数。
            字段：
            data：数据
            kinds：图的类型。如折线图，条形图，散点图等
            color：曲线颜色
            width：柱状图的柱体宽度
        """

        # ---1.清洗数据
        if config.get('keep_0', False):
            x = data.dropna()
        else:
            x = data.replace({0: None}).dropna()

        # ---2.获取配置
        kinds = config.get('kinds', 'plot')
        zorder = config.get('zorder', None)
        linestyle = config.get('linestyle', '-')
        lw = config.get('lw', 3.5)
        al = config.get('alpha', 1)

        # ---3.绘制曲线
        if kinds == 'bar':  # 柱状图
            w = config.get('width', 2)
            line = ax.bar(x.index, x.values, facecolor=config['color'],
                          label=lable_fmt % x.name, linewidth=0, width=w, alpha=al, zorder=zorder)
        elif kinds == 'thinbar':  # 条形图
            w = config.get('width', 0.6)
            line = ax.bar(np.arange(len(x)), x.values, facecolor=config['color'], align='center',
                          label=x.name, linewidth=0, width=w, alpha=al, zorder=zorder)
        elif kinds == 'stackedbar':  # 堆叠柱状图
            w = config.get('width', 2)
            line = ax.bar(x.index, x.values, facecolor=config['color'], bottom=config['bottom'].values,
                          label=lable_fmt % x.name, linewidth=0, width=w, alpha=al, zorder=zorder)
        elif kinds == 'groupedbar':  # 双柱状图
            w = config.get('width', 0.3)
            if config['side'] == 'l':
                line = ax.bar(np.arange(len(x)) - w / 2, x.values, facecolor=config['color'],
                              label=lable_fmt % x.name, linewidth=0, width=w, alpha=al, zorder=zorder)
            else:
                line = ax.bar(np.arange(len(x)) + w / 2, x.values, facecolor=config['color'],
                              label=lable_fmt % x.name, linewidth=0, width=w, alpha=al, zorder=zorder)

        elif kinds == 'scatter':  # 散点图
            line = ax.scatter(x.index, x.values, c=config['color'],
                              label=lable_fmt % x.name, marker=config['marker'], s=90, alpha=al, linewidths=0,
                              zorder=zorder)

        elif kinds == 'fill':  # 填充横轴的曲线
            line = ax.fill_between(x.index, x.values, facecolor=config['color'], linewidth=0, alpha=al,
                                   label=lable_fmt % x.name,
                                   zorder=zorder)

        elif kinds == 'fillbetween':  # 填充两条曲线
            x2 = config['data_2'].replace({0: None}).dropna()
            line = ax.fill_between(x.index, x.values, x2.values, facecolor=config['color'], linewidth=0, alpha=al,
                                   label=lable_fmt % x.name, zorder=zorder)
        else:  # 普通折线图
            line, = ax.plot(x, color=config['color'], label=lable_fmt % x.name, linestyle=linestyle, linewidth=lw,
                            zorder=zorder,
                            alpha=al)

        return line

    # 绘制一张图片
    def draw_picture(self, data, config):
        """
        函数功能：绘制一张图片
        参数：
        ax：画图的轴。可选x轴和y轴。
        fs：画布尺寸
        **config: dict。存画图的参数列表（可选）
            字段：
            lines_left：list，存左轴曲线
            lines_right：list，存右轴曲线
            lines_config: list，存曲线画图参数
            fname：图片保存时的文件名
            zorder：右轴曲线的图层。左轴曲线图层设置为2，因此，zorder>2时右轴曲线在上，zorder<2时右轴曲线在下
            n_col：图例列数
            month_gap：横坐标（时间）的月份间隔
            axis_label：横坐标标签，文字标签
            monthly：boolean，控制横坐标标签是否为月份
            range_yax1：左轴纵坐标显示范围
            range_yax2：右轴纵坐标显示范围
        """
        lines = []
        labels = []

        ax = config.get('axis', 'y')
        fs = config.get('figure_size', 'small')

        # ---1.设置画布尺寸
        fig = plt.figure(figsize=self.FIG_SIZE[fs])
        ax1 = fig.add_subplot(111)
        ax2 = None

        # ---2.绘制主图（左侧Y轴）
        if config.get('lines_left', []):

            # -----2.1取消边框
            self.delete_frame(ax1)

            # -----2.2画线
            for index in config.get('lines_left', []):
                line_data = data[index]
                line_config = config.get('lines_config', [])[index]
                ln = self.draw_curve(ax1, line_data, line_config)
                lines.append(ln)
                labels.append(ln.get_label())

            # -----2.3纵坐标轴字体大小
            plt.setp(ax1.get_yticklabels(), fontsize='xx-large')

        # ---3.绘制主图（右侧Y轴）
        if config.get('lines_right', []):  # 双Y轴
            ax2 = ax1.twinx()  # this is the important function

            # -----3.0 设置图层，默认左轴曲线在上，右轴曲线在下
            ax1.set_zorder(2)
            ax2.set_zorder(config.get('zorder', 1))
            ax1.patch.set_visible(False)

            # -----3.1取消边框
            self.delete_frame(ax2)

            # -----3.2画线
            for index in config.get('lines_right', []):
                line_data = data[index]
                line_config = config.get('lines_config', [])[index]
                ln = self.draw_curve(ax2, line_data, line_config, '%s(右轴)')
                lines.append(ln)
                labels.append(ln.get_label())

            # -----3.3纵坐标轴字体大小
            plt.setp(ax2.get_yticklabels(), fontsize='xx-large')

            # -----3.4不要刻度线（即，凸起部分）
            ax2.yaxis.set_ticks_position('none')

        # ---4.设置图例
        if config.get('axis_label', False) or config.get('monthly', False) or config.get('title', False):
            ax1.set_xlabel('xlable', labelpad=30, color='white')
        else:
            v_pos = -0.5 if len(lines) <= config.get('n_col', 2) else -0.55
            plt.title(u'legend', x=0, y=v_pos, color='white')

        # h_pos = 0.47 if fs in ['small', 's'] else 0.44
        # fig.legend(lines, labels, loc='center', bbox_to_anchor=(h_pos, 0.02),
        fig.legend(lines, labels, loc='lower center', borderpad=0, borderaxespad=0.05,
                   ncol=config.get('n_col', 2), fontsize='xx-large', frameon=False, scatterpoints=1)

        # ---5.设置标题
        if config.get('title', False):
            plt.title(config['title'], fontsize='xx-large', color='black', y=1.02)

        # ---6.设置横坐标标签
        # format xaxis with 4 month intervals
        if config.get('axis_label', False):  # 文字标签
            ax1.set_xticks(np.arange(len(config['axis_label'])))
            ax1.set_xticklabels(config['axis_label'])
            plt.setp(ax1.get_xticklabels(), fontsize='xx-large')
            ax1.set_xlim(-0.5, len(config['axis_label']) - 0.5)

        elif config.get('monthly', False):  # '1-12月'标签
            ax1.set_xlim(ax1.get_xlim()[0] - 1, )
            ax1.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
            ax1.set_xticklabels(['%d月' % m for m in range(1, 13)])
            plt.setp(ax1.get_xticklabels(), rotation=0, ha="left", fontsize='xx-large')

        else:  # 时间序列标签
            # ax1.set_xlim(ax1.get_xlim()[0]-20,)
            month_seq = data[config.get('lines_left', [])[-1]].index
            by_month = month_seq.max().month % config.get('month_gap', 3)
            by_month = config.get('month_gap', 3) if by_month == 0 else by_month
            by_month = range(by_month, 13, config.get('month_gap', 3))
            ax1.get_xaxis().set_major_locator(mdates.MonthLocator(bymonth=by_month))
            ax1.get_xaxis().set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", fontsize='xx-large')

        # ---7.纵坐标轴显示范围
        if config.get('range_yax1', False):
            ax1.set_ylim(config['range_yax1'][0], config['range_yax1'][1])
        if config.get('range_yax2', False):
            ax2.set_ylim(config['range_yax2'][0], config['range_yax2'][1])

        # ---8.纵坐标轴显示自定义规则
        if config.get('Fmter_yax1', False):
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: config['Fmter_yax1'] % (x)))
        if config.get('Fmter_yax2', False):
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: config['Fmter_yax2'] % (x)))

        # ---9.横坐标轴两侧留白
        if config.get('margins_ax1', False):
            ax1.margins(x=config['margins_ax1'])
        if config.get('margins_ax2', False):
            ax2.margins(x=config['margins_ax2'])

        # ---9.不要刻度线（即，凸起部分）
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')

        # ---10.设置网格线
        if config.get('lines', []):
            ax2.grid(False)
        ax1.grid(axis='x', linestyle='solid', c='dimgray', linewidth=0 if ax == 'y' else 0.75)
        ax1.grid(axis='y', linestyle='solid', c='dimgray', zorder=0)

        # ---11.保存图片
        figure_name = config['fname'] + '.png'
        save_path = os.path.join(self.fig_path, figure_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()