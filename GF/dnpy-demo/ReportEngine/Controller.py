
import ReportEngine as Engine
import Projects.AmReport as Report


class Controller:

    def __init__(self):

        global_config = Report.config.global_config
        plt_rc_config = Report.config.plt_rc_config
        plt_config = Report.config.plt_config
        db_config = Report.config.db_config

        self.date = global_config['date']

        self.db_manager = Engine.MongoManager(settings={'host': db_config['db_host'], 'port': db_config['db_port']})
        self.drawer = Engine.ImageEngine(global_config['fig_path'], plt_rc_config, plt_config)
        self.renderer = Engine.DocxRenderer(global_config)  # 渲染器。这里可优化改成动态加载，现在写死了

        self.models = []    # 数据模型列表，每个元素一个model的实例
        self.fun_map = {}   # 函数名——实例  的映射

        self.init_context()

    # 初始化，动态加载project的模块
    def init_context(self):

        model_name = []
        for m in dir(Report.Model):
            if not m.startswith("_") and not m == 'BaseModel':
                model_name.append(m)

        for m in model_name:
            class_meta = getattr(getattr(Report.Model, m), m)
            obj = class_meta(self.date, self.db_manager)
            self.models.append(obj)

        for model in self.models:
            methods_list = model.get_methods()
            for func in methods_list:
                self.fun_map[func] = model

    # 解析文本，把文本中的占位符替换成数据
    def parse_text(self):
        # 文字段
        for key, value in Report.oriData.ori_text.items():
            records = self.fun_map[key].__getattribute__(key)()
            self.renderer.text_data_dict[key] = [value[0].format(*records[0]), value[1].format(*records[1])]
        # 图片/表格标题
        for key, value in Report.oriData.ori_title_text.items():
            records = self.fun_map[key].__getattribute__(key)()
            self.renderer.figure_title_dict[key] = [value.format(*records)]

    # 计算表格数据
    def pre_table(self):
        for key in Report.oriData.table_list:
            records = self.fun_map[key].__getattribute__(key)()
            self.renderer.table_data_dict[key] = records

    # 画图，并保存到指定路径
    def pre_figure(self):
        for key in Report.oriData.figure_list:
            self.drawer.draw_picture(
                data=self.fun_map[key].__getattribute__(key)(),
                config=Report.oriData.figure_config.get(key),
            )

    # 渲染，把数据填充到模板上
    def render(self):
        self.renderer.run()

    def generate_report(self):
        self.pre_figure()
        print('finish pre_figure')

        self.pre_table()
        print('finish pre_table')

        self.parse_text()
        print('finish parse_text')

        self.render()
        print('finish render')