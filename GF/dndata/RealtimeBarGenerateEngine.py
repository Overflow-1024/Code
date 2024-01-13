"""实时k线生成器"""
import copy
import datetime
import math
from collections import deque
from typing import Union
import logging

import numpy as np
import pandas as pd

from dndata.API.cons import VARIETY_MATCH_MARKET, GENERAL_TRADE_PERIOD
from dndata.API.datatype import TickData, BarData
from dndata.API.engine.subscribeEngine import encapsulate_irs_tick_callback, encapsulate_future_tick_callback, encapsulate_xbond_tick_callback
from dndata.API.utility.datatypeUtility import to_int, to_float
from dndata.API.utility.datetimeUtility import to_date

from dndata.Event import EventEngine, Event
from dndata.Event.EventEngine import TickEvent, TimerEvent, EVENT_TIMER, BarEvent

from dndata.IRS.subscribe import sub_irs_tick_clean
from dndata.future.query import get_future_main_contract
from dndata.future.subscribe import sub_future_tick
from dndata.bond.subscribe import sub_bond_tick_xbond

from dndata.future.cons import FUTURE_CODE_PATTERN
from dndata.bond.cons import BOND_CODE_PATTERN

from dndata.tools.datetimetools import TimeLocator
from dndata.configs import config

try:
    import ujson as json
except:
    import json

logger = logging.getLogger('dnScript.' + __name__)


class BarGenerator:
    """
    tick实时生成器
    """

    def __init__(
        self,
        code: str,
        cycle: str,
        use_timer: bool,
        from_tick: bool,
        callback: callable,
        open_offset: int = 0,
        close_offset: int = 0,
        auto_fill: str = 'last',
        add_amount: bool = False,
        holiday_mode: str = 'exchange',
    ):
        """
        实例化
        Parameters
        ----------
        code: str, 品种名，用于确定开/收盘时间
        cycle: str|int, K线周期，如: 5, 15, 30, 60, 'Day'
        use_timer: bool, 是否使用计时器辅助K线合成，不活跃合约才需要
        from_tick: bool, 是否使用tick直接合成K线，设为False则使用1min-K作为原料，提升性能
        callback: callable, K线推送的回调
        open_offset: int, 开盘时间偏移量，单位为分钟，负数向前偏移
        close_offset: int, 收盘时间偏移量，单位为分钟，正数向后偏移
        auto_fill: 对于不活跃合约，当前k线时间范围无tick时，如何填充当前K线
            'last': 使用上一个tick进行填充
            'nan' : 使用nan进行填充
            'ignore': 忽略当前k线，不进行推送
        add_amount: 是否在K线中增加[成交额]字段，=成交价*成交量
        holiday_mode: 假期模式，可选exchange或interbank
        """

        self.use_timer = use_timer  # 是否使用timer事件驱动
        self._use_timer = use_timer  # 储存用户输入，因k合成过程中可能改变self.use_timer
        self.from_tick = from_tick  # 使用tick直接合k线
        self.cycle = str(cycle)
        self.callback = callback
        self.open_offset = open_offset
        self.close_offset = close_offset

        assert auto_fill in {'last', 'nan', 'ignore'}
        self.auto_fill = auto_fill

        self.add_amount = add_amount

        # tick时间戳与系统时间的差值，用于时间同步
        self.time_delta = datetime.timedelta(0)

        # 延迟收盘相关
        self.delay = 10
        self.delay_cnt = 0

        # 实例化时间定位器
        if FUTURE_CODE_PATTERN.match(code):
            code = FUTURE_CODE_PATTERN.findall(code)[0][0]

        # 适配rq的时间切片方式，如果周期小于等于15，则使用有中途休息的时间参数
        rest_suffix = '_rest'
        # if self.cycle != 'Day' and self._cycle <= 15:
        #     rest_suffix = '_rest'

        # 所属日期定位器
        self.date_locator = TimeLocator(
            **GENERAL_TRADE_PERIOD,
            cycle='Day',
            inclusive='both',
            holiday_mode=holiday_mode,
        )

        if BOND_CODE_PATTERN.match(code):
            market = 'XBOND' + rest_suffix
        else:
            market = VARIETY_MATCH_MARKET[code] + rest_suffix

        # tick时间定位器
        self.locator = TimeLocator(
            **config.get_cons('MARKET_TRADE_TIMES')[market],  # periods and day_offsets
            cycle=self._cycle,
            inclusive='left',  # 左闭右开，配合后续1min Bar线合成多min Bar可实现合成后立即推出
            open_offset=open_offset,
            close_offset=close_offset,
            holiday_mode=holiday_mode,
        )

        # 用于辅助最后一根k的推送
        if self.cycle == 'Day':
            self.last_locator_index = 1
        else:
            self.last_locator_index = math.ceil(self.locator.locator_list[-1].end_index / self._cycle)

        self.bar: BarData = None
        self.last_bar_label = dict(
            date_belong=None, bar_index=-1, time_str='0:0'
        )  # 当前bar所属区间的标签，用于确定bar是否生成完毕

        # 记录已推送的label的str，避免同一根bar重复推送
        self.pushed_bar_labels = deque(maxlen=1000)  # 1000足够一天用

        self.last_high = np.nan  # 最新的最低价
        self.last_low = np.nan  # 最新的最高价
        self.last_tick: TickData = None  # 上一条收到的tick缓存

    def update(self, event: Event):
        """
        用于订阅tick/bar事件的回调函数，接受以下三种Event对象：
        TickEvent, BarEvent, TimerEvent

        Parameters
        ----------
        event: 事件对象
        """
        # 与tick同步时间
        if isinstance(event, TickEvent):
            self.time_delta = datetime.datetime.now() - event.dict_['data'].datetime

        # -----生成多分钟周期的bar
        # 时间事件
        if isinstance(event, TimerEvent):
            self._on_timer(event)
            return
        # tick或bar事件统一处理
        self._generate_bar(event.dict_['data'])

    def _generate_bar(self, item: Union[BarData, TickData]):
        """
        使用tick生成bar，或用1分钟bar生成多分钟bar
        """
        bar_index, date_belong, time_str = self.locator.locate(
            item.datetime, start_from=self.last_bar_label['bar_index']
        )

        # -1代表在范围外，也可能是临界值
        if bar_index == -1:
            self._process_critical_value(item)
            return

        # 最后一根1分钟k线需要计时器辅助推送
        if self.from_tick and bar_index == self.last_locator_index - 1:
            self.use_timer = True

        current_bar_label = dict(date_belong=date_belong, bar_index=bar_index, time_str=time_str)

        # 如果已推送过，则舍弃
        if str(current_bar_label) in self.pushed_bar_labels:
            return

        # 如果当前没有bar, 则用收到的tick/bar新建一个
        if not self.bar:
            self.bar = self._update_bar_by_item(item, None, time_str, new=True)
            # 并且更新current_bar_label
            self.last_bar_label = current_bar_label
            return

        # 如果当前已有bar, 则判断bar是否完整
        # 如果label已改变，且index大于上一个index，则处于临界值，时间判断为左闭右开
        if self.last_bar_label != current_bar_label and bar_index > self.last_bar_label['bar_index']:

            # tick合成k，换barIndex后应算入下一区间
            if self.from_tick:
                self._call_back(self.bar)
                self.pushed_bar_labels.append(str(self.last_bar_label))
                self.last_bar_label = current_bar_label
                self.bar = self._update_bar_by_item(item, None, time_str, new=True)
                return

            # 多分钟K线为了及时推出，换barIndex后更新上一根bar并直接推出
            self.bar = self._update_bar_by_item(
                item, self.bar, time_str, new=False
            )
            self._call_back(self.bar)
            self.pushed_bar_labels.append(str(self.last_bar_label))
            self.bar = None
            return

        # 如果label未改变，则更新bar
        self.bar = self._update_bar_by_item(
            item, self.bar, time_str, new=False
        )

    def _process_critical_value(self, item):
        """处理barIndex为-1的情况
        有如下可能:
        1. tick/bar在时间范围外，直接舍弃
        2. tick/bar在临界值，根据情况处理
        """
        # -----收盘
        if self.last_bar_label['bar_index'] == self.last_locator_index - 1:

            # 使用计时器时延迟收盘，以尽避免因网络延迟丢失tick
            if isinstance(item, TimerEvent):
                self.delay_cnt += 1
                if self.use_timer and self.delay_cnt <= self.delay:
                    logger.debug(f'使用计时器时，延迟收盘，delay_cnt:{self.delay_cnt}')
                    return

            if isinstance(item, BarData):
                self.bar = self._update_bar_by_item(
                    item, self.bar, '', new=not self.bar  # 根据当前有没有bar判断是否要新建一根
                )

            if isinstance(item, TickData):
                pass  # 当前逻辑，Tick时间戳超时应该直接扔，有需要纳入的可以考虑在这里改

            if not self.bar:
                self._fill_empty_bar()

            # 完成收盘，推送最后一根bar
            self._call_back(self.bar)
            self.pushed_bar_labels.append(str(self.last_bar_label))
            self.reset()
            return

        # -----非收盘，跨时间段情况
        # 忽略timerEvent
        if not isinstance(item, (TickData, BarData)):
            return

        # 检查3秒前的情况（通常误差在1s内）
        bar_index, _, time_str = self.locator.locate(
            item.datetime - datetime.timedelta(seconds=3),
            start_from=self.last_bar_label['bar_index']
        )
        # 如果不处于临界值，直接舍弃
        if bar_index != self.last_bar_label['bar_index']:
            return

        # bar已推送并置空，则无需再更新
        if not self.bar:
            return

        # 临界点的tick，算入前一根k，但不推送，由下一根k的tick驱动推送
        if isinstance(item, TickData):
            self.bar = self._update_bar_by_item(item, self.bar, time_str, new=False)
            return

        # 临界点的bar，算入前一根k，推送，并置空self.bar
        if isinstance(item, BarData):
            self.bar = self._update_bar_by_item(item, self.bar, time_str, new=False)
            self._call_back(self.bar)
            self.pushed_bar_labels.append(str(self.last_bar_label))
            self.bar = None

    def reset(self):
        self.use_timer = self._use_timer
        self.bar: BarData = None
        self.last_bar_label = dict(
            date_belong=None, bar_index=-1, time_str='0:0'
        )  # 当前bar所属区间的标签，用于确定bar是否生成完毕

        self.last_high = np.nan  # 最新的最低价
        self.last_low = np.nan  # 最新的最高价
        self.last_tick: TickData = None  # 上一条收到的tick缓存

        # tick时间戳与系统时间的差值，用于时间同步
        self.time_delta = datetime.timedelta(0)

        # 延迟收盘相关
        self.delay = 10
        self.delay_cnt = 0

    def _on_timer(self, event):
        """时间事件驱动，根据情况更新bar推送"""

        if not self.use_timer:
            return

        # 没有收到过tick，则timer无意义
        if not self.last_tick:
            return

        # 模拟交易所时间
        now = datetime.datetime.now() - self.time_delta

        # 生成当前时间对应的label
        bar_index, date_belong, time_str = self.locator.locate(now, start_from=self.last_bar_label['bar_index'])
        current_bar_label = dict(date_belong=date_belong, bar_index=bar_index, time_str=time_str)

        # 不在范围内，可能收盘了
        if bar_index == -1:
            self._process_critical_value(event)
            return

        # 如果时间已过，推送该bar并清空缓存
        if self.last_bar_label != current_bar_label and bar_index > self.last_bar_label['bar_index']:
            # 无行情导致空bar情况
            if not self.bar:
                self._fill_empty_bar()

            self._call_back(self.bar)
            self.pushed_bar_labels.append(str(self.last_bar_label))
            self.last_bar_label = current_bar_label
            self.bar = None

    def _fill_empty_bar(self):
        """根据需求填充k线"""
        if self.auto_fill == 'last':  # 使用最近一个tick填充k线
            last_tick_zero_vol = copy.copy(self.last_tick)
            last_tick_zero_vol.volume = 0
            self.bar = self._update_bar_by_item(last_tick_zero_vol, None, self.last_bar_label['time_str'], new=True)
        elif self.auto_fill == 'nan':  # 使用nan填充k线各值
            self.bar = self._update_bar_by_item(
                source=TickData(
                    symbol=self.last_tick.symbol, exchange=self.last_tick.exchange,
                    datetime=self.last_tick.datetime, name=self.last_tick.name,
                    gateway_name=self.last_tick.gateway_name, last_price=np.nan
                ),
                target=None, new=True,
                time_str=self.last_bar_label['time_str'],
            )
        elif self.auto_fill == 'ignore':
            pass

    def _update_bar_by_item(self, source, target, time_str, new):
        if isinstance(source, BarData):
            return self._update_bar_by_bar(source, target, time_str, new)
        elif isinstance(source, TickData):
            return self._update_bar_by_tick(source, target, time_str, new)
        else:
            return self._update_bar_by_tick(source, target, time_str, new)
            # raise TypeError(f'传入的对象不是Tick也不是Bar对象')

    def _update_bar_by_bar(self, source_bar: BarData, target_bar: BarData, time_str: str, new: bool):
        # If not inited, create window bar object
        if new:
            # Generate timestamp for bar data
            dt = to_date(f"{source_bar.datetime.strftime('%Y%m%d')} {time_str}")

            if dt < source_bar.datetime:  # 0时换日
                dt += datetime.timedelta(days=1)

            target_bar = BarData(
                symbol=source_bar.symbol,
                exchange=source_bar.exchange,
                datetime=dt,
                date=source_bar.date,
                gateway_name=source_bar.gateway_name,
                open_price=source_bar.open_price,
                high_price=source_bar.high_price,
                low_price=source_bar.low_price,
                close_price=source_bar.close_price,
                open_interest=source_bar.open_interest,
                interval=self.cycle,
            )
        # Otherwise, update high/low price into multi minute bar
        else:
            target_bar.high_price = max(
                target_bar.high_price, source_bar.high_price)
            target_bar.low_price = min(
                target_bar.low_price, source_bar.low_price)

        # Update close price/volume into bar
        target_bar.close_price = source_bar.close_price
        target_bar.volume += to_int(source_bar.volume, errors='zero')
        target_bar.open_interest = source_bar.open_interest

        if self.add_amount:
            target_bar.amount += to_float(source_bar.amount, errors='zero')

        return target_bar

    def _update_bar_by_tick(self, tick: TickData, bar: BarData, time_str: str, new: bool):
        """使用tick更新bar"""
        if new:
            # Generate timestamp for bar data
            dt = to_date(f"{tick.datetime.strftime('%Y%m%d')} {time_str}")

            if dt < tick.datetime:  # 0时换日
                dt += datetime.timedelta(days=1)

            _, date_belong, __ = self.date_locator.locate(dt)
            bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=self.cycle,
                datetime=dt,
                date=date_belong,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest,
            )
        else:
            bar.low_price = min(bar.low_price, tick.last_price)
            bar.high_price = max(bar.high_price, tick.last_price)

            bar.close_price = tick.last_price
            bar.open_interest = tick.open_interest

        if tick.low_price and tick.low_price < self.last_low:
            bar.low_price = min(bar.low_price, tick.low_price)
        self.last_low = tick.low_price

        if bar.high_price and tick.high_price > self.last_high:
            bar.high_price = max(bar.high_price, tick.high_price)
        self.last_high = tick.high_price

        bar = self._sum_up_volume(tick, bar)

        if self.add_amount:
            bar.amount += to_float(tick.last_price * tick.volume, errors='zero')

        self.last_tick = tick

        return bar

    def _sum_up_volume(self, source, target):
        """累积量"""
        # xswap和bar的volume每个tick是独立的
        if isinstance(source, BarData) or source.gateway_name == 'XSWAP' or source.gateway_name == 'XBOND':
            target.volume += source.volume
            return target

        # CTP是累积量
        if source.gateway_name == 'CTP':
            if self.last_tick:
                volume_change = source.volume - self.last_tick.volume
            else:
                volume_change = source.volume
            target.volume += max(volume_change, 0)
            return target

        logger.warning(f'gateway:{source.gateway_name}未能识别，无法进行成交量加总，请检查BarManager._sum_up_volume方法')
        return target

    def _call_back(self, bar: BarData):
        """使用用户的回调函数推送bar对象"""
        if bar:  # 过滤空bar（None）
            self.callback(bar)

    @property
    def _cycle(self):
        """将'Day'以外的cycle转为int，方便计算"""
        return to_int(self.cycle, errors='ignore')

    def generate(self):
        """
        推送现有的bar
        """
        bar = copy.copy(self.bar)
        self._call_back(bar)


class BarManager:
    """
    K线生成管理：加载配置，获取订阅，生成k线，储存到redis
    """

    def __init__(self, callback, start='$', open_offset=-3, close_offset=0):
        """
        初始化K线生成管理器，给定一个开始订阅TICK的时间，默认最后一条开始

        """

        self.pending_to_run = []
        self.subscribed = {}  # 需要订阅的合约代码

        self.irs_callback = encapsulate_irs_tick_callback(self.sub_tick_callback)
        self.future_callback = encapsulate_future_tick_callback(self.sub_tick_callback)
        self.xbond_callback = encapsulate_xbond_tick_callback(self.sub_tick_callback)

        self.event_engine: EventEngine = EventEngine()

        # self.callback = callback
        self.callback = callback

        # tick从什么时候开始订阅
        self.start = start

        self.open_offset = open_offset
        self.close_offset = close_offset

        self.bar_list = []

    def set_param(self, market: str, code: str, cycle: str,
                  from_tick=False, use_timer=False,
                  main_force=True, auto_fill='last',
                  add_amount=False):
        """
        设置单品种单周期k线生成的参数，不启动订阅

        Parameters
        ----------
        market: 市场，目前支持’IRS‘和’FUTURE‘和’XBOND‘
        code: 合约代码，如订阅期货特定月份合约，则需要给定月份代码，并设置main_force为False
        cycle: str, k线周期, 分钟数或'Day'
        from_tick: 直接使用tick生成该周期的k线
        use_timer: 是否同时使用时间事件驱动
        main_force: 是否自动获取主力合约
        auto_fill: 如何填充空bar，详见BarGenerator
        add_amount: 是否在K线中增加[成交额]字段，=成交价*成交量

        """
        if market not in ['IRS', 'FUTURE', 'XBOND']:
            raise ValueError("market仅支持IRS与FUTRUE")

        if market == 'IRS':
            code_set = self.subscribed.setdefault('IRS', set())
            code_set.add(code)
            holiday_mode = 'interbank'
        elif market == 'FUTURE':
            if main_force:
                code = get_future_main_contract(
                    code,
                    # 夜盘启动，取第二天的主力
                    date=datetime.date.today() + datetime.timedelta(days=1)
                )
            code_set = self.subscribed.setdefault('FUTURE', set())
            code_set.add(code.upper())
            holiday_mode = 'exchange'
        else:
            code_set = self.subscribed.setdefault('XBOND', set())
            code_set.add(code)
            holiday_mode = 'exchange'

        # 1分钟k线必须使用tick来生成
        if str(cycle) == '1':
            from_tick = True

        self.pending_to_run.append({
            'code': code,
            'market': market,
            'use_timer': use_timer,
            'from_tick': from_tick,
            'cycle': cycle,
            'auto_fill': auto_fill,
            'add_amount': add_amount,
            'holiday_mode': holiday_mode,
        })

    def set_params(self, settings: dict):
        """
        一次性给定多个参数，自动解包
        settings字典同一个市场只能提供一种from_tick/use_timer参数，
        有特殊需求可调用set_param方法，按单品种单周期进行设置

        Examples：
        params = {
            'IRS': {
                'code': ['FR007_5Y', 'FR007_1Y'],
                'cycle': ['5', '15', '30', '60'],
                'use_timer': True,
                'from_tick': True,
            },
            'FUTURE': {
                'code': ['IF', 'IC', 'IH'],
                'cycle': ['30', '60']
            }
            'XBOND': {

            }
        }
        """
        for market, info in settings.items():
            use_timer = info.get('use_timer', False)
            from_tick = info.get('from_tick', False)
            code_list = info.get('code', [])
            cycle_list = info.get('cycle', ['1'])
            auto_fill = info.get('auto_fill', 'last')
            main_force = info.get('main_force', True)
            add_amount = info.get('add_amount', False)

            if cycle_list and not from_tick and ('1' not in cycle_list):
                cycle_list.append('1')

            for code in code_list:
                for cycle in cycle_list:
                    self.set_param(
                        market=market, code=code, cycle=cycle,
                        from_tick=from_tick, use_timer=use_timer,
                        auto_fill=auto_fill, main_force=main_force,
                        add_amount=add_amount,
                    )

    def sub_tick_callback(self, tick: TickData):
        """用于行情tick订阅的回调函数"""
        event = TickEvent(type_=f'TICK.{tick.symbol}')
        event.dict_['data'] = tick
        self.event_engine.put(event)

    def register_events(self):
        """使用BarGenerator监听对应的事件，生成K线"""
        for record in self.pending_to_run:
            self.register_event(
                record['code'], record['cycle'], record['use_timer'],
                record['from_tick'], record['auto_fill'],
                record['add_amount'], record['holiday_mode'],
            )

    def register_event(self, code, cycle, use_timer, from_tick, auto_fill, add_amount, holiday_mode):
        """使用BarGenerator监听对应的事件，生成K线"""
        if from_tick:
            event_name = f'TICK.{code}'
        else:
            event_name = f'BAR.1.{code}'  # 对应品种的1分钟K线

        cycle = cycle if cycle == 'Day' else int(cycle)

        bg = BarGenerator(
            code=code, cycle=cycle, use_timer=use_timer, callback=self.bar_callback,
            from_tick=from_tick,
            open_offset=self.open_offset,  # 计入开盘前集合竞价
            close_offset=self.close_offset,  # 提前生成收盘k线以执行信号
            auto_fill=auto_fill, add_amount=add_amount,
            holiday_mode=holiday_mode,
        )

        self.event_engine.register(type_=event_name, handler=bg.update)
        self.event_engine.register(type_=EVENT_TIMER, handler=bg.update)

    def sub_ticks(self):
        """订阅所有self.subscribed里面的合约"""

        if self.subscribed.get('IRS'):
            sub_irs_tick_clean(
                code=self.subscribed.get('IRS'), callback=self.irs_callback,
                start=self.start, block_abnormal=True, unify=True,
                quote_spread=1, outlier=(3, 2),
            )

        if self.subscribed.get('FUTURE'):
            sub_future_tick(
                codes=self.subscribed.get('FUTURE'), fun=self.future_callback, start=self.start
            )

        if self.subscribed.get('XBOND'):
            sub_bond_tick_xbond(
                code=self.subscribed.get('XBOND'), fun=self.xbond_callback, start=self.start
            )

    def bar_callback(self, bar: BarData):
        """使用用户回调推送bar，1min-bar额外重新推送至事件引擎"""
        if str(bar.interval) == '1':
            event = BarEvent(type_=f'BAR.{bar.interval}.{bar.symbol.upper()}')
            event.dict_['data'] = bar
            self.event_engine.put(event)

        self.callback(bar)  # 用户回调
        # self.bar_list.append(bar)


    def run(self):
        self.event_engine.start(timer_interval=1)
        self.register_events()
        self.sub_ticks()
        self.pending_to_run = []
        logger.info('Bar Manager started')


if __name__ == '__main__':

    bm = BarManager(print, start='9:00', close_offset=0)

    bm.set_param('XBOND', 'IB220402', '15', from_tick=True, use_timer=True, add_amount=True)
    bm.set_param('XBOND', 'IB220405', '15', from_tick=True, use_timer=True, add_amount=True)
    bm.set_param('XBOND', 'IB220016', '15', from_tick=True, use_timer=True, add_amount=True)
    bm.set_param('XBOND', 'IB220017', '5', from_tick=True, use_timer=True, add_amount=True)
    bm.set_param('XBOND', 'IB220208', '5', from_tick=True, use_timer=True, add_amount=True)
    bm.set_param('XBOND', 'IB220210', '5', from_tick=True, use_timer=True, add_amount=True)

    # bm.set_param('FUTURE', 'IF', '15', from_tick=False)
    # bm.set_param('FUTURE', 'IF', '30', from_tick=False)
    # bm.set_param('FUTURE', 'IF', '60', from_tick=False)
    # bm.set_param('FUTURE', 'IF', '1', from_tick=True)
    #
    # bm.set_param('FUTURE', 'I', '1', from_tick=True)
    # bm.set_param('FUTURE', 'I', '5', from_tick=False)
    # bm.set_param('FUTURE', 'I', '15', from_tick=False)
    # bm.set_param('FUTURE', 'I', '30', from_tick=False)
    # bm.set_param('FUTURE', 'I', '60', from_tick=False)
    #
    # bm.set_param('FUTURE', 'T', '1', from_tick=True)
    # bm.set_param('FUTURE', 'T', '5', from_tick=False)
    # bm.set_param('FUTURE', 'T', '15', from_tick=False)
    # bm.set_param('FUTURE', 'T', '30', from_tick=False)
    # bm.set_param('FUTURE', 'T', '60', from_tick=False)
    #
    # bm.set_param('FUTURE', 'I', '1', from_tick=True)
    # bm.set_param('FUTURE', 'I', '5', from_tick=False)
    # bm.set_param('FUTURE', 'I', '15', from_tick=False)
    # bm.set_param('FUTURE', 'I', '30', from_tick=False)
    # bm.set_param('FUTURE', 'I', '60', from_tick=False)
    # bm.event_engine.register('TICK.FR007_5Y', print)

    bm.run()

    # 5min tick to bar 已通
