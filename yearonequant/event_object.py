
ALL_EVENTS = dict()
EVENT_NAME_MAP = {'holding_increase': '增持',
                  'holding_decrease': '减持',
                  'forecast_growth': '预增',
                  'forecast_decline': '预减'}


class EventDict:
    def __init__(self, name, target_words, filter_words, filter_mode):
        self.name = name
        self.chinese_name = EVENT_NAME_MAP.get(name)
        self.target_words = target_words
        self.filter_words = filter_words
        self.filter_mode = filter_mode


def append_event(event_dict):
    """
    Put into ALL_EVENTS.
    :param event_dict:  dict of event
    :return: void
    """
    ALL_EVENTS[event_dict.name] = event_dict


# 增持
holding_increase_target = ['增持']
holding_increase_filter = ['完成', '误操作', '倡议书', '进展', '核查意见', '补充',
                           '完毕', '法律意见书']
holding_increase_mode = 'OR'

holding_increase = EventDict("holding_increase", holding_increase_target,
                             holding_increase_filter, holding_increase_mode)
append_event(holding_increase)

# 减持
holding_decrease_target = ['减持']
holding_decrease_filter = ['不减持', '不存在减持', '终止', '限制', '购回', '回购',
                           '完成', '更正', '更新', '完毕', '用于', '进展', '法律意见书']
holding_decrease_mode = 'OR'

holding_decrease = EventDict("holding_decrease", holding_decrease_target,
                             holding_decrease_filter, holding_decrease_mode)
append_event(holding_decrease)

# 预增
forecast_growth_target = ['预增', '年度']
forecast_growth_filter = ['更正', '修正', '更改', '补充']
forecast_growth_mode = 'AND'

forecast_growth = EventDict("forecast_growth", forecast_growth_target,
                            forecast_growth_filter, forecast_growth_mode)
append_event(forecast_growth)

# 预减
forecast_decline_target = ['预减', '年度']
forecast_decline_filter = ['更正', '修正', '更改', '补充']
forecast_decline_mode = 'AND'

forecast_decline = EventDict("forecast_decline", forecast_decline_target,
                             forecast_decline_filter, forecast_decline_mode)
append_event(forecast_decline)
