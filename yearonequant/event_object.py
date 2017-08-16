
ALL_EVENTS = dict()

# English to Chinese
EVENT_NAME_E2C = {'holding_increase': '增持',
                  'holding_decrease': '减持',
                  'forecast_growth': '预增',
                  'forecast_decline': '预减',
                  'research': '调研',
                  'high_stock_dividend': '高送转',
                  'major_asset_restructure': '重大资产重组',
                  'rename': '更名'}
# Chinese to English
EVENT_NAME_C2E = {'增持': 'holding_increase',
                  '减持': 'holding_decrease',
                  '预增': 'forecast_growth',
                  '预减': 'forecast_decline',
                  '调研': 'research',
                  '高送转': 'high_stock_dividend',
                  '重大资产重组': 'major_asset_restructure',
                  '更名': 'rename'}


class EventDict:
    def __init__(self, name, target_words, filter_words, filter_mode):
        self.name = name
        self.chinese_name = EVENT_NAME_E2C.get(name)
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
forecast_growth_mode = 'OR'

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

# 调研
research_target = ['调研', '投资者关系活动']
research_filter = ['完成', '完毕', '办法', '制度']
research_mode = 'OR'

research = EventDict("research", research_target,
                     research_filter, research_mode)
append_event(research)

# 高送转
high_stock_dividend_target = ['高送转']
high_stock_dividend_filter = ['更正', '修正', '更改', '补充', '问询', '取消']
high_stock_dividend_mode = 'OR'

high_stock_dividend = EventDict("high_stock_dividend", high_stock_dividend_target,
                                high_stock_dividend_filter, high_stock_dividend_mode)
append_event(high_stock_dividend)

# 重大资产重组
major_asset_restructure_target = ['重大资产重组']
major_asset_restructure_filter = ['更正', '修正', '更改', '补充', '问询', '取消', '停牌',
                                  '复牌', '延期', '进展', '终止', '意见', '承诺函', '规定']
major_asset_restructure_mode = 'OR'

major_asset_restructure = EventDict("major_asset_restructure", major_asset_restructure_target,
                                    major_asset_restructure_filter, major_asset_restructure_mode)
append_event(major_asset_restructure)

# 更名
rename_target = ['更名']
rename_filter = ['控股', '债券', '附属', '下属', '代管', '股东', '进展']
rename_mode = 'OR'

rename = EventDict("rename", rename_target,
                   rename_filter, rename_mode)
append_event(rename)
