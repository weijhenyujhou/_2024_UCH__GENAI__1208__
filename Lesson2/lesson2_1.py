# 台幣轉美金函數
def currency_converter(TWD_dollar):
    exchange_rate = 0.035
    USdollar = TWD_dollar*exchange_rate
    return USdollar


currency_converter(1000)
