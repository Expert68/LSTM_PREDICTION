import pandas as pd
import tushare as ts
from abupy import ABuSymbolPd

'---------------------------------------------------'
#获取股价数据
def get_symbol():
    stock_list = []
    with open(r'D:\LSTM_PREDICTION\data\stock.txt') as f:
        for stock in f:
            stock = stock[1:7]
            stock_list.append(stock)
    return stock_list

stock_list = get_symbol()

# 对股票数据进行重构
# 定义函数，从df中取指定的列
# 预测第二天的平均价格，所以对DataFrame进行相应的处理
def reformat_df(df):
    new_df = df[['open','high','low','close','volume']]
    new_df['xopen'] = new_df['open'].shift(-1)
    new_df['xclose'] = new_df['close'].shift(-1)
    return new_df



# 将重构后的股票数据写入stock_data文件夹中
for stock in stock_list:
    df = ABuSymbolPd.make_kl_df(stock,n_folds=4)
    # df = ts.get_hist_data(stock)
    if df is not None:
        new_df = reformat_df(df)
        new_df.to_csv(r'stock_data\%s.csv' %stock)
        print('%s is written' %stock)


'---------------------------------------------------'



