# 股票策略模版
# 初始化函数,全局只运行一次

def init(context):
    # 设置基准收益：沪深300指数
    set_benchmark('000300.SH')
    # 打印日志
    log.info('策略开始运行,初始化函数全局只运行一次')
    # 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
    set_commission(PerShare(type='stock',cost=0.0002))
    # 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
    set_slippage(PriceSlippage(0.005))
    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    # 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
    # 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
    set_volume_limit(0.25,0.5)
    
    context.buy_temp_time = {}
    context.buy_temp_price = {}
    context.temp_time = 19
    
    # 设置要操作的股票
    context.security = get_index_stocks('000300.SH')
    # n1为模型买入启动或卖出启动形态形成时的价格比较滞后期数
    # n2为模型买入启动或卖出启动形态形成的价格关系单向连续个数
    # n3为模型计数阶段的最终信号发出所需的计数值。
    # n4为计算止损点的ATR系数
    # n5为计算第一次加仓价的ATR系数
    # n6为计算止盈点的ATR系数
    # n7为计算第二次加仓价的ATR系数
    context.n1 = 3
    context.n2 = 3
    context.n3 = 2
    context.n4 = 1.5
    context.n5 = 0.4
    context.n6 = 2.5
    context.n7 = 0.4
    # 回测区间、初始资金、运行频率请在右上方设置
    
    # 计数期间最高价与最低价

#每日开盘前9:00被调用一次,用于储存自定义参数、全局变量,执行盘前选股等
def before_trading(context):

    # 获取日期
    date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')

    # 打印日期
    log.info('{} 盘前运行'.format(date))

#因子数据缺失填补函数
def fill_ndarray(t1):
    import numpy as np
    for i in range(t1.shape[1]):  # 遍历每一列（每一列中的nan替换成该列的均值）
        temp_col = t1[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            temp_not_nan_col = temp_col[temp_col == temp_col]  # 去掉nan的ndarray
            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # mean()表示求均值。
    return t1

## 开盘时运行函数
def handle_bar(context, bar_dict):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.decomposition import PCA
    import time
    import datetime
    import math
    print(len(context.security))
    context.temp_time += 1
    if context.temp_time == 20:
        context.temp_time = 0
        # 获取股票代码
        print('获取股票代码')
        stock_list = get_index_stocks('000300.SH')

        #剔除回测当日前6个月内上市的股票
        time1 = get_datetime()-datetime.timedelta(days=180)
        time1 = time1.strftime('%Y-%m-%d')
        for s in stock_list:
            info = get_security_info(s)
            # print(str(info.start_date)[0:10])
            if str(info.start_date)[0:10] > time1:
                print(time1 + ' ' + str(info.start_date)[0:10])
                stock_list.remove(s)

        # 设置训练时间
        train_time = 60

        # 设置获取数据截止的时间
        train_date_end = '20200530'

        # 获取回测前一天时间
        predict_date = get_datetime()-datetime.timedelta(days=1)
        predict_date = predict_date.strftime('%Y%m%d')

        # 获取股票信息，以字典形式保存
        print('获取股票信息，以字典形式保存')
        b = get_candle_stick(stock_list, end_date=predict_date, fre_step='1d', fields=['close'], skip_paused=False, fq='pre', bar_count=train_time, is_panel=1)
        c = get_candle_stick(stock_list, end_date=predict_date, fre_step='1d', fields=['close'], skip_paused=False, fq='pre', bar_count=1, is_panel=1)

        result = {}                   # key为股票名，value为index为到设定时间60天有有效数据的时间，columns为收盘价的dataframe
        all_stocks = {}               # key为股票名，value为index为预测涨跌的时间，columns为收盘价的dataframe
        for item in stock_list:
            temp = b.minor_xs(item).loc[:,['close']]
            temp1 = c.minor_xs(item).loc[:,['close']]
            result[item] = temp
            all_stocks[item] = temp1

        # 获取开始训练的时间
        train_date_start = str(result[item].index[0])[0:10]
        train_date_start = datetime.datetime.strptime(train_date_start, '%Y-%m-%d').strftime('%Y%m%d')

        # 获取结束训练的时间
        train_date_end = str(result[item].index[int(train_time/2)-1])[0:10]
        train_date_end = datetime.datetime.strptime(train_date_end, '%Y-%m-%d').strftime('%Y%m%d')

        #剔除股票池中在训练和预测期间停牌股票和ST股
        judge=get_price(stock_list,train_date_start,predict_date,'1d',['is_paused','is_st'])
        for s in stock_list:
            stock_spare=stock_list
            for row in judge[s].iterrows():
                if row[1].loc['is_paused']==1 or row[1].loc['is_st']==1:
                    stock_spare.remove(s)
                    break
        stock_list=stock_spare

        #重新获取股票信息
        result = {}                   # key为股票名，value为index为到设定时间60天有有效数据的时间，columns为收盘价的dataframe
        all_stocks = {}               # key为股票名，value为index为预测涨跌的时间，columns为收盘价的dataframe
        for item in stock_list:
            temp = b.minor_xs(item).loc[:,['close']]
            temp1 = c.minor_xs(item).loc[:,['close']]
            result[item] = temp
            all_stocks[item] = temp1

        # 添加label（一个月收益率）
        print('添加label（一个月收益率）')
        for key in result:
            temp = result[key]
            list_temp_time = list(temp.index)
            temp['label'] = np.nan
            for i in range(int(train_time/2)):
                temp['label'][list_temp_time[i]] = (temp['close'][list_temp_time[i+30]]-temp['close'][list_temp_time[i]])/temp['close'][list_temp_time[i]]
            temp.dropna(axis=0, how='any', inplace=True)

        #获取对应股票因子,归一化处理
        print('获取对应股票因子,归一化处理')
        factor_list = ['bbi','ma','expma','dbcd','wad','obv','bbiboll','boll','cdp','env','vstd','micd','pb','pcf_cash_flow_ttm','ps','ps_ttm']
        for key in list(result.keys()):
            result_spare=result
            train = result[key]             # 对应股票的日期及其收盘价和未来一个月收益率
            test = all_stocks[key]          # 需要预测股票回测日收盘价

            # 获取对应股票在训练期间内的因子数据
            dict_train = get_sfactor_data(start_date=train_date_start,end_date=train_date_end,stocks=[key],factor_names=factor_list)
            
            #剔除训练期间因子数据缺失的股票
            train_nan=0
            for i in range(16):
                if math.isnan(dict_train[factor_list[i]].iloc[0,0]) or math.isinf(dict_train[factor_list[i]].iloc[0,0]):
                    train_nan=train_nan+1
                    
            if train_nan!=0:    
                del result_spare[key]
                stock_list.remove(key)
                print('训练时间剔除',key)
                continue

            # 获取对应股票在预测日期的因子数据
            temp_i = 1
            while 1:
                try:
                    dict_test = get_sfactor_data(start_date=predict_date,end_date=predict_date,stocks=[key],factor_names=factor_list)
                except IndexError:
                    temp_i += 1
                    predict_date = get_datetime()-datetime.timedelta(days=temp_i)
                    predict_date = predict_date.strftime('%Y%m%d')
                else:
                    break

             #剔除回测前一天因子数据缺失的股票
            predict_nan=0
            for i in range(16):
                if math.isnan(dict_test[factor_list[i]].iloc[0,0]) or math.isinf(dict_test[factor_list[i]].iloc[0,0]):
                    predict_nan=predict_nan+1
                    
            if predict_nan!=0:    
                del result_spare[key]
                stock_list.remove(key)
                print('回测时间剔除',key)
                continue
 
            #数据归一化处理   
            for key1 in dict_train:
                train[key1] = dict_train[key1].values.T
                test[key1] = dict_test[key1].values.T
                MIN = train.loc[:,key1].min()
                MAX = train.loc[:,key1].max()
                if MAX-MIN!=0:
                    train.loc[:,key1] = (train.loc[:,key1]-MIN)/(MAX-MIN)
                    test.loc[:,key1] = (test.loc[:,key1]-MIN)/(MAX-MIN)
            train.drop(['close'],axis=1,inplace=True)
            test.drop(['close'],axis=1,inplace=True)
        print('完成因子数据归一化操作')
        result=result_spare

        # 生成训练集和测试集
        print('准备生成训练集和测试机')
        train_label = np.zeros((2,1))
        train_data = np.zeros((2,16)) #根据因子数变动
        test_data = np.zeros((2,16))
        for key in result:
            train = result[key]
            test = all_stocks[key]
            temp1 = train.loc[:,'label'].values
            temp1 = np.reshape(temp1,(30,1))
            temp2 = train.loc[:,factor_list].values #根据因子数变动
            temp3 = test.loc[:,factor_list].values
            train_label = np.concatenate((train_label,temp1),axis=0)
            train_data = np.concatenate((train_data,temp2),axis=0)
            test_data = np.concatenate((test_data,temp3),axis=0)

        train_label = np.delete(train_label,(0,1),axis=0)
        train_label = np.reshape(train_label,(len(stock_list)*30))
        train_data = np.delete(train_data,(0,1),axis=0)
        test_data = np.delete(test_data,(0,1),axis=0)

        # 因子数据缺失填补
        print('因子数据缺失值填补')
        train_data = fill_ndarray(train_data)

        # 开始训练和预测
        print('开始训练和预测')
        pca = PCA(n_components=10,copy=True,whiten=False)
        pca.fit(train_data)
        new_train_data = pca.transform(train_data)
        new_test_data = pca.transform(test_data)
        Ada1 = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy'),n_estimators=15,algorithm="SAMME.R",learning_rate=0.2)
        train_label = train_label * 100
        Ada1.fit(new_train_data,train_label.astype(int))
        test_predict = Ada1.predict(new_test_data)

        select_stocks = []
        stocks_dict={}
        for i in range(len(test_predict)):
            stocks_dict[stock_list[i]] = test_predict[i]
        stocks_dict = sorted(stocks_dict.items(), key=lambda d: d[1], reverse=True) 
        for i in range(30):
            if stocks_dict[i][1]>0:
                select_stocks.append(stocks_dict[i][0])
        print(select_stocks)
        context.security = select_stocks
    
    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')

    # 打印时间
    log.info('{} 盘中运行'.format(time))
    
    for symbol in context.security:
    # 获取股票过去20天的收盘价数据
        closeprice = history(context.security, ['close'], 20, '1d', False, 'pre', is_panel=1)
        highprice = history(context.security, ['high'], 20, '1d', False, 'pre', is_panel=1)
        lowprice = history(context.security, ['low'], 20, '1d', False, 'pre', is_panel=1)
        count_highest_price = 0
        count_minimum_price = 999999999999
        # 获取账户持仓股票列表
        stocklist = list(context.portfolio.stock_account.positions)
        
        if symbol in stocklist:
            
            ud = get_ud(symbol, closeprice, context.n1)
            flag, start = sell_out_start_count(symbol, ud, 0, context.n2)
            if flag == 1:
                sell_out_time = sell_out_signal_count(symbol, start, ud, closeprice, highprice, lowprice, context.n3)
                if sell_out_time != -1:
                # 得到卖出信号，卖出所有本只股票,使这只股票的最终持有量为0
                   order_target(symbol, 0)
                   log.info('得到卖出信号，卖出股票：%s' % symbol)
        else:
            ud = get_ud(symbol, closeprice, context.n1)
            flag, start = buy_start_count(symbol, ud, 0, context.n2)
            if flag == 1:
                buy_time = buy_signal_count(symbol, start, ud, closeprice, highprice, lowprice, context.n3)
                if buy_time != -1:
                # 按金额下单,异常处理
                    try:
                        id = order_value(symbol, 400000, price=None, style=None)
                        log.info('得到买入信号，买入股票：%s' % symbol)
                        context.buy_temp_price[symbol] = get_order(id).avg_price
                    except AttributeError:
                        continue
                    else:
                        context.buy_temp_time[symbol] = time

    log.info('{} 盘中运行完成'.format(time))

## 收盘后运行函数,用于储存自定义参数、全局变量,执行盘后选股等
def after_trading(context):
    # 获取时间
    time = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    # 打印时间
    log.info('{} 盘后运行'.format(time))
    log.info('一天结束')


def get_ud(symbol, price_close, n1=4):
    # 获得指定股票ud序列
    ud = []
    for i in range(n1,len(price_close['close'])):
        if price_close['close'][symbol][i]-price_close['close'][symbol][i-n1] > 0:
            ud.append(1)
        elif price_close['close'][symbol][i]-price_close['close'][symbol][i-n1] < 0:
            ud.append(-1)
        else:
            ud.append(0)
    return ud

def sell_out_start_count(symbol, ud, start, n2=4):
    #卖出_启动计数
    count = ud[0]
    for i in range(start, len(ud)):
        if ud[i] != ud[i-1]:
            count = ud[i]
        else:
            count += ud[i]
        if count == n2:
            # 返回flag 1为卖出启动，i为卖出启动时间
            return 1, i
    return 0, 0
        
def buy_start_count(symbol, ud, start, n2=4):
    #买入_启动计数
    count = ud[0]
    for i in range(start, len(ud)):
        if ud[i] != ud[i-1]:
            count = ud[i]
        else:
            count += ud[i]
        if count == -n2:
            # 返回flag 1为卖入启动，i为卖入启动时间
            return 1, i
    return 0, 0

def buy_signal_count(symbol, start, ud, price_close, price_high, price_low, n3=4):
    #买入信号_计数
    global count_minimum_price
    flag,end = buy_start_count(symbol, ud, start)
    first_temp = -1
    temp = 0
    if flag == 1:
        count_minimum_price = min(tuple(price_low['low'][symbol][start:end]))
        while True:
            for i in range(start,end):
                if price_close['close'][symbol][i] >= price_high['high'][symbol][start+1]:
                    if price_high['high'][symbol][i] > price_high['high'][symbol][i-1]:
                        first_temp = i
                        temp = 1
                        break;
            break
        if first_temp == -1:
            return -1
        for i in range(first_temp, end):
            if price_close['close'][symbol][i] >= price_high['high'][symbol][start+1]:
                    if price_high['high'][symbol][i] > price_high['high'][symbol][i-1]:
                        if price_close['close'][symbol][i] > price_close['close'][symbol][first_temp]:
                            temp += 1
                            if temp == n3:
                                return i
        return -1
    elif flag == 0:
        count_minimum_price = min(tuple(price_low['low'][symbol]))
        while True:
            for i in range(start,len(price_close)):
                if price_close['close'][symbol][i] >= price_high['high'][symbol][start+1]:
                    if price_high['high'][symbol][i] > price_high['high'][symbol][i-1]:
                        first_temp = i
                        temp = 1
                        break;
            break
        if first_temp == -1:
            return -1
        for i in range(first_temp, end):
            if price_close['close'][symbol][i] >= price_high['high'][symbol][start+1]:
                    if price_high['high'][symbol][i] > price_high['high'][symbol][i-1]:
                        if price_close['close'][symbol][i] > price_close['close'][symbol][first_temp]:
                            temp += 1
                            if temp == n3:
                                return i
        return -1

def sell_out_signal_count(symbol, start, ud, price_close, price_high, price_low, n3=4):
    #卖出信号_计数
    global count_highest_price
    flag, end = sell_out_start_count(symbol, ud, start)
    first_temp = -1
    temp = 0
    if flag == 1:
        count_highest_price = max(tuple(price_high['high'][symbol][start:end]))
        while True:
            for i in range(start,end):
                if price_close['close'][symbol][i] <= price_low['low'][symbol][start+1]:
                    if price_low['low'][symbol][i] < price_low['low'][symbol][i-1]:
                        temp = 1
                        first_temp = i
                        break;
            break
        if first_temp == -1:
            return -1
        for i in range(first_temp, end):
            if price_close['close'][symbol][i] <= price_low['low'][symbol][start+1]:
                    if price_low['low'][symbol][i] < price_low['low'][symbol][i-1]:
                        if price_close['close'][symbol][i] < price_close['close'][symbol][first_temp]:
                            temp += 1
                            if temp == n3:
                                return i
        return -1
    elif flag == 0:
        count_highest_price = max(tuple(price_high['high'][symbol]))
        while True:
            for i in range(start,len(price_close)):
                if price_close['close'][symbol][i] <= price_low['low'][symbol][start+1]:
                    if price_low['low'][symbol][i] < price_low['low'][symbol][i-1]:
                        first_temp = i
                        temp = 1
                        break;
            break
        if first_temp == -1:
            return -1
        for i in range(first_temp, end):
            if price_close['close'][symbol][i] <= price_low['low'][symbol][start+1]:
                    if price_low['low'][symbol][i] < price_low['low'][symbol][i-1]:
                        if price_close['close'][symbol][i] < price_close['close'][symbol][first_temp]:
                            temp += 1
                            if remp == n3:
                                return i
        return -1

