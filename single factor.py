from scipy.io import loadmat
import pandas as pd
import numpy as np

# load data
data = loadmat('D:/CSLv1.mat')
industry = pd.DataFrame(data['CSLv1'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
dateMap = pd.DataFrame([element[0][0] for element in data['date']], index=data['date_num'].T[0], columns=['date'])
data = loadmat('D:/MV.mat')
MV = pd.DataFrame(data['MV'],index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
data = loadmat('D:/NPParentCompanyOwners.mat')
NPTTM = pd.DataFrame( data['NPParentCompanyOwners_TTM'], index=data['Report_dates_num'].T[0], columns=[element[0][0] for element in data['RICs']])
dateMap2 = pd.DataFrame([element[0][0] for element in data['Report_dates']], index=data['Report_dates_num'].T[0], columns=['date'])
dateMap = dateMap.append(dateMap2)
dateMap.drop_duplicates(inplace=True)
data = loadmat('D:/DayPrice_Stock.mat')
close = pd.DataFrame(data['ClosePrice'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
status = pd.DataFrame(data['TradeStatus'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
AF = pd.DataFrame(data['AF'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
high = pd.DataFrame(data['HighPrice'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
low = pd.DataFrame(data['LowPrice'], index=data['date_num'].T[0], columns=[element[0][0] for element in data['RICs']])
adjClose = close*AF
close_pct = adjClose.fillna(method='ffill').pct_change()
close_pct.iloc[0]=0
close_pct.fillna(0, inplace=True)
close_pct = close_pct+1
dateMap2 = pd.DataFrame([element[0][0] for element in data['date']], index=data['date_num'].T[0], columns=['date'])
dateMap = dateMap.append(dateMap2)
dateMap.drop_duplicates(inplace=True)
dateMap['date'] = pd.to_datetime(dateMap['date'],infer_datetime_format=True)
dateDict = dateMap.squeeze().to_dict()
data = loadmat('D:/IPO_date.mat')
IPO = pd.DataFrame([str(element[0][0]) for element in data['IPO_date']],index=[element[0][0] for element in data['RICs']], columns=['date'])
IPO['date'] = pd.to_datetime(IPO['date'], infer_datetime_format=True,errors='coerce')
IPO.dropna(inplace=True)

# some functions for conputation
def winsorization(x, md, mad):
    if x > (md+5*mad):
        return md+5*mad
    elif x < (md-5*mad):
        return md-5*mad
    else:
        return x

def outlier_modify(data):
    md = data.median()
    from scipy.stats import median_abs_deviation
    mad = median_abs_deviation(data.dropna())
    data.map(lambda x:winsorization(x,md,mad))
    return data

def fill_na_by_industry_median(data, industry):
    industry.rename('ind',inplace=True)
    data.rename('value', inplace=True)
    if data.isna().sum():
        industry = industry[industry != 0]
        data = pd.merge(data, industry, how='left', left_index=True, right_index=True)
        data.dropna(axis=0, inplace=True)
        data = data.groupby('ind').transform(lambda x:x.fillna(x.median()))
    return data

def normalize(data):
    return (data-data.mean())/data.std()

def calculate_ep(date_num):
    common_index = MV.columns & NPTTM.columns
    current_mv = MV.loc[date_num, common_index]
    current_npttm = NPTTM.loc[:date_num, common_index].iloc[-1]
    current_ep = current_npttm / current_mv
    current_ep = outlier_modify(current_ep)
    current_ep = fill_na_by_industry_median(current_ep, industry.loc[date_num])
    # .dropna().rename('value')
    current_ep = current_ep.squeeze().rename('value')
    stocks = current_ep.index
    current_mv = MV.loc[date_num, stocks].rename('mv')
    current_ind = industry.loc[date_num, stocks].rename('ind')
    current_ind = current_ind[current_ind != 0]
    sheet = pd.concat([current_ep, current_mv, current_ind], axis=1)
    sheet.value = normalize(sheet.value)
    sheet.mv = normalize(outlier_modify(sheet.mv))
    sheet.dropna(inplace=True)
    from statsmodels.formula.api import ols
    fit = ols('mv ~ C(ind)', data=sheet).fit()
    sheet.mv = fit.resid
    fit = ols('value ~ mv +  C(ind) -1 ', data=sheet).fit()
    return fit.resid

def calculate_mv(date_num):
    current_mv = np.log(MV.loc[date_num])
    current_mv = outlier_modify(current_mv).dropna().rename('value')
    current_mv = normalize(current_mv)
    current_ind = industry.loc[date_num, current_mv.index].rename('ind')
    current_ind = current_ind[current_ind != 0]
    sheet = pd.concat([current_mv, current_ind], axis=1)
    from statsmodels.formula.api import ols
    fit = ols('value ~ C(ind) ', data=sheet.dropna()).fit()
    return fit.resid

def get_tradable_list(data):
    return list(data[data != 0].index)

def get_group(data,N,i):
    interval = data.shape[0]/N
    start = round(i*interval)
    end = round((i+1)*interval)
    return data.sort_values(ascending=True).iloc[start: end]

start_date = adjClose.iloc[1920].name
end_date = NPTTM.iloc[-2].name

# main function
def Backtesting(start, end, N, calculator):
    dates = list(adjClose.loc[start:end, :].index)
    myindex = adjClose.loc[start:end, :].index
    result = pd.DataFrame(np.ones(N * len(dates)).reshape(len(dates), N), index=myindex,
                          columns=['group %i' % j for j in range(1, N + 1)])

    for i in range(0, len(dates)):
        current_date_num = dates[i]
        current_date = dateDict[current_date_num]

        if i == 0:
            # initial day, determine the stocks to hold
            stock_pool = get_tradable_list(status.loc[current_date_num, :])

            for stock in stock_pool:
                if not (stock in IPO.index):
                    stock_pool.remove(stock)
                elif (current_date - IPO.loc[stock][0]).days < 365:
                    stock_pool.remove(stock)
                elif high.loc[current_date_num, stock] == low.loc[current_date_num, stock]:
                    stock_pool.remove(stock)

            factor = calculator(current_date_num)
            factor = factor.loc[factor.index.intersection(stock_pool)]

            stock_in_pos = {}
            stock_value = {}
            for i in range(0, N):
                stock_in_pos[i] = list(get_group(factor, N, i).index)
                n_stock = len(stock_in_pos[i])
                stock_value[i] = pd.Series(np.ones(n_stock) / n_stock, index=stock_in_pos[i])
                print('number of stock in group %i' % (i + 1), n_stock)

            past_date_num = current_date_num
            past_date = current_date

        else:
            print('current date', current_date)

            for i in range(0, N):
                stock_value[i] = stock_value[i] * (close_pct.loc[current_date_num, stock_in_pos[i]].clip(0.9, 1.1))
                result.loc[current_date_num, 'EP group %i' % (i + 1)] = stock_value[i].sum()

            if (current_date.month != past_date.month) or (
                    i == (len(dates) - 1)):  # rebalance at first trade day of each month
                print('Begin adjusting position')

                # determine the new stocks to hold
                stock_pool = get_tradable_list(status.loc[current_date_num, :])

                for stock in stock_pool:
                    if not (stock in IPO.index):
                        stock_pool.remove(stock)
                    elif (current_date - IPO.loc[stock][0]).days < 365:
                        stock_pool.remove(stock)
                    elif high.loc[current_date_num, stock] == low.loc[current_date_num, stock]:
                        stock_pool.remove(stock)

                factor = calculator(past_date_num)
                factor = factor.loc[factor.index.intersection(stock_pool)]

                old_pos = stock_value

                stock_in_pos = {}
                stock_value = {}
                # dedcut the transaction fee while recording the new stock list
                for i in range(0, N):
                    stock_in_pos[i] = list(get_group(factor, N, i).index)
                    n_stock = len(stock_in_pos[i])
                    w = result.loc[current_date_num, 'EP group %i' % (i + 1)]
                    stock_value[i] = pd.Series([w / n_stock] * n_stock, index=stock_in_pos[i])
                    turn = pd.concat([old_pos[i], stock_value[i]], axis=1).fillna(0)
                    turn = turn.apply(lambda x: x / x.sum())
                    t = sum(abs(turn.iloc[:, 0] - turn.iloc[:, -1]))
                    stock_value[i] = stock_value[i] * (1 - 0.003 * t)
                    print('number of stock in group %i' % (i + 1), n_stock)

            past_date_num = current_date_num
            past_date = current_date

    return result

result = Backtesting(start_date,end_date,5,calculate_ep)

