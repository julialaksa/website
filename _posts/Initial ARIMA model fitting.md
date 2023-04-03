
---
layout: post
title: Random post!
---



## 1. Initial dataset creation

We aim to predict the change in price of a stock based on changes in various fundamental characteristics of the stock over time. We will start by building an ARIMA model on one stock (we pick AAPL) to get an initial understanding of:

- What is an appropriate target variable
- What explanatory variables might be relevant / irrelevant
- What are the shortcomings in the ARIMA model that might be addressed by LSTM model (for example)
- What are additional features we might be able to include in an ML model through feature engineering

We will gather the following data:

- Daily prices of AAPL
- Quarterly financial statements data of AAPL
- Quarterly fundamental ratios of AAPL

Quarterly financial statements data of AAPL is available for the period from Q1-2008. We will therefore use this as our sample timeframe.

### 1.1 Data download and cleaning

#### 1.1.1 Daily prices data download


```python
import pandas as pd
import numpy as np
import yfinance as yf # daily historical prices are available on Yahoo finance

import warnings
warnings.filterwarnings("ignore")
```


```python
# start date: 01-01-2008, i.e. start Q1 2008
# end date: 2023-01-01, i.e. end Q4 2022
# picked to match the quarterly statements data

AAPL_daily_prices = yf.Ticker('AAPL').history(start="2008-01-01",end="2023-01-01",interval="1d").reset_index()
```


```python
AAPL_daily_prices # 2008-01-02 to 2022-12-30, 3777 rows × 8 columns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-01-02 00:00:00-05:00</td>
      <td>6.057226</td>
      <td>6.087319</td>
      <td>5.852958</td>
      <td>5.922566</td>
      <td>1079178800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-01-03 00:00:00-05:00</td>
      <td>5.939896</td>
      <td>6.000082</td>
      <td>5.857215</td>
      <td>5.925305</td>
      <td>842066400</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-01-04 00:00:00-05:00</td>
      <td>5.819524</td>
      <td>5.866639</td>
      <td>5.437736</td>
      <td>5.472997</td>
      <td>1455832000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-01-07 00:00:00-05:00</td>
      <td>5.509470</td>
      <td>5.580904</td>
      <td>5.174495</td>
      <td>5.399737</td>
      <td>2072193200</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-01-08 00:00:00-05:00</td>
      <td>5.475731</td>
      <td>5.546253</td>
      <td>5.191823</td>
      <td>5.205502</td>
      <td>1523816000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>2022-12-23 00:00:00-05:00</td>
      <td>130.720412</td>
      <td>132.218125</td>
      <td>129.442364</td>
      <td>131.658981</td>
      <td>63814900</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3773</th>
      <td>2022-12-27 00:00:00-05:00</td>
      <td>131.179720</td>
      <td>131.209673</td>
      <td>128.523771</td>
      <td>129.831772</td>
      <td>69007800</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>2022-12-28 00:00:00-05:00</td>
      <td>129.472318</td>
      <td>130.830245</td>
      <td>125.678116</td>
      <td>125.847855</td>
      <td>85438400</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>2022-12-29 00:00:00-05:00</td>
      <td>127.794881</td>
      <td>130.281083</td>
      <td>127.535283</td>
      <td>129.412415</td>
      <td>75703700</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>2022-12-30 00:00:00-05:00</td>
      <td>128.214246</td>
      <td>129.751892</td>
      <td>127.235737</td>
      <td>129.731918</td>
      <td>77034200</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3777 rows × 8 columns</p>
</div>




```python
AAPL_daily_prices.info() # Date is already in date format, no null values
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3777 entries, 0 to 3776
    Data columns (total 8 columns):
     #   Column        Non-Null Count  Dtype                           
    ---  ------        --------------  -----                           
     0   Date          3777 non-null   datetime64[ns, America/New_York]
     1   Open          3777 non-null   float64                         
     2   High          3777 non-null   float64                         
     3   Low           3777 non-null   float64                         
     4   Close         3777 non-null   float64                         
     5   Volume        3777 non-null   int64                           
     6   Dividends     3777 non-null   float64                         
     7   Stock Splits  3777 non-null   float64                         
    dtypes: datetime64[ns, America/New_York](1), float64(6), int64(1)
    memory usage: 236.2 KB
    


```python
# We will be matching the financial statements data on year / quarter. Creating year / quarter columns, in format 'Q1', 2017.
AAPL_daily_prices['Fiscal Year'] = pd.DatetimeIndex(AAPL_daily_prices['Date']).year
AAPL_daily_prices['Fiscal Period'] = pd.DatetimeIndex(AAPL_daily_prices['Date']).quarter
AAPL_daily_prices['Fiscal Period'] = 'Q' + AAPL_daily_prices['Fiscal Period'].astype(str)
```


```python
AAPL_daily_prices['Fiscal Year'] # int
AAPL_daily_prices['Fiscal Period'] # object
```




    0       Q1
    1       Q1
    2       Q1
    3       Q1
    4       Q1
            ..
    3772    Q4
    3773    Q4
    3774    Q4
    3775    Q4
    3776    Q4
    Name: Fiscal Period, Length: 3777, dtype: object




```python
AAPL_daily_prices.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
      <th>Fiscal Year</th>
      <th>Fiscal Period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>2008-05-23 00:00:00-04:00</td>
      <td>5.494880</td>
      <td>5.531965</td>
      <td>5.404601</td>
      <td>5.507039</td>
      <td>906917200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008</td>
      <td>Q2</td>
    </tr>
    <tr>
      <th>2307</th>
      <td>2017-03-02 00:00:00-05:00</td>
      <td>32.846625</td>
      <td>32.912318</td>
      <td>32.555696</td>
      <td>32.602623</td>
      <td>104844000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>915</th>
      <td>2011-08-18 00:00:00-04:00</td>
      <td>11.272455</td>
      <td>11.327474</td>
      <td>10.984594</td>
      <td>11.126852</td>
      <td>851435200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>Q3</td>
    </tr>
    <tr>
      <th>3570</th>
      <td>2022-03-08 00:00:00-05:00</td>
      <td>157.863715</td>
      <td>161.899267</td>
      <td>154.861895</td>
      <td>156.492020</td>
      <td>131148300</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>2015-01-06 00:00:00-05:00</td>
      <td>23.938801</td>
      <td>24.138777</td>
      <td>23.509636</td>
      <td>23.875887</td>
      <td>263188400</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2015</td>
      <td>Q1</td>
    </tr>
    <tr>
      <th>624</th>
      <td>2010-06-24 00:00:00-04:00</td>
      <td>8.237609</td>
      <td>8.304483</td>
      <td>8.149458</td>
      <td>8.176816</td>
      <td>714277200</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2010</td>
      <td>Q2</td>
    </tr>
    <tr>
      <th>883</th>
      <td>2011-07-05 00:00:00-04:00</td>
      <td>10.426200</td>
      <td>10.633813</td>
      <td>10.411002</td>
      <td>10.621654</td>
      <td>355054000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>Q3</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>2020-12-11 00:00:00-05:00</td>
      <td>120.793304</td>
      <td>121.118894</td>
      <td>118.938439</td>
      <td>120.773575</td>
      <td>86939800</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2020</td>
      <td>Q4</td>
    </tr>
    <tr>
      <th>991</th>
      <td>2011-12-06 00:00:00-05:00</td>
      <td>11.931160</td>
      <td>11.995603</td>
      <td>11.836018</td>
      <td>11.883740</td>
      <td>283598000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>Q4</td>
    </tr>
    <tr>
      <th>2990</th>
      <td>2019-11-15 00:00:00-05:00</td>
      <td>64.481887</td>
      <td>64.995435</td>
      <td>64.318046</td>
      <td>64.990547</td>
      <td>100206400</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2019</td>
      <td>Q4</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.1.2 Financial statements data download

We will be getting the following quarterly data:
1. Balance sheet
2. Income statement
3. Cash flow statement
4. Financial ratios

I have come across an error trying to obtain the data on `yfinance` (seems to be a recent `yfinance` encryption issue / bug). I have included the code below to re-use if it is gets resolved. In the meantime, we will be using data from SimFin.


```python
# CREATE A TICKER INSTANCE PASSING TESLA AS THE TARGET COMPANY
# tsla = yf.Ticker('TSLA')

# # CALL THE MULTIPLE FUNCTIONS AVAILABLE AND STORE THEM IN VARIABLES.
# actions = tsla.get_actions()
# tsla.get_balance_sheet()
# calendar = tsla.get_calendar()
# cf = tsla.get_cashflow()
# info = tsla.get_info()
# inst_holders = tsla.get_institutional_holders()
# news = tsla.get_news()
# recommendations = tsla.get_recommendations()
# sustainability = tsla.get_sustainability()

# Error: yfinance failed to decrypt Yahoo data response
```

Data source: https://app.simfin.com/fundamentals/view/111052

I have downloaded CSV files. After initial ARIMA is complete, I will re-visit this and try to obtain data via scraping.


```python
AAPL_bs = pd.read_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_bs_exp.csv",index_col=0).T
AAPL_cf = pd.read_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_cf_exp.csv",index_col=0).T
AAPL_pnl = pd.read_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_pnl_exp.csv",index_col=0).T
AAPL_ratios = pd.read_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_ratios_exp.csv",index_col=0).T
```


```python
# Units in AAPL_bs, AAPL_cf, AAPL_pnl are in mUSD.
```

#### 1.1.3 Financial statements data cleaning


```python
# Due to the format of the data, we have some columns that only contain NAN values (e.g. no Treasury stock present / no data available)
AAPL_bs_1 = AAPL_bs.dropna(axis=1, how='all')
AAPL_pnl_1 = AAPL_pnl.dropna(axis=1, how='all')
AAPL_cf_1 = AAPL_cf.dropna(axis=1, how='all')
AAPL_ratios_1 = AAPL_ratios.dropna(axis=1, how='all')
```


```python
# The date column is imported as nan
AAPL_bs_1 = AAPL_bs_1.rename(columns={np.nan: 'Date'})
AAPL_pnl_1 = AAPL_pnl_1.rename(columns={np.nan: 'Date'})
AAPL_cf_1 = AAPL_cf_1.rename(columns={np.nan: 'Date'})
AAPL_ratios_1 = AAPL_ratios_1.rename(columns={np.nan: 'Date'})
```

NAN values

NANs are plausibly zeros due to the format of the financial statements. 


```python
nullseries = AAPL_bs_1.isna().sum() # Zeros in some periods.
nullseries[nullseries>0]

```




    Deferred Tax Assets                     30
    Long Term Investments & Receivables      1
    Intangible Assets                       21
    Goodwill                                21
    Other Payables & Accruals               19
    Short Term Debt                         24
    Other Short Term Liabilities             1
    Deferred Revenue                         1
    Miscellaneous Short Term Liabilities    41
    Long Term Debt                          20
    Deferred Revenue                        19
    dtype: int64




```python
nullseries = AAPL_pnl_1.isna().sum()
nullseries[nullseries>0]
```




    Series([], dtype: int64)




```python
nullseries = AAPL_cf_1.isna().sum()
nullseries[nullseries>0]
```




    Purchase of Fixed Assets                         58
    Decrease in Long Term Investment                  6
    Increase in Long Term Investment                  6
    Net Cash From Acquisitions & Divestitures         9
    Dividends Paid                                   13
    Cash From (Repayment of) Debt                    18
    Cash From (Repayment of) Short Term Debt, net    26
    Cash From (Repayment of) Long Term Debt, net     27
    Repayments of Long Term Debt                     43
    Cash From Long Term Debt                         31
    Increase in Capital Stock                        12
    Decrease in Capital Stock                        14
    dtype: int64




```python
nullseries = AAPL_ratios_1.isna().sum() 
nullseries[nullseries>0]
# Dividends Paid are zero in 13 periods, we expect those ratios to be zeros.
```




    Dividends Per Share      13
    Dividend Payout Ratio    13
    dtype: int64




```python
AAPL_ratios_1['Dividends Per Share'][AAPL_ratios_1['Dividends Per Share'].isna()]
```




    Q1 2009    NaN
    Q2 2008    NaN
    Q3 2008    NaN
    Q4 2008    NaN
    Q1 2010    NaN
    Q2 2009    NaN
    Q3 2009    NaN
    Q4 2009    NaN
    Q1 2011    NaN
    Q2 2010    NaN
    Q3 2010    NaN
    Q2 2011    NaN
    Q3 2011    NaN
    Name: Dividends Per Share, dtype: object




```python
AAPL_cf_1['Dividends Paid'][AAPL_cf_1['Dividends Paid'].isna()] # matching up with missing data in AAPL_ratios_1
```




    Q1 2009    NaN
    Q2 2008    NaN
    Q3 2008    NaN
    Q4 2008    NaN
    Q1 2010    NaN
    Q2 2009    NaN
    Q3 2009    NaN
    Q4 2009    NaN
    Q1 2011    NaN
    Q2 2010    NaN
    Q3 2010    NaN
    Q2 2011    NaN
    Q3 2011    NaN
    Name: Dividends Paid, dtype: object




```python
AAPL_cf_1 = AAPL_cf_1.fillna(0)
AAPL_bs_1 = AAPL_bs_1.fillna(0)
AAPL_ratios_1 = AAPL_ratios_1.fillna(0)
```

Merging dataset


```python
# Prior to merging, ensure that the dates are in the correct order.
AAPL_bs_1.sort_values(by='Date', inplace = True)
AAPL_cf_1.sort_values(by='Date', inplace = True)
AAPL_pnl_1.sort_values(by='Date', inplace = True)
AAPL_ratios_1.sort_values(by='Date', inplace = True)
```


```python
AAPL_cf_1.shape # (60, 29)
AAPL_bs_1.shape # (60, 39)
AAPL_pnl_1.shape # (60, 17)
AAPL_ratios_1.shape # (60, 25)
```




    (60, 25)




```python
cols_to_use = AAPL_cf_1.columns.difference(AAPL_bs_1.columns) # only add cols unique to AAPL ratios
AAPL_cf_bs = AAPL_bs_1.join(AAPL_cf_1[cols_to_use])

AAPL_cf_bs.shape # (60, 67) - excl. date column
```




    (60, 67)




```python
cols_to_use = AAPL_pnl_1.columns.difference(AAPL_cf_bs.columns) # only add cols unique to AAPL ratios
AAPL_cf_bs_pnl = AAPL_cf_bs.join(AAPL_pnl_1[cols_to_use])
AAPL_cf_bs_pnl.shape # (60, 83) - excl. date column
```




    (60, 83)




```python
cols_to_use = AAPL_ratios_1.columns.difference(AAPL_cf_bs_pnl.columns) # only add cols unique to AAPL ratios
AAPL_cf_bs_pnl_ratios = AAPL_cf_bs_pnl.join(AAPL_ratios_1[cols_to_use])
AAPL_cf_bs_pnl_ratios.shape # (60, 107) - excl. date column
```




    (60, 107)




```python
AAPL_financials = AAPL_cf_bs_pnl_ratios
```

Duplicated columns


```python
AAPL_financials.shape # (60, 107)
```




    (60, 107)




```python
AAPL_financials.duplicated().sum() # No quarters had the exact same financials data, this is reassuring
```




    0




```python
dup_cols = AAPL_financials.T.duplicated()
dup_cols[dup_cols == 1]

# We have 12 duplicate columns because the financial statements include totals / restatements for each category. We will remove these columns.
```




    Accounts Receivable, Net                       True
    Total Equity                                   True
    Total Liabilities & Equity                     True
    Change in Fixed Assets & Intangibles           True
    Net Cash Before FX                             True
    Net Changes in Cash                            True
    Income (Loss) Including Minority Interest      True
    Income (Loss) from Continuing Operations       True
    Net Income                                     True
    Net Income Available to Common Shareholders    True
    Other Non-Operating Income (Loss)              True
    Pretax Income (Loss), Adjusted                 True
    dtype: bool




```python
AAPL_financials = AAPL_financials.drop(columns=dup_cols[dup_cols == 1].index)
```

Creating year / quarter tag


```python
# Currently the quarters notation is forward-looking, e.g. 31-12-2022 is Q1 2023
# To match the daily pricing data, we want 31-12-2022 to be represented as Q4 2022

AAPL_financials['Fiscal Year'] = pd.DatetimeIndex(AAPL_financials['Date']).year
AAPL_financials['Fiscal Period'] = pd.DatetimeIndex(AAPL_financials['Date']).quarter
AAPL_financials['Fiscal Period'] = 'Q' + AAPL_financials['Fiscal Period'].astype(str)

AAPL_financials.shape # (60, 97)

```




    (60, 97)



Left join daily price data and financial statements data on year and quarter


```python
AAPL_data = AAPL_daily_prices.merge(AAPL_financials, on=['Fiscal Year','Fiscal Period'], how='left', indicator=True) # 3777 rows × 106 columns
```


```python
AAPL_data[['Date_x','Date_y']] # as expected, we have the same quarterly financial statements entry for all daily price data rows in a given quarter
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_x</th>
      <th>Date_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-01-02 00:00:00-05:00</td>
      <td>2008-03-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-01-03 00:00:00-05:00</td>
      <td>2008-03-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-01-04 00:00:00-05:00</td>
      <td>2008-03-31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-01-07 00:00:00-05:00</td>
      <td>2008-03-31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-01-08 00:00:00-05:00</td>
      <td>2008-03-31</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>2022-12-23 00:00:00-05:00</td>
      <td>2022-12-31</td>
    </tr>
    <tr>
      <th>3773</th>
      <td>2022-12-27 00:00:00-05:00</td>
      <td>2022-12-31</td>
    </tr>
    <tr>
      <th>3774</th>
      <td>2022-12-28 00:00:00-05:00</td>
      <td>2022-12-31</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>2022-12-29 00:00:00-05:00</td>
      <td>2022-12-31</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>2022-12-30 00:00:00-05:00</td>
      <td>2022-12-31</td>
    </tr>
  </tbody>
</table>
<p>3777 rows × 2 columns</p>
</div>




```python
AAPL_data = AAPL_data.rename(columns={'Date_x':'Date_day','Date_y':'Date_quarter'})
```

Missing dates


```python
our_list = AAPL_data['Date_day'].values # dates we have
full_list = pd.bdate_range(start='2008-01-02',end='2022-12-30',freq='B') # all business dates in the time period
our_list = [pd.to_datetime(x).strftime('%Y-%m-%d') for x in our_list] # our dates as a list
full_list = [x.strftime('%Y-%m-%d') for x in full_list] # full list as a list

diff = list(set(full_list)-set(our_list))
diff.sort()
diff

# Looks like list of US holidays: e.g. '2008-01-21' is Martin Luther King Jr. Day; '2008-02-18' is President's day. '2008-03-21' is Good Friday, '2008-05-26' is spring bank holiday, etc.
```




    ['2008-01-21',
     '2008-02-18',
     '2008-03-21',
     '2008-05-26',
     '2008-07-04',
     '2008-09-01',
     '2008-11-27',
     '2008-12-25',
     '2009-01-01',
     '2009-01-19',
     '2009-02-16',
     '2009-04-10',
     '2009-05-25',
     '2009-07-03',
     '2009-09-07',
     '2009-11-26',
     '2009-12-25',
     '2010-01-01',
     '2010-01-18',
     '2010-02-15',
     '2010-04-02',
     '2010-05-31',
     '2010-07-05',
     '2010-09-06',
     '2010-11-25',
     '2010-12-24',
     '2011-01-17',
     '2011-02-21',
     '2011-04-22',
     '2011-05-30',
     '2011-07-04',
     '2011-09-05',
     '2011-11-24',
     '2011-12-26',
     '2012-01-02',
     '2012-01-16',
     '2012-02-20',
     '2012-04-06',
     '2012-05-28',
     '2012-07-04',
     '2012-09-03',
     '2012-10-29',
     '2012-10-30',
     '2012-11-22',
     '2012-12-25',
     '2013-01-01',
     '2013-01-21',
     '2013-02-18',
     '2013-03-29',
     '2013-05-27',
     '2013-07-04',
     '2013-09-02',
     '2013-11-28',
     '2013-12-25',
     '2014-01-01',
     '2014-01-20',
     '2014-02-17',
     '2014-04-18',
     '2014-05-26',
     '2014-07-04',
     '2014-09-01',
     '2014-11-27',
     '2014-12-25',
     '2015-01-01',
     '2015-01-19',
     '2015-02-16',
     '2015-04-03',
     '2015-05-25',
     '2015-07-03',
     '2015-09-07',
     '2015-11-26',
     '2015-12-25',
     '2016-01-01',
     '2016-01-18',
     '2016-02-15',
     '2016-03-25',
     '2016-05-30',
     '2016-07-04',
     '2016-09-05',
     '2016-11-24',
     '2016-12-26',
     '2017-01-02',
     '2017-01-16',
     '2017-02-20',
     '2017-04-14',
     '2017-05-29',
     '2017-07-04',
     '2017-09-04',
     '2017-11-23',
     '2017-12-25',
     '2018-01-01',
     '2018-01-15',
     '2018-02-19',
     '2018-03-30',
     '2018-05-28',
     '2018-07-04',
     '2018-09-03',
     '2018-11-22',
     '2018-12-05',
     '2018-12-25',
     '2019-01-01',
     '2019-01-21',
     '2019-02-18',
     '2019-04-19',
     '2019-05-27',
     '2019-07-04',
     '2019-09-02',
     '2019-11-28',
     '2019-12-25',
     '2020-01-01',
     '2020-01-20',
     '2020-02-17',
     '2020-04-10',
     '2020-05-25',
     '2020-07-03',
     '2020-09-07',
     '2020-11-26',
     '2020-12-25',
     '2021-01-01',
     '2021-01-18',
     '2021-02-15',
     '2021-04-02',
     '2021-05-31',
     '2021-07-05',
     '2021-09-06',
     '2021-11-25',
     '2021-12-24',
     '2022-01-17',
     '2022-02-21',
     '2022-04-15',
     '2022-05-30',
     '2022-06-20',
     '2022-07-04',
     '2022-09-05',
     '2022-11-24',
     '2022-12-26']




```python
len(diff) # 136 holiday dates in the period
```




    136




```python
# Create empty CSV to save down the initial dataframe

df = pd.DataFrame(list())
df.to_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_dataset_expanded.csv")
AAPL_data.to_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_dataset_expanded.csv")
```


```python
# Quick sense check - no duplicated rows or columns
AAPL_data.duplicated().sum() 
AAPL_data.T.duplicated().sum()
```




    0




```python
AAPL_data.isna().sum().sum() # no null values
```




    0



We are good to go.

### 2.1 EDA


```python
AAPL_data = pd.read_csv("C:\\Users\\julia\\OneDrive\\Documents\\Brainstation\\Capstone\\AAPL_dataset_expanded.csv")
```


```python
AAPL_data.drop(columns=['_merge'],inplace=True)
AAPL_data.drop(columns=['Unnamed: 0'],inplace=True)
```


```python
AAPL_data['Date_short'] = our_list
```


```python
AAPL_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3777 entries, 0 to 3776
    Columns: 106 entries, Date_day to Date_short
    dtypes: float64(26), int64(76), object(4)
    memory usage: 3.1+ MB
    


```python
AAPL_data['Date_day'] == pd.to_datetime(AAPL_data['Date_day'])
AAPL_data = AAPL_data.set_index('Date_day')
```


```python
AAPL_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3777 entries, 2008-01-02 00:00:00-05:00 to 2022-12-30 00:00:00-05:00
    Columns: 105 entries, Open to Date_short
    dtypes: float64(26), int64(76), object(3)
    memory usage: 3.1+ MB
    

Importing plotting libraries


```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly.graph_objects as go
```


```python
plt.plot(AAPL_data['Date_short'],AAPL_data['Close'])
plt.xlabel('Date')
plt.ylabel('AAPL price, close')
plt.xticks(AAPL_data['Date_short'][::365])
plt.xticks(rotation=45)
# plt.plot(AAPL_data['Date_day'],AAPL_data['Close'])
# plt.plot(AAPL_data['Date_day'],AAPL_data['High'])
# plt.plot(AAPL_data['Date_day'],AAPL_data['Low'])
plt.show()
```


    
![png](output_63_0.png)
    



```python
fig = go.Figure()

fig.add_trace(go.Scatter(x=AAPL_data['Date_short'],y=AAPL_data['Close']))

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.update_layout(
    title_text="AAPL Close Price, $"
)
fig.show()
```




```python
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Total Liabilities',x=AAPL_data['Date_quarter'].unique(), y=abs(AAPL_data['Dividends Paid']).unique())
])

fig.update_layout(
    title_text="Total Dividends Paid out, mUSD"
)

fig.show()
```




```python
fig = go.Figure()

fig.add_trace(go.Scatter(x=AAPL_data['Date_quarter'],y=AAPL_data['Cash, Cash Equivalents & Short Term Investments']))

fig.add_trace(go.Scatter(x=AAPL_data['Date_quarter'],y=AAPL_data['Cash & Cash Equivalents']))

fig.update_layout(
    title_text="Cash and Cash equivalents, mUSD"
)
fig.show()
```




```python
fig = go.Figure()

fig.add_trace(go.Scatter(x=AAPL_data['Date_quarter'],y=AAPL_data['Retained Earnings']))

fig.update_layout(
    title_text="Retained Earnings, mUSD"
)
fig.show()
```




```python
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Total Liabilities',x=AAPL_data['Date_quarter'].unique(), y=AAPL_data['Total Liabilities'].unique()),
    go.Bar(name='Total Equity',x=AAPL_data['Date_quarter'].unique(), y=AAPL_data['Equity Before Minority Interest'].unique())
])

fig.update_layout(
    title_text="Total Assets Breakdown, mUSD"
)
# Change the bar mode
fig.update_layout(barmode='stack')
fig.show()
```




```python
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='ROE',x=AAPL_data['Date_quarter'].unique(), y=AAPL_data['Return on Equity'].unique()),
    go.Bar(name='EPS',x=AAPL_data['Date_quarter'].unique(), y=AAPL_data['Earnings Per Share, Basic'].unique()),
    go.Bar(name='Net profit margin',x=AAPL_data['Date_quarter'].unique(), y=AAPL_data['Net Profit Margin'].unique())
])

fig.update_layout(
    title_text="Key financial ratios"
)

fig.show()
```



### 3.1 Creating initial ARIMA model


```python
# We will try to predict: daily return, weekly return, monthly return

AAPL_data['1d_ret'] = (AAPL_data['Close']-AAPL_data['Close'].shift(1))/AAPL_data['Close'].shift(1) # Monday to Tuesday returns
AAPL_data['5d_ret'] = (AAPL_data['Close']-AAPL_data['Close'].shift(5))/AAPL_data['Close'].shift(5) # 1 week returns
AAPL_data['20d_ret'] = (AAPL_data['Close']-AAPL_data['Close'].shift(20))/AAPL_data['Close'].shift(20) # 1 month returns
```


```python
fig = go.Figure()

fig.add_trace(go.Scatter(name='Daily return',x=AAPL_data['Date_short'],y=AAPL_data['1d_ret']))
fig.add_trace(go.Scatter(name='Moving average, 20-day',x=AAPL_data['Date_short'],y=AAPL_data['1d_ret'].rolling(20).mean()))

fig.update_layout(
    title_text="AAPL 1-day returns, %"
)
fig.show()

# Looks uninformative
```




```python
fig = go.Figure()

fig.add_trace(go.Scatter(name="5-day return",x=AAPL_data['Date_short'],y=AAPL_data['5d_ret']))
fig.add_trace(go.Scatter(name='Moving average, 20-day',x=AAPL_data['Date_short'],y=AAPL_data['5d_ret'].rolling(20).mean()))


fig.update_layout(
    title_text="AAPL 5-day returns, %"
)
fig.show()

# Looks a littive more cyclical
```




```python
fig = go.Figure()

fig.add_trace(go.Scatter(x=AAPL_data['Date_short'],y=AAPL_data['20d_ret']))
fig.add_trace(go.Scatter(name='Moving average, 20-day',x=AAPL_data['Date_short'],y=AAPL_data['20d_ret'].rolling(20).mean()))

fig.update_layout(
    title_text="AAPL 20-day returns, %"
)
fig.show()

# Moving average over 20 days follows the day-on-day 20-day returns more closely.
```



We will try creating ARIMA on all 3 return time series.


```python

```


```python
AAPL_data = AAPL_data.set_index("Date_day")
```


```python
AAPL_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3777 entries, 2008-01-02 00:00:00-05:00 to 2022-12-30 00:00:00-05:00
    Columns: 108 entries, Open to 20d_ret
    dtypes: float64(29), int64(76), object(3)
    memory usage: 3.1+ MB
    


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(15, 5))
plot_acf(AAPL_data['1d_ret'].dropna(), lags=20, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('1-day return AC')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(AAPL_data['1d_ret'].dropna(), lags=20, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('1-day return PAC')
plt.show()

# 1-day returns are not correlated with their past values - i.e. today's 1-day return has no clear relationship with yesterday's return, or any other daily return in the immediate past
```


    
![png](output_79_0.png)
    



    
![png](output_79_1.png)
    



```python
plt.figure(figsize=(15, 5))
plot_acf(AAPL_data['5d_ret'].dropna(), lags=20, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('5-day return AC')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(AAPL_data['5d_ret'].dropna(), lags=20, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('5-day return PAC')
plt.show()

# Weekly returns show stronger AC and PAC. Today's weekly return is correlated with yesterday's weekly return, as well as weekly return 5-days (one trading week) ago.
```


    
![png](output_80_0.png)
    



    
![png](output_80_1.png)
    



```python
plt.figure(figsize=(15, 5))
plot_acf(AAPL_data['20d_ret'].dropna(), lags=150, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('20-day return AC')
plt.show()

plt.figure(figsize=(15, 5))
plot_pacf(AAPL_data['20d_ret'].dropna(), lags=150, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('20-day return PAC')
plt.show()

# Monthly return is correlated with yesterday's monthly return, and with 1-month return one one / two / three months ago.
```


    
![png](output_81_0.png)
    



    
![png](output_81_1.png)
    


5-day ARIMA


```python
AAPL_data_ARIMA = AAPL_data['5d_ret'].dropna()
AAPL_data_ARIMA
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    c:\Users\julia\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3360             try:
    -> 3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    

    c:\Users\julia\anaconda3\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    c:\Users\julia\anaconda3\lib\site-packages\pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: '5d_ret'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_27756\4051727636.py in <module>
    ----> 1 AAPL_data_ARIMA = AAPL_data['5d_ret'].dropna()
          2 AAPL_data_ARIMA
    

    c:\Users\julia\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       3456             if self.columns.nlevels > 1:
       3457                 return self._getitem_multilevel(key)
    -> 3458             indexer = self.columns.get_loc(key)
       3459             if is_integer(indexer):
       3460                 indexer = [indexer]
    

    c:\Users\julia\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    -> 3363                 raise KeyError(key) from err
       3364 
       3365         if is_scalar(key) and isna(key) and not self.hasnans:
    

    KeyError: '5d_ret'



```python
train = AAPL_data_ARIMA.loc[AAPL_data_ARIMA.index < "2022-01-01"]
test = AAPL_data_ARIMA.loc[AAPL_data_ARIMA.index >= "2022-01-01"]
```


```python
train.index
```




    Index(['2008-01-09 00:00:00-05:00', '2008-01-10 00:00:00-05:00',
           '2008-01-11 00:00:00-05:00', '2008-01-14 00:00:00-05:00',
           '2008-01-15 00:00:00-05:00', '2008-01-16 00:00:00-05:00',
           '2008-01-17 00:00:00-05:00', '2008-01-18 00:00:00-05:00',
           '2008-01-22 00:00:00-05:00', '2008-01-23 00:00:00-05:00',
           ...
           '2021-12-17 00:00:00-05:00', '2021-12-20 00:00:00-05:00',
           '2021-12-21 00:00:00-05:00', '2021-12-22 00:00:00-05:00',
           '2021-12-23 00:00:00-05:00', '2021-12-27 00:00:00-05:00',
           '2021-12-28 00:00:00-05:00', '2021-12-29 00:00:00-05:00',
           '2021-12-30 00:00:00-05:00', '2021-12-31 00:00:00-05:00'],
          dtype='object', name='Date_day', length=3521)




```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

p_param = 20

model = SARIMAX(train, order=(p_param, 0, 0), trend="c")
model_fit = model.fit(disp=0)

model_fit.summary()
```

    c:\Users\julia\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning:
    
    An unsupported index was provided and will be ignored when e.g. forecasting.
    
    c:\Users\julia\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:471: ValueWarning:
    
    An unsupported index was provided and will be ignored when e.g. forecasting.
    
    c:\Users\julia\anaconda3\lib\site-packages\statsmodels\base\model.py:604: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>5d_ret</td>       <th>  No. Observations:  </th>    <td>3521</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(20, 0, 0)</td> <th>  Log Likelihood     </th>  <td>8501.987</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 24 Mar 2023</td>  <th>  AIC                </th> <td>-16959.973</td>
</tr>
<tr>
  <th>Time:</th>                <td>12:14:24</td>      <th>  BIC                </th> <td>-16824.310</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>         <th>  HQIC               </th> <td>-16911.571</td>
</tr>
<tr>
  <th></th>                      <td> - 3521</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>        <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.0013</td> <td>    0.000</td> <td>    3.376</td> <td> 0.001</td> <td>    0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>    0.9062</td> <td>    0.012</td> <td>   75.730</td> <td> 0.000</td> <td>    0.883</td> <td>    0.930</td>
</tr>
<tr>
  <th>ar.L2</th>     <td>    0.0313</td> <td>    0.015</td> <td>    2.099</td> <td> 0.036</td> <td>    0.002</td> <td>    0.060</td>
</tr>
<tr>
  <th>ar.L3</th>     <td>    0.0036</td> <td>    0.016</td> <td>    0.225</td> <td> 0.822</td> <td>   -0.028</td> <td>    0.035</td>
</tr>
<tr>
  <th>ar.L4</th>     <td>    0.0361</td> <td>    0.015</td> <td>    2.418</td> <td> 0.016</td> <td>    0.007</td> <td>    0.065</td>
</tr>
<tr>
  <th>ar.L5</th>     <td>   -0.7618</td> <td>    0.016</td> <td>  -48.897</td> <td> 0.000</td> <td>   -0.792</td> <td>   -0.731</td>
</tr>
<tr>
  <th>ar.L6</th>     <td>    0.6057</td> <td>    0.018</td> <td>   34.040</td> <td> 0.000</td> <td>    0.571</td> <td>    0.641</td>
</tr>
<tr>
  <th>ar.L7</th>     <td>    0.1146</td> <td>    0.018</td> <td>    6.419</td> <td> 0.000</td> <td>    0.080</td> <td>    0.150</td>
</tr>
<tr>
  <th>ar.L8</th>     <td>   -0.0673</td> <td>    0.019</td> <td>   -3.584</td> <td> 0.000</td> <td>   -0.104</td> <td>   -0.031</td>
</tr>
<tr>
  <th>ar.L9</th>     <td>    0.0796</td> <td>    0.018</td> <td>    4.336</td> <td> 0.000</td> <td>    0.044</td> <td>    0.116</td>
</tr>
<tr>
  <th>ar.L10</th>    <td>   -0.5311</td> <td>    0.019</td> <td>  -27.448</td> <td> 0.000</td> <td>   -0.569</td> <td>   -0.493</td>
</tr>
<tr>
  <th>ar.L11</th>    <td>    0.3496</td> <td>    0.019</td> <td>   18.389</td> <td> 0.000</td> <td>    0.312</td> <td>    0.387</td>
</tr>
<tr>
  <th>ar.L12</th>    <td>    0.1596</td> <td>    0.019</td> <td>    8.567</td> <td> 0.000</td> <td>    0.123</td> <td>    0.196</td>
</tr>
<tr>
  <th>ar.L13</th>    <td>   -0.0835</td> <td>    0.019</td> <td>   -4.313</td> <td> 0.000</td> <td>   -0.121</td> <td>   -0.046</td>
</tr>
<tr>
  <th>ar.L14</th>    <td>    0.0778</td> <td>    0.020</td> <td>    3.877</td> <td> 0.000</td> <td>    0.038</td> <td>    0.117</td>
</tr>
<tr>
  <th>ar.L15</th>    <td>   -0.3432</td> <td>    0.020</td> <td>  -17.470</td> <td> 0.000</td> <td>   -0.382</td> <td>   -0.305</td>
</tr>
<tr>
  <th>ar.L16</th>    <td>    0.2227</td> <td>    0.015</td> <td>   14.521</td> <td> 0.000</td> <td>    0.193</td> <td>    0.253</td>
</tr>
<tr>
  <th>ar.L17</th>    <td>    0.0689</td> <td>    0.016</td> <td>    4.181</td> <td> 0.000</td> <td>    0.037</td> <td>    0.101</td>
</tr>
<tr>
  <th>ar.L18</th>    <td>   -0.0471</td> <td>    0.018</td> <td>   -2.653</td> <td> 0.008</td> <td>   -0.082</td> <td>   -0.012</td>
</tr>
<tr>
  <th>ar.L19</th>    <td>    0.0298</td> <td>    0.017</td> <td>    1.734</td> <td> 0.083</td> <td>   -0.004</td> <td>    0.063</td>
</tr>
<tr>
  <th>ar.L20</th>    <td>   -0.0739</td> <td>    0.014</td> <td>   -5.391</td> <td> 0.000</td> <td>   -0.101</td> <td>   -0.047</td>
</tr>
<tr>
  <th>sigma2</th>    <td>    0.0005</td> <td> 6.23e-06</td> <td>   74.918</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.53</td> <th>  Jarque-Bera (JB):  </th> <td>4934.64</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.47</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.70</td> <th>  Skew:              </th>  <td>-0.14</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>  <td>8.79</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
predictions = model_fit.predict(start=0, end=len(train)+len(test)-1)
```

    c:\Users\julia\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:834: ValueWarning:
    
    No supported index is available. Prediction results will be given with an integer index beginning at `start`.
    
    


```python
predictions
```




    0       0.005775
    1      -0.061215
    2      -0.066195
    3      -0.025802
    4       0.015590
              ...   
    3767    0.005775
    3768    0.005775
    3769    0.005775
    3770    0.005775
    3771    0.005775
    Name: predicted_mean, Length: 3772, dtype: float64




```python
check1 = AAPL_data_ARIMA
```


```python
check1 = pd.DataFrame(check1).reset_index()
```


```python
check1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_day</th>
      <th>5d_ret</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-01-09 00:00:00-05:00</td>
      <td>-0.079244</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-01-10 00:00:00-05:00</td>
      <td>-0.086749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-01-11 00:00:00-05:00</td>
      <td>-0.040878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-01-14 00:00:00-05:00</td>
      <td>0.006417</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-01-15 00:00:00-05:00</td>
      <td>-0.012905</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3767</th>
      <td>2022-12-23 00:00:00-05:00</td>
      <td>-0.019701</td>
    </tr>
    <tr>
      <th>3768</th>
      <td>2022-12-27 00:00:00-05:00</td>
      <td>-0.017678</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>2022-12-28 00:00:00-05:00</td>
      <td>-0.047317</td>
    </tr>
    <tr>
      <th>3770</th>
      <td>2022-12-29 00:00:00-05:00</td>
      <td>-0.043115</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>2022-12-30 00:00:00-05:00</td>
      <td>-0.017394</td>
    </tr>
  </tbody>
</table>
<p>3772 rows × 2 columns</p>
</div>




```python
check1['predictions'] = predictions.values
```


```python
check1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_day</th>
      <th>5d_ret</th>
      <th>predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-01-09 00:00:00-05:00</td>
      <td>-0.079244</td>
      <td>0.005775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-01-10 00:00:00-05:00</td>
      <td>-0.086749</td>
      <td>-0.061215</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-01-11 00:00:00-05:00</td>
      <td>-0.040878</td>
      <td>-0.066195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-01-14 00:00:00-05:00</td>
      <td>0.006417</td>
      <td>-0.025802</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-01-15 00:00:00-05:00</td>
      <td>-0.012905</td>
      <td>0.015590</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3767</th>
      <td>2022-12-23 00:00:00-05:00</td>
      <td>-0.019701</td>
      <td>0.005775</td>
    </tr>
    <tr>
      <th>3768</th>
      <td>2022-12-27 00:00:00-05:00</td>
      <td>-0.017678</td>
      <td>0.005775</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>2022-12-28 00:00:00-05:00</td>
      <td>-0.047317</td>
      <td>0.005775</td>
    </tr>
    <tr>
      <th>3770</th>
      <td>2022-12-29 00:00:00-05:00</td>
      <td>-0.043115</td>
      <td>0.005775</td>
    </tr>
    <tr>
      <th>3771</th>
      <td>2022-12-30 00:00:00-05:00</td>
      <td>-0.017394</td>
      <td>0.005775</td>
    </tr>
  </tbody>
</table>
<p>3772 rows × 3 columns</p>
</div>




```python
fig = go.Figure()

fig.add_trace(go.Scatter(name='Actual',x=check1['Date_day'],y=check1['5d_ret']))
fig.add_trace(go.Scatter(name='Predicted',x=check1['Date_day'],y=check1['predictions']))

fig.update_layout(
    title_text="5-day return ARIMA"
)
fig.show()
```




```python

```
