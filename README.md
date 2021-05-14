# Assignment-3
This assignment is about testing single factor in A share market. We test two common value factor, namely, EP ratio and market value. EP is computed by dividing the net profit TTM by market value. It is the inverse of PE, which signifies the valuation of stocks, We use its inverse considering that in case of negative proft, relative small negative profit will result in a very negative PE, which is not wisher for. 

You can find the dataset used in this link: https://pan.baidu.com/s/1TPnTQxEBAhCL1cPoQc6dSA (8438)

# Procedure 
stock pool : all tradable stocks in A shares are included, except for those that are listed in less a year or missing the factor values needed. 

data preparation: all factor values underwent the following transformation

  winsorization: clip the value that are outside the range of 5 times median absolute deviation of the smaple median
  
  fill na values: we use median value in that industry for those missing factor values
  
  neutralization: run regression of factos values on market value and industry dummy variables and use the residual as the surrogate for factor value
  
backtesting period: the backtesting period starts from 2010 to 2020. On the first trade day of every month, rebalance according to the factor values on the day before. Stratified classification is used where we group stocks into 5 equal classes according to the value of factors. Stocks are bought with equal weights.

transaction cost: simple fixed fee is used at 0.003

evaluation: we compare the net value curve of each group and calculate the value process of long-short portfolio. Standard metrics like Sharpe ratio can also be computed

# Results
Here is the net value curve of five groups and longing group with high EP and shorting low EP class
![avatar](https://github.com/algo21-220040039/Assignment-3/raw/main/EP%20result.png)

![avatar](https://github.com/algo21-220040039/Assignment-3/raw/main/EP%20long%20short.png)
