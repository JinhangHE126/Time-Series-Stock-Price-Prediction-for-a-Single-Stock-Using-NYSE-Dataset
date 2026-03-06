# Time-Series-Stock-Price-Prediction-for-a-Single-Stock-Using-NYSE-Dataset

> links:[https://www.kaggle.com/datasets/dgawlik/nyse/data](https://www.kaggle.com/datasets/dgawlik/nyse/data)
>
> Goal: use time series models (e.g., LSTM) on a single stock (e.g., AAPL) from recent years' trends to forecast future short-term closing prices

## Understand the dataset

- prices.csv 
  > raw, as-is daily prices. Most of data spans from 2010 to the end 2016, for companies new on stock market date range is shorter. There have been approx. 140 stock splits in that time, this set doesn't account for that.
     * 
- prices-split-adjustment.csv
  > same as prices, but there have been added adjustments for splits.
- securities.csv (have null values)
  > general description of each company with division on sectors
  - securities shape: (505, 8)
  - securities.csv null values:
   Date first added    198
   dtype: int64
- fundamentals.csv (have null values)
  > metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators
  - fundamentals shape: (1781, 79)
  - fundamentals.csv null values:
   Cash Ratio                      299
   Current Ratio                   299
   Quick Ratio                     299
   For Year                        173
   Earnings Per Share              219
   Estimated Shares Outstanding    219
   dtype: int64






## Modelling
> method:Prep sequences (60-day input, 5-day output); scale with MinMax. Baseline: ARIMA. Main: LSTM (50 units, Dense output); train with MSE/Adam, time-split (80/20).



