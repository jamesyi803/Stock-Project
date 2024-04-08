import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# define class ticker
class Ticker:
    def __init__(self):
        self.spy = self.spy()
        self.btc = self.btc()
        self.eth = self.eth()

    def spy(self):
        spy_ticker = yf.Ticker("SPY")
        return spy_ticker.history(period="10y")

    def btc(self):
        btc_ticker = yf.Ticker("BTC-USD")
        return btc_ticker.history(period="10y")

    def eth(self):
        eth_ticker = yf.Ticker("ETH-USD")
        return eth_ticker.history(period="10y")

ticker = Ticker()
spy = ticker.spy
btc = ticker.btc
eth = ticker.eth

# show only relevant data
# ":" select all the rows (dates) and anything after the "," selects all the columns (the data we want to see)
spy = spy.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
btc = btc.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
eth = eth.loc[:, ["Open", "High", "Low", "Close", "Volume"]]

# add tomorrow
spy["Tomorrow"] = spy["Close"].shift(-1)
btc["Tomorrow"] = btc["Close"].shift(-1)
eth["Tomorrow"] = eth["Close"].shift(-1)

# add target
# 1 if tomorrow > today else 0
spy["Target"] = spy["Tomorrow"] > spy["Close"].astype(int)
btc["Target"] = btc["Tomorrow"] > btc["Close"].astype(int)
eth["Target"] = eth["Tomorrow"] > eth["Close"].astype(int)
# machine learning model
model = RandomForestClassifier(n_estimators=300, min_samples_split=25, random_state=1)

# train and test
train_spy = spy.iloc[:-200]
train_btc = btc.iloc[:-200]
train_eth = eth.iloc[:-200]

test_spy = spy.iloc[-200:]
test_btc = btc.iloc[-200:]
test_eth = eth.iloc[-200:]

# predict
predictors = ["Close", "Volume", "Open", "High", "Low"]

def predict_spy(train_spy,test,predictors,model):
    model.fit(train_spy[predictors],train_spy["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined

def backtest_spy(data,model,predictors,start=2000,step=200):
    all_predictions = []

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_spy(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def predict_btc(train_btc,test_btc,predictors,model):
    model.fit(train_btc[predictors],train_btc["Target"])
    preds = model.predict_proba(test_btc[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test_btc.index,name="Predictions")
    combined = pd.concat([test_btc["Target"],preds],axis=1)
    return combined

def backtest_btc(data,model,predictors,start=2000, step=200):
    all_predictions = []

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_btc(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def predict_eth(train_eth,test_eth,predictors,model):
    model.fit(train_eth[predictors],train_eth["Target"])
    preds = model.predict_proba(test_eth[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test_eth.index,name="Predictions")
    combined = pd.concat([test_eth["Target"],preds],axis=1)
    return combined

def backtest_eth(data,model,predictors,start=2000, step=200):
    all_predictions = []

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_eth(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# count how many of each prediction was made
value_spy = predictions_spy["Predictions"].value_counts()
value_btc = predictions_btc["Predictions"].value_counts()
value_eth = predictions_eth["Predictions"].value_counts()

# precision score
precision_score_spy = precision_score(predictions_spy["Target"],predictions_spy["Predictions"])
precision_score_btc = precision_score(predictions_btc["Target"],predictions_btc["Predictions"])
precision_score_eth = precision_score(predictions_eth["Target"],predictions_eth["Predictions"])

# print("The accuracy score for spy in the past 2000 trading days is: " + str(precision_score_spy))
# print("The accuracy score for btc in the past 2000 trading days is: " + str(precision_score_btc))
# print("The accuracy score for eth in the past 2000 trading days is: " + str(precision_score_eth))

percent_spy = predictions_spy["Target"].value_counts() / predictions_spy.shape[0]
percent_btc = predictions_btc["Target"].value_counts() / predictions_btc.shape[0]
percent_eth = predictions_eth["Target"].value_counts() / predictions_eth.shape[0]

# print("The percentage for of days spy went up is " + str(percent_spy))
# print("The percentage for of days btc went up is " + str(percent_btc))
# print("The percentage for of days eth went up is " + str(percent_eth))

horizons = [2,5,60,250,1000]

new_predictors_spy = []
new_predictors_btc = []
new_predictors_eth = []

for horizon in horizons:
    rolling_avg_spy = spy.rolling(horizon).mean()
    rolling_avg_btc = btc.rolling(horizon).mean()
    rolling_avg_eth = eth.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    spy[ratio_column] = spy["Close"] / rolling_avg_spy["Close"]
    btc[ratio_column] = btc["Close"] / rolling_avg_btc["Close"]
    eth[ratio_column] = eth["Close"] / rolling_avg_eth["Close"]

    trend_column = f"Trend_{horizon}"
    spy[trend_column] = spy.shift(1).rolling(horizon).sum()["Target"]
    btc[trend_column] = btc.shift(1).rolling(horizon).sum()["Target"]
    eth[trend_column] = eth.shift(1).rolling(horizon).sum()["Target"]

    new_predictors_spy += [ratio_column,trend_column]
    new_predictors_btc += [ratio_column,trend_column]
    new_predictors_eth += [ratio_column,trend_column]

spy = spy.dropna()
btc = btc.dropna()
eth = eth.dropna()

predictions_spy = backtest_spy(spy,model,new_predictors_spy)
predictions_btc = backtest_btc(btc,model,new_predictors_btc)
predictions_eth = backtest_eth(eth,model,new_predictors_eth)


print(predictions_spy["Predictions"].value_counts())
print(predictions_btc["Predictions"].value_counts())
print(predictions_eth["Predictions"].value_counts())
