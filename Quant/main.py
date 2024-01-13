import quandl
import fbprophet
import plotly
from matplotlib import pyplot as plt
from stocker import Stocker
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

ms = Stocker('MSFT')

stock_history = ms.stock
stock_history.head()

ms.plot_stock()

ms.buy_and_hold(start_date='1986-03-13', end_date='2018-01-16', nshares=100)

ms.buy_and_hold(start_date='1999-01-05', end_date='2002-01-03', nshares=100)

model, model_data = ms.create_prophet_model(days=0)

model.plot_components(model_data)

ms.changepoint_date_analysis()

model, future = ms.create_prophet_model(days=180)

amazon = Stocker('AMZN')

amazon.plot_stock()
amazon.plot_stock(stats=['Daily Change'])

_, model_data = amazon.create_prophet_model(days=90)

amazon.evaluate_prediction()

amazon.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

amazon.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03', changepoint_priors=[0.1, 0.2, 0.4, 0.6, 0.8])

amazon.changepoint_prior_scale = 0.6

amazon.evaluate_prediction()

amazon.predict_future(days=10)

