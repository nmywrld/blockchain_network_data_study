import pandas as pd
import os
from coinmetrics.api_client import CoinMetricsClient
from os import environ

client = CoinMetricsClient()

# ethmetric_data = client.get_asset_metrics(assets='eth', 
# 										metrics=['ReferenceRateUSD', 'AdrActCnt', 'HashRate', 'TxCnt'], 
# 										frequency='1d', 
# 										start_time='2019-03-17T14:17:00.000Z', 
# 										end_time='2022-03-17T14:17:00.000Z')

# ethmetric_data_df = pd.DataFrame.from_dict(ethmetric_data)

# print(ethmetric_data_df.head())
	
# ethmetric_data_df.to_csv('ETH.csv')  


btcmetric_data = client.get_asset_metrics(assets='btc', 
										metrics=['ReferenceRateUSD', 'AdrActCnt', 'HashRate', 'TxCnt'], 
										frequency='1d', 
										start_time='2011-01-11T14:17:00.000Z', 
										end_time='2022-05-07T14:17:00.000Z')

btcmetric_data_df = pd.DataFrame.from_dict(btcmetric_data)

print(btcmetric_data_df.head())
	
btcmetric_data_df.to_csv('crypto_datav1/BTC.csv')