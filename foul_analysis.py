import pandas as pd
import numpy as np
import pybaseball as pyb
import matplotlib.pyplot as plt
from pybaseball import pitching_stats
from sklearn.metrics import r2_score
from pybaseball import cache

cache.enable()

IP_threshold = 50;

data = pitching_stats(2002,2023)
data = data.query('IP > @IP_threshold')
#print(data.keys().values.tolist())
data.rename(columns = {'CSW%':'CSW'}, inplace=True)
data.insert(13, "Fouls", round(data.Strikes - (data.Pitches * data.CSW)))
data.insert(14, "Foul_rate", data.Fouls / data.Pitches)

#print(data.head(10))

coefficients_ERA = np.polyfit(data['Foul_rate'], data['ERA'], 1)
regression_line_ERA = np.poly1d(coefficients_ERA)
r2_ERA = r2_score(data['ERA'], regression_line_ERA(data['Foul_rate']))

#coefficients_xERA = np.polyfit(data['Foul_rate'], data['xERA'], 1)
#regression_line_xERA = np.poly1d(coefficients_xERA)
#r2_xERA = r2_score(data['xERA'], regression_line_xERA(data['Foul_rate']))

coefficients_WAR = np.polyfit(data['Foul_rate'], data['WAR'], 1)
regression_line_WAR = np.poly1d(coefficients_WAR)
r2_WAR = r2_score(data['WAR'], regression_line_WAR(data['Foul_rate']))

coefficients_K = np.polyfit(data['Foul_rate'], data['K/9'], 1)
regression_line_K = np.poly1d(coefficients_K)
r2_K = r2_score(data['K/9'], regression_line_K(data['Foul_rate']))

#data only available from 2015+
#coefficients_barrel = np.polyfit(data['Foul_rate'], data['Barrel%'], 1)
#regression_line_barrel = np.poly1d(coefficients_barrel)
#r2_barrel = r2_score(data['Barrel%'], regression_line_barrel(data['Foul_rate']))

coefficients_IP = np.polyfit(data['Foul_rate'], data['IP'], 1)
regression_line_IP = np.poly1d(coefficients_IP)
r2_IP = r2_score(data['IP'], regression_line_IP(data['Foul_rate']))

coefficients_SIERA = np.polyfit(data['Foul_rate'], data['SIERA'], 1)
regression_line_SIERA = np.poly1d(coefficients_SIERA)
r2_SIERA = r2_score(data['SIERA'], regression_line_SIERA(data['Foul_rate']))

print("ERA r2 = ",r2_ERA)
#print("xERA r2 = ",r2_xERA)
print("WAR r2 = ",r2_WAR)
print("K/9 r2 = ",r2_K)
#print("barrel r2 = ",r2_barrel)
print("IP r2 = ",r2_IP)
print("SIERA r2 = ",r2_SIERA)

plt.figure()
plt.scatter(data['Foul_rate'],data['K/9'], c='blue')
plt.plot(data['Foul_rate'],regression_line_K(data['Foul_rate']), 'r-')
plt.text(min(data['Foul_rate']), max(data['K/9']), f'R^2 = {r2_K:.4f}', fontsize=12)
plt.title("Foul Rate vs K/9")
plt.xlabel("Foul Rate")
plt.ylabel("K/9")
plt.grid(True)
plt.show()

data = data[['IDfg', 'Season', 'Name', 'Team', 'WAR', 'ERA', 'IP', 'CSW', 'K/9', 'Foul_rate']].dropna()
data = data.sort_values(
    by="Foul_rate",
    ascending=False,
    kind="mergesort"
)

print(data.head(10))




