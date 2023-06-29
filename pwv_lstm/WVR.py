"""
see also
https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
https://stackoverflow.com/questions/72978955/time-intervals-to-evenly-spaced-time-series?rq=1
"""

import pandas as pd
import matplotlib.pyplot as plt

filename = "TA-2"

series = pd.read_csv(# "WVR_UdeC.dat",
                     "data/{}.dat".format(filename),
                     parse_dates=[0],
                     sep=' ',
                     header=10,
                     names=['MJD', 'PWV'])

series.squeeze("columns")
series['MJD'] = series['MJD'].astype(float)
series['date'] = series['MJD'] + 2400000.5
series['date'] = pd.to_datetime(
    series['date'] - pd.Timestamp(0).to_julian_date(),
    unit='D',
    utc=True
)
series = series.drop('MJD', axis=1)
columns_switch = ['date', 'PWV']
series.reindex(columns=columns_switch)
series.set_index('date', inplace=True)
downsampled = series.resample('6H').mean()
interpolated = downsampled.interpolate(method='linear')

interpolated.to_csv("{}.csv".format(filename))

plt.plot(interpolated)
plt.show()
