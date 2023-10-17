import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import timedelta
from timeit import default_timer as default_timer


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def fetch_data():
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
    downsampled = series.resample('3H').mean()
    interpolated = downsampled.interpolate(method='linear')

    return interpolated


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to(device)
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=200,
                            num_layers=1,
                            batch_first=True)
        self.linear = nn.Linear(200, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


def main():
    _start_time = default_timer()
    lookback = 30

    df = fetch_data()
#    df = pd.read_csv('TA-2.csv')
#    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.index > '2007-01-01') & (df.index < '2008-01-01')]
    timeseries = df[["PWV"]].values.astype('float32')
#    delta = int(timedelta(days=1) * 5 / (df['date'][1] - df['date'][0]))
    delta = int(timedelta(days=1) * 5 / (df.iloc[1].name - df.iloc[0].name))

    # train-test split for time series
    train_size = len(timeseries) - delta
    test_size = len(timeseries) - lookback - delta
    train, test = timeseries[:train_size], timeseries[test_size:]

    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    model = AirModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=8)

    n_epochs = 500
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch,
                                                             train_rmse,
                                                             test_rmse))

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[test_size + lookback:len(timeseries)] = \
            model(X_test)[:, -1, :]
    # plot
    print("Time consumed: {}".format(default_timer() - _start_time))
    plt.plot(timeseries)
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()


if __name__ == "__main__":
    main()
