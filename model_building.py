import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# saving model
import joblib

# read csv file
weather = pd.read_csv("./466881_2024_daily.csv")
weather = weather.rename(columns={"Unnamed: 0": "Date"})
weather = weather.drop_duplicates(subset="Date", keep="first")
weather = weather.set_index(["Date"])

# filter the cols
valid_col = weather.select_dtypes(include="number").columns

# replace neg number to NaN
weather = weather[valid_col].mask(weather[valid_col] < 0, np.nan)
weather = weather.ffill()

# test repeat date
weather.index = pd.to_datetime(weather.index)
weather.index.year.value_counts().sort_index()

# determine which cols we want
used_col = ["Tx", "RH", "WS", "WD", "Precp", "TxMaxAbs", "TxMinAbs"]
weather[used_col]

weather["target"] = weather["Tx"].shift(-1)
weather = weather.dropna()

X = weather[used_col]
y = weather["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# create model and train
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
melb_preds = forest_model.predict(X_val)
print(f"diff = {mean_absolute_error(y_val, melb_preds)}")

# testing
X_test = pd.DataFrame(
    {
        "Tx": [15.9],
        "RH": [85.0],
        "WS": [2.3],
        "WD": [80.0],
        "Precp": [8.0],
        "TxMaxAbs": [17.3],
        "TxMinAbs": [14.8],
    }
)
pred = forest_model.predict(X_test)
print(pred)

# save model as joblib file
joblib.dump(forest_model, "forest_model.joblib")
