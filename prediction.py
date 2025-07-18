# fetching today's data to predict tomorrow's temperature
import datetime
import urllib.request
import json
import joblib
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

yesterday = datetime.date.today() - datetime.timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")

url = (
    "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization="
    + os.getenv("API_KEY", "")
    + "&format=JSON&StationId=C0AD10"
)

# print(url)

# fetch data
with urllib.request.urlopen(url) as resp:
    data = json.load(resp)

station_list = data["records"]["Station"]
rec = station_list[0]

# pass data to X
we = rec["WeatherElement"]
X_test = pd.DataFrame(
    [
        {
            "Tx": float(we["AirTemperature"]),
            "RH": float(we["RelativeHumidity"]),
            "WS": float(we["WindSpeed"]),
            "WD": float(we["WindDirection"]),
            "Precp": float(we["Now"]["Precipitation"]),
            "TxMaxAbs": float(
                we["DailyExtreme"]["DailyHigh"]["TemperatureInfo"]["AirTemperature"]
            ),
            "TxMinAbs": float(
                we["DailyExtreme"]["DailyLow"]["TemperatureInfo"]["AirTemperature"]
            ),
        }
    ]
)

# print(X_test)

# load the model
model_loaded = joblib.load("./forest_model.joblib")
y_pred = model_loaded.predict(X_test)
print(f"tomorrow's temperature is {y_pred[0]:.2f}°C")
