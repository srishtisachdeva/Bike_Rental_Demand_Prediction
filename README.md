# 🚲 Bike Rental Demand Predictor

Predict how many bikes people will rent in one hour in Seoul. The prediction uses the hour, the weather and the type of day. The project compares a **Gradient Boosting** model with a **Neural Network** and shows both in a Streamlit website.

## Dataset

[Seoul Bike Sharing Demand Prediction](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction) on Kaggle: 8,760 rows, one for every hour of one year (Dec 2017 to Nov 2018), with 14 columns.

Each row has the number of bikes rented in that hour, which is the target, along with the weather (temperature, humidity, wind, rain, snow and more), the season and whether it was a holiday.

Cleaning steps:
- the file contains a `°` symbol that is not UTF-8, so it is read as latin-1
- column names like `Temperature(°C)` are renamed to short names
- rows from days when the service was closed are removed, because those hours always show 0 bikes
- rows with 0% humidity are removed as sensor errors
- the date is converted to a weekend yes/no column
- technical columns (dew point, solar radiation, visibility) are dropped to keep the inputs simple

## Pipeline (`main.py`)

1. **Load:** download the dataset with `kagglehub`.
2. **Clean:** apply the steps above (8,760 rows go down to 8,448).
3. **Explore:** average bikes by season and by rain, plus 4 plots.
4. **Prepare:** one-hot encode the text columns (`pd.get_dummies`), make an 80/20 train/test split and scale the features (`StandardScaler`).
5. **Train:** `GradientBoostingRegressor` and `MLPRegressor`, a neural network with layers of 64 and 32.
6. **Compare:** score both models on the test set with MAE and R², and pick the winner.

## Results

| Model | Average error (MAE) | R² |
|---|---|---|
| Gradient Boosting | 163 bikes | 0.852 |
| **Neural Network** | **140 bikes** | **0.881** |

The Neural Network wins. In an average hour about 730 bikes are rented, and its guesses are off by about 140 bikes.

What the data shows:
- Summer is the busiest season, with about 1,034 bikes per hour. Winter has about 226.
- Rain cuts demand sharply: about 766 bikes per hour without rain and 167 with rain.
- Demand peaks at the evening commute, around 6 pm.

## Website (`app.py`)

- **Sidebar:** enter the hour, weather, season, holiday and weekend.
- **Predict tab:** the number of bikes predicted by both models, updated as you change the inputs.
- **Explore data tab:** pick one of the plots.
- **Compare models tab:** the error table, the winner and a bar chart.

The light teal theme is set in `.streamlit/config.toml`.

## How to run

```bash
pip install -r requirements.txt
cd bike-demand-predictor
streamlit run app.py
```

Run the command from inside the project folder so that Streamlit finds the theme file. The first start takes about 30 seconds while the models train. You can also run `python main.py` to see the results in the terminal and the plots in windows.

## Project structure

```
bike-demand-predictor/
├── main.py              # data pipeline: load, clean, explore, train, compare
├── app.py               # Streamlit website
├── .streamlit/
│   └── config.toml      # website colours
├── requirements.txt
└── README.md
```

## Tech used

Python, pandas, scikit-learn, matplotlib, Streamlit, kagglehub

