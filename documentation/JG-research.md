# CSE 691 Project

This project aims to predict if a flight's arrival time into Syracuse (SYR) would be early, on-time, delayed, or severely delayed, 1-4 days in advance. We will use various regression and classification techniques excluding neural networks or deep learning.

## Steps to Complete the Project

### 1. Data Collection and Preprocessing

- Collect flight data from the Bureau of Transportation Statistics, United Airlines, and other sources as needed.
- Create a new target variable based on the ground truth criteria provided (early, on-time, delayed, or severely delayed).
- Collect additional data, such as weather forecasts, which might impact flight delays.
- Merge and preprocess the datasets by cleaning, handling missing values, and encoding categorical variables.

### 2. Feature Engineering

- Create relevant features that might impact flight delays, such as scheduled departure time, day of the week, season, and weather-related variables.
- Normalize or standardize the data if needed.

### 3. Data Split

- Split the dataset into a training set and a testing set (e.g., using an 80:20 ratio) with stratification to ensure the distribution of the target variable is maintained.

### 4. Model Selection and Training

- Choose a suitable model based on the problem description (e.g., linear regression, logistic regression, Ridge or Lasso regression, or bagging regression).
- Train the model on the training set and validate its performance using cross-validation techniques.

### 5. Model Evaluation

- Evaluate the trained model on the testing set using appropriate metrics such as accuracy, precision, recall, and F1 score.

### 6. Final Predictions

- Use the trained model to make predictions on the provided CSV file.

## Important Weather Features for Predictions

Weather plays a significant role in flight delays and cancellations. Some important weather features to consider for your predictions include:

1. **Temperature**: Extreme temperatures, both high and low, can impact aircraft performance and lead to delays.

2. **Precipitation**: Rain, snow, sleet, and other forms of precipitation can reduce visibility, disrupt airport operations, and affect aircraft performance, resulting in delays.

3. **Wind speed and direction**: Strong winds, especially crosswinds, can make takeoffs and landings more difficult, leading to delays or cancellations. Wind direction may also affect the choice of runways in use at an airport.

4. **Visibility**: Poor visibility due to fog, mist, or heavy precipitation can significantly impact flight operations, causing delays or cancellations.

5. **Cloud cover**: Low cloud ceilings can affect flight operations, especially at airports with limited instrument landing systems or in situations where pilots rely on visual approaches.

6. **Thunderstorms and lightning**: Thunderstorms and lightning pose a significant risk to flights, often causing delays, diversions, or cancellations due to safety concerns.

7. **Turbulence**: Turbulence, which can be caused by various weather phenomena, can lead to delays as flights may need to take alternative routes or adjust their altitudes to avoid turbulent areas.

8. **Icing conditions**: Icing on aircraft wings, engines, or other surfaces can impact performance and pose a safety risk, leading to delays or cancellations.

9. **Extreme weather events**: Hurricanes, typhoons, blizzards, and other extreme weather events can cause widespread disruptions to air travel, resulting in numerous delays and cancellations.

It's important to note that weather conditions at both the origin and destination airports can impact flight delays. Additionally, the severity and combination of various weather features can also play a role. When incorporating weather data into your analysis, you may need to preprocess and aggregate the data to match the granularity of your flight data (e.g., hourly or daily).


## Flight Features to Consider for Accurate Predictions

To make accurate predictions for flight delays, you can consider the following features:

1. **Scheduled departure and arrival times**: Flights scheduled during peak hours might be more prone to delays due to increased air traffic.

2. **Day of the week**: Flight delays may be more common on certain days due to increased air traffic or other operational factors.

3. **Season or month**: Weather conditions and flight demand can vary by season, potentially influencing delays.

4. **Origin airport**: Delays can be more prevalent at certain airports due to operational efficiency or air traffic volume.

5. **Weather conditions at origin and destination**: Weather conditions such as temperature, precipitation, wind speed, and visibility can significantly impact flight delays.

6. **Historical flight delay data**: The past delay patterns for a particular flight or route can provide insights into the likelihood of future delays.

7. **Elapsed time since the last delay**: If the flight has recently experienced a delay, it might be more likely to be delayed again.

8. **Flight distance**: Longer flights might be more susceptible to delays due to factors like fueling, crew changes, or needing to address technical issues.

9. **Aircraft type**: Some aircraft types may have a higher likelihood of experiencing delays due to their performance characteristics or maintenance requirements.

10. **Carrier**: Different carriers might have varying operational efficiency and delay patterns.

When selecting features, it's essential to consider their relevance, quality, and potential multicollinearity. You can use feature selection techniques like recursive feature elimination, forward selection, or backward elimination to identify the most relevant features for your model. Additionally, before using them in your model, you may need to preprocess certain features (e.g., normalize or standardize continuous variables, or encode categorical variables).

## Example using Logistic Regression

Here's an example using logistic regression with scikit-learn in Python:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv("flight_data.csv")
weather_data = pd.read_csv("weather_data.csv")
data = data.merge(weather_data, on=["date", "origin_airport"])

# Create a new target variable based on the ground truth criteria
def classify_arrival_status(arrival_diff):
    if arrival_diff <= -10:
        return "early"
    elif -10 < arrival_diff <= 10:
        return "on-time"
    elif 10 < arrival_diff <= 30:
        return "delayed"
    else:
        return "severely delayed"

data["arrival_status"] = data["arrival_diff"].apply(classify_arrival_status)

# Feature engineering
data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek
data = pd.get_dummies(data, columns=["origin_airport"])

# Prepare X and y
feature_columns = [...] # Fill in the relevant feature columns
X = data[feature_columns]
y = data["arrival_status"]

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Make predictions on the provided CSV file
data_to_predict = pd.read_csv("prediction_data.csv")
data_to_predict = data_to_predict.merge(weather_data, on=["date", "origin_airport"])

# Feature engineering and standardization for prediction data
data_to_predict["day_of_week"] = pd.to_datetime(data_to_predict["date"]).dt.dayofweek
data_to_predict = pd.get_dummies(data_to_predict, columns=["origin_airport"])
X_to_predict = data_to_predict[feature_columns]
X_to_predict = scaler.transform(X_to_predict)

# Predict and save results
predictions = model.predict(X_to_predict)
data_to_predict["arrival_status"] = predictions
data_to_predict.to_csv("final_predictions.csv", index=False)
```
