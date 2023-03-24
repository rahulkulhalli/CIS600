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

## Weather Features to Consider for Accurate Predictions

The importance of weather features in predictions depends on the specific problem or application you are addressing. However, some commonly important weather features include:

1. **Temperature**: Ambient temperature can impact energy consumption, agricultural productivity, human comfort, and various other aspects of daily life.

2. **Precipitation**: Rainfall, snowfall, and other forms of precipitation affect water resources, agriculture, transportation, and infrastructure.

3. **Humidity**: Relative humidity can impact human comfort, air quality, and the efficiency of certain energy systems, such as cooling equipment.

4. **Wind speed and direction**: Wind can affect renewable energy production (wind turbines), aviation, marine transportation, and the dispersion of air pollutants.

5. **Solar radiation**: Solar irradiance is crucial for solar energy production and can also impact temperature, evaporation rates, and plant growth.

6. **Cloud cover**: The amount of cloud cover can influence solar radiation, temperature, and visibility, which can impact energy production, aviation, and other sectors.

7. **Atmospheric pressure**: Changes in atmospheric pressure can impact weather systems and are often used in weather forecasting models.

8. **Visibility**: Reduced visibility due to fog, haze, or other factors can have significant impacts on transportation, particularly aviation and marine navigation.

9. **Extreme events**: Severe weather events such as storms, hurricanes, tornadoes, and heatwaves can have a wide range of impacts on human life, infrastructure, and economic activity.

The specific combination of weather features that are most important for predictions will depend on the context and the goals of the analysis. It is essential to understand the relationships between different weather features and the variables you are trying to predict to build accurate and reliable models.


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
