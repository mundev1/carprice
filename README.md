# Car Price Prediction Using Machine Learning

## Overview

This project predicts the price of a used car based on various features such as car name, company, manufacturing year, kilometers driven, and fuel type. The model is built using Python and Scikit-Learn and uses Linear Regression for price prediction.

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Encoding using One-Hot Encoding
* Linear Regression Model Training
* Model Evaluation using R² Score
* Model Serialization using Pickle
* Predict Price of New Cars

## Dataset

The dataset used is `quikr_car.csv`, which contains information about used cars including:

* Car Name
* Company
* Year
* Price
* Kilometers Driven
* Fuel Type

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Pickle

## Project Workflow

### 1. Data Cleaning

* Removed invalid year values.
* Removed rows with "Ask For Price".
* Converted price values to integers.
* Cleaned kilometer-driven values.
* Removed missing fuel type records.
* Filtered out extreme price outliers.

### 2. Exploratory Data Analysis

Visualizations include:

* Company vs Price Boxplot
* Year vs Price Swarm Plot
* Kilometers Driven vs Price Relationship
* Fuel Type vs Price Distribution
* Company, Fuel Type, and Year Combined Analysis

### 3. Feature Engineering

Selected features:

* Name
* Company
* Year
* Kilometers Driven
* Fuel Type

Target variable:

* Price

### 4. Model Building

A machine learning pipeline is created using:

* OneHotEncoder for categorical features
* Linear Regression for prediction

### 5. Model Evaluation

The model is evaluated using the R² Score metric.

### 6. Model Saving

The trained model is saved as:

```python
LinearRegressionModel.pkl
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd car-price-prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run the Project

```bash
python car_price_prediction.py
```

## Example Prediction

```python
pipe.predict(pd.DataFrame(
    columns=['name','company','year','kms_driven','fuel_type'],
    data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)
))
```

## Output

The model predicts the estimated selling price of the car based on the provided details.

## Model Performance

The project uses multiple train-test splits and selects the best-performing model based on the highest R² Score.

## Future Improvements

* Use advanced regression models such as Random Forest Regressor and XGBoost.
* Deploy the model using Flask or Streamlit.
* Add more car specifications for improved accuracy.
* Create a web application for real-time predictions.

## Author
Naveen Rathoue

Naven

B.Tech Artificial Intelligence & Machine Learning

COER University
