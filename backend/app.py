import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor

# import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option("display.max_columns", None)
df = pd.read_csv("C:/Users/HarryJoseph/Desktop/h/invstment/cvProjects/MLreact/backend/cardekho_dataset.csv")
df.drop(['brand','model', 'Unnamed: 0'], axis=1, inplace=True)
def detect_outliers(col):
    # Finding the IQR
    percentile25 = df[col].quantile(0.25)
    percentile75 = df[col].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    df.loc[(df[col]>upper_limit), col]= upper_limit
    df.loc[(df[col]<lower_limit), col]= lower_limit    
    return df

num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
cat_features = [feature for feature in df.columns if df[feature].dtype == 'O']
discrete_features=[feature for feature in num_features if len(df[feature].unique())<=25]
continuous_features=[feature for feature in num_features if feature not in discrete_features]

for col in continuous_features:
         detect_outliers(col)

from sklearn.model_selection import train_test_split
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type','fuel_type','transmission_type']
binary_columns = ['car_name']


numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
binary_transformer = BinaryEncoder()

default_values = {}
numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
categorical_features = ['car_name', 'seller_type', 'fuel_type', 'transmission_type']
for feature in numerical_features:
    default_values[feature] = X[feature].median()  # Use median for numerical

for feature in categorical_features:
    default_values[feature] = X[feature].mode()[0]



unique_values = {feature: df[feature].unique().tolist() for feature in categorical_features}


preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
         ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)
        
    ]
)



preprocessor.fit(X)
X=preprocessor.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model=CatBoostRegressor(verbose=False)
model.fit(X_train, y_train)

from flask import Flask, request, jsonify
from flask_cors import CORS
# Import your ML model and other necessary libraries
# from your_model_file import predict_price

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        data=pd.DataFrame([data])
        print(data.iloc[0,:])
        data=preprocessor.transform(data)
        predicted_price=model.predict(data).tolist()
        
        return jsonify({'predicted_price': predicted_price[0]})
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

@app.route('/get-unique-values', methods=['GET'])
def get_unique_values():
    try:
        return jsonify({
            'unique_values': unique_values,
            'default_values': default_values
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
