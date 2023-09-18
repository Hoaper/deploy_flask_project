from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask("Churn Prediction")


numerical = [
    'tenure', 
    'monthlycharges', 
    'totalcharges'
]

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod'
]

le_enc_cols = ['gender', 'partner', 'dependents','paperlessbilling', 'phoneservice']
gender_map = {'male': 0, 'female': 1}
y_n_map = {'yes': 1, 'no': 0}

# Логистическая модель
model_file_path = "models/lr_model_churn_prediction.sav"
model = pickle.load(open(model_file_path, 'rb'))

encoding_model_file_path = "models/lr_model_churn_prediction_encode.sav"
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

treshold = 0.281


@app.route("/predict", methods=["POST"])
def predict():
    customers = request.get_json()
    data = pd.DataFrame(customers)

    scaler = MinMaxScaler()

    df_copy = data.copy()
    df_copy[numerical] = scaler.fit_transform(df_copy[numerical])

    for col in le_enc_cols:
        if col == 'gender':
            df_copy[col] = df_copy[col].map(gender_map)
        else:
            df_copy[col] = df_copy[col].map(y_n_map)

    dicts_df = df_copy[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    y_pred = model.predict_proba(X)[:, 1]
    churn_descision = (y_pred >= treshold)
    
    result = []
    for i in range(len(df_copy)):
        result.append({
            "churn_descision": bool(churn_descision[i]),
            "churn_probability": float(y_pred[i]),
            "customer_id": df_copy["customerid"][i],
        })


    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)