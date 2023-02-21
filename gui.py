import streamlit
import requests
import json
import pandas as pd

df = pd.read_csv("train.csv")
categoric_type = df.select_dtypes(include=['object'])
numeric_type = df.select_dtypes(exclude=['object'])
for col in categoric_type:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in numeric_type:
    df[col] = df[col].fillna(df[col].mean())


def run():
    streamlit.title("House Price Prediction")
    #numerical inputs
    OverallQual = streamlit.number_input("OverallQual",min_value=1, max_value=10, value=5, step=1)
    GrLivArea = streamlit.number_input("GrLivArea", min_value=0, max_value=10000, value=1629, step=1)
    GarageCars = streamlit.number_input("GarageCars", min_value=0, max_value=10, value=2, step=1)
    YearBuilt = streamlit.number_input("YearBuilt", min_value=1600, max_value=2100, value=1997, step=1)
    TotalBsmtSF = streamlit.number_input("TotalBsmtSF", min_value=0, max_value=10000, value=928, step=1)
    FirstFlrSF = streamlit.number_input("FirstFlrSF",  min_value=0, max_value=10000, value=928, step=1)
    FullBath = streamlit.number_input("FullBath", min_value=0, max_value=10, value=2, step=1)
    YearRemodAdd = streamlit.number_input("YearRemodAdd", min_value=1600, max_value=2100, value=1998, step=1)
    Fireplaces = streamlit.number_input("Fireplaces", min_value=0, max_value=10, value=1, step=1)
    BsmtFinSF1 = streamlit.number_input("BsmtFinSF1", min_value=0, max_value=10000, value=791, step=1)
    LotFrontage = streamlit.number_input("LotFrontage", min_value=0, max_value=1000, value=74, step=1)
    OverallCond = streamlit.number_input("OverallCond", min_value=1, max_value=10, value=5, step=1)
    SecondFlrSF = streamlit.number_input("SecondFlrSF", min_value=0, max_value=10000, value=701, step=1)
    MasVnrArea = streamlit.number_input("MasVnrArea", min_value=0, max_value=10000, value=0, step=1)
    LotArea = streamlit.number_input("LotArea", min_value=0, max_value=1000000, value=13830, step=1)
    HalfBath = streamlit.number_input("HalfBath", min_value=0, max_value=10, value=1, step=1)
    #categorical inputs
    Neighborhood = streamlit.selectbox("Neighborhood", df.Neighborhood.unique(), index=17)
    ExterQual = streamlit.selectbox("ExterQual", df.ExterQual.unique(), index=1)
    KitchenQual = streamlit.selectbox("KitchenQual", df.KitchenQual.unique(), index=1)
    GarageFinish = streamlit.selectbox("GarageFinish", df.GarageFinish.unique(), index=2)
    FireplaceQu = streamlit.selectbox("FireplaceQu", df.FireplaceQu.unique(), index=1)
    Foundation = streamlit.selectbox("Foundation", df.Foundation.unique(), index=0)
    HeatingQC = streamlit.selectbox("HeatingQC", df.HeatingQC.unique(), index=1)
    MSZoning = streamlit.selectbox("MSZoning", df.MSZoning.unique(), index=0)
    Exterior1st = streamlit.selectbox("Exterior1st", df.Exterior1st.unique(), index=0)
    BsmtFinType1 = streamlit.selectbox("BsmtFinType1", df.BsmtFinType1.unique(), index=0)
    LotShape = streamlit.selectbox("LotShape", df.LotShape.unique(), index=1)


    data = { 
        'OverallQual': OverallQual,
        'Neighborhood': Neighborhood,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'ExterQual': ExterQual,
        'KitchenQual': KitchenQual,
        'YearBuilt': YearBuilt,
        'GarageFinish': GarageFinish,
        'TotalBsmtSF': TotalBsmtSF,
        'FirstFlrSF': FirstFlrSF,
        'FullBath': FullBath,
        'FireplaceQu': FireplaceQu,
        'YearRemodAdd': YearRemodAdd,
        'Foundation': Foundation,
        'Fireplaces': Fireplaces,
        'HeatingQC': HeatingQC,
        'BsmtFinSF1': BsmtFinSF1,
        'MSZoning': MSZoning,
        'LotFrontage': LotFrontage,
        'Exterior1st': Exterior1st,
        'OverallCond': OverallCond,
        'BsmtFinType1': BsmtFinType1, 
        'SecondFlrSF': SecondFlrSF,
        'MasVnrArea': MasVnrArea,
        'LotArea': LotArea,
        'LotShape': LotShape,
        'HalfBath': HalfBath
    }
    
    if streamlit.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")
    
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()