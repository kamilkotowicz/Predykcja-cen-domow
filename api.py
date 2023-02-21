import pickle
import uvicorn
import sklearn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(title='House Price Prediction', version='1.0',
              description='Linear Regression model is used for prediction')

model = pickle.load(open('model.pkl', 'rb'))


class Data(BaseModel):
    # Order of features is important!!!
    OverallQual: int
    Neighborhood: str
    GarageCars: int
    GrLivArea: int
    ExterQual: str
    KitchenQual: str
    YearBuilt: int
    TotalBsmtSF: int
    GarageFinish: str
    FullBath: int
    FirstFlrSF: int
    FireplaceQu: str
    YearRemodAdd: int
    Foundation: str
    Fireplaces: int
    HeatingQC: str
    MSZoning: str
    BsmtFinSF1: int
    LotFrontage: int
    Exterior1st: str
    BsmtFinType1: str 
    OverallCond: int
    SecondFlrSF: int
    LotArea: int
    HalfBath: int
    MasVnrArea: int
    LotShape: str
    

def transform(df):
    # renaming columns
    df = df.rename(columns={'FirstFlrSF':'1stFlrSF', 'SecondFlrSF':'2ndFlrSF'})
    # filling nulls
    categoric_type = df.select_dtypes(include=['object'])
    for col in categoric_type:
        df[col] = df[col].fillna(-1)
    # maping categorical to numerical
    df.Neighborhood = df.Neighborhood.map({ 'MeadowV': 0, 'IDOTRR': 1, 'BrDale': 2, 'OldTown': 3, 'Edwards': 4,
                                        'BrkSide': 5,'Sawyer': 6,'Blueste': 7, 'SWISU': 8, 'NAmes': 9,
                                        'NPkVill': 10,'Mitchel': 11, 'SawyerW': 12, 'Gilbert': 13, 'NWAmes': 14,
                                        'Blmngtn': 15,'CollgCr': 16, 'ClearCr': 17, 'Crawfor': 18, 'Veenker': 19,
                                        'Somerst': 20,'Timber': 21, 'StoneBr': 22, 'NoRidge': 23, 'NridgHt': 24})
    df.ExterQual = df.ExterQual.map({'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
    df.KitchenQual = df.KitchenQual.map({'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3})
    df.GarageFinish = df.GarageFinish.map({'Unf': 0, 'RFn': 1, 'Fin': 2})
    df.FireplaceQu = df.FireplaceQu.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
    df.HeatingQC = df.HeatingQC.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
    df.Foundation = df.Foundation.map({'Slab': 0, 'BrkTil': 1, 'Stone': 2, 'CBlock': 3, 'Wood': 4, 'PConc': 5})
    df.MSZoning = df.MSZoning.map({'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4})
    df.Exterior1st = df.Exterior1st.map({'BrkComm': 0, 'AsphShn': 1, 'CBlock': 2, 'AsbShng': 3, 'WdShing': 4,
                                        'Wd Sdng': 5, 'MetalSd': 6, 'Stucco': 7, 'HdBoard': 8, 'BrkFace': 9,
                                        'Plywood': 10,'VinylSd': 11, 'CemntBd': 12, 'Stone': 13, 'ImStucc': 14})
    df.BsmtFinType1 = df.BsmtFinType1.map({'LwQ': 0, 'BLQ': 1, 'Rec': 2, 'ALQ': 3, 'Unf': 4, 'GLQ': 5})
    df.LotShape = df.LotShape.map({'Reg': 0, 'IR1': 1, 'IR3': 2, 'IR2': 3})
    # transforming variables with negative skewness
    df.MSZoning = df.MSZoning ** 3
    df.BsmtFinType1 = df.BsmtFinType1 **3
    # transforming variables with positive skewness
    pos_skewed_cols = ['GrLivArea','ExterQual', '1stFlrSF', '2ndFlrSF', 'MasVnrArea', 'LotArea', 'LotShape', 'BsmtFinSF1']
    for col in pos_skewed_cols:
        df[col] = df[col] ** (1/3)
    return df


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'OK'}


@app.post("/predict")
def predict(data: Data):
    # converting Data object to Dataframe
    dict = data.dict()
    print(dict)
    df = pd.DataFrame(dict.items()).T
    df.columns = df.iloc[0]
    df = df[1:]
    df = transform(df)
    print(df)
    for col in df.columns:
        print({col:df[col]})
    # making predictions
    result = np.expm1(model.predict(df)[0])
    return result

if __name__ == '__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)