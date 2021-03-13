import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data/Train.zip')
trunc_data = data.sample(frac = .1, axis = 0)
targets = trunc_data.pop('SalePrice')
# relevant columns:
# cols = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 
#         'MachineHoursCurrentMeter', 'YearMade', 'Transmission', 
#         'Hydraulics', 'Coupler', 'state', 
#         'Enclosure', 'Tire_Size', 'Engine_Horsepower']

#pared_df = trunc_data[cols]

# encoder = OrdinalEncoder()
# pared_df['state'] = encoder.transform(pared_df['state'])

# X_train, X_test, y_train, y_test = train_test_split(pared_df, targets)

# dummied_train = pd.get_dummies(X_train, dummy_na=True, drop_first=True,dtype='int')

# dummied_test = pd.get_dummies(X_test, dummy_na=True, drop_first=True,dtype='int')

# ss = StandardScaler()

# scaled_dummy_train = ss.fit_transform(dummied_train) #need to clarify which columns are scaled

# scaled_dummy_test = ss.fit_transform(dummied_test)

# models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

# for model in models:
#     model.fit(scaled_dummy_train, y_train) # throws error regarding 'cannot convert *somthing* to float'
#     model.score(dummied_test,y_test)

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return pd.DataFrame(data)

# def train_test_split_helper(df, target_col):
#     X_train, y_train, X_test, y_test = train_test_split(df.drop(target_col), df[target_col]) 
#     return X_train, y_train, X_test, y_test

def make_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    return X_train, X_test, y_train, y_test

def fit_and_score(model_name, X_train, X_test, y_train, y_test):
    model = model_name
    model.fit(X_train,y_train)
    return model.score(y_test, model.predict(X_test))
     
def nan_helper(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            median = np.median(df[column].dropna())
            df[column] = df[column].replace(to_replace = np.nan, value = median)
        else:
            df[column] = df[column].replace(to_replace = np.nan, value = 'None or Unspecified')

    #return df

def dummy_helper(df):
    machine_hours = df['MachineHoursCurrentMeter']
    cols = df.columns
    good_cols = []
    for col in cols:
        if len(df[col].unique())<10:
            good_cols.append(col)
    with_dummies =  pd.get_dummies(df[good_cols], dummy_na = True, drop_first = True, dtype = 'int')

    with_dummies['MachineHoursCurrentMeter'] = machine_hours
    return with_dummies
    # for col in trunc_data.columns:
    #     print(f"{col} : {} values")

def aggregation_function(df, target_col, model_name): 
    df = dummy_helper(df)
    print (type(df))
    df = scale_data(df)
    print (type(df))
    nan_helper(df)
    print (type(df))
    X_train, X_test, y_train, y_test = make_split(df, target_col)
    return fit_and_score(model_name, X_train, X_test, y_train, y_test)

def plot_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=sorted_idx)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()