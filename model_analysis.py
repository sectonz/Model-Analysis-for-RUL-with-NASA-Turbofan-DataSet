import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
pd.set_option("display.max_rows", None)

# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names# read data
train = pd.read_csv('train_FD001.txt',sep='\s+', header=None, names=col_names)
test = pd.read_csv('../test_FD001.txt',sep='\s+', header=None, names=col_names)
y_test = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])# Train data contains all features (Unit Number + setting parameters & sensor parameters)
# Test data contains all features (Unit Number + setting parameters & sensor parameters)
# Y_test contains RUL for the test data.

train.head()

train['unit_nr'].unique()

train.describe()

print("Dimensões de y_test:",{y_test.shape})
print("Dimensões de test:",{train.shape})
print("Dimensões de train:",{test.shape})

train=train.drop('setting_3',axis=1)

def add_remaining_useful_life(df):
    #Pega a quantidade de ciclos para cada motor
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame
    
train = add_remaining_useful_life(train)
train[sensor_names+['RUL']].head()

df_max_rul = train[['unit_nr', 'RUL']].groupby('unit_nr').max().reset_index()
df_max_rul['RUL'].hist(bins=15, figsize=(15,7))
plt.xlabel('RUL')
plt.ylabel('frequency')
# plt.show()

def plot_sensor(sensor_name):
    plt.figure(figsize=(13,5))
    for i in train['unit_nr'].unique():
        if (i % 10 == 0):  # only plot every 10th unit_nr
            plt.plot('RUL', sensor_name, 
                     data=train[train['unit_nr']==i])
    plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()
    
# for sensor_name in sensor_names:
#     plot_sensor(sensor_name)

plt.figure(figsize=(25,18))
sns.heatmap(train.corr(),annot=True ,cmap='Reds')
# plt.show()

cor=train.corr()
#Selecting highly correlated features
train_relevant_features = cor[abs(cor['RUL'])>=0.5]

train_relevant_features['RUL']

list_relevant_features=train_relevant_features.index
list_relevant_features=list_relevant_features[1:]
list_relevant_features

# Now we will keep onlt these imprtant features in both train & test dataset.
train=train[list_relevant_features]

# train & y_train
# Calculated RUL variable is our Target variable.
y_train=train['RUL']
X_train=train.drop(['RUL'],axis=1)
X_train.head(5)

# Test data set , keeping only train columns/features.
X_test=test[X_train.columns]
X_test.head(5)

y_train= y_train.clip(upper=125)

# funcao para avaliar os modelos
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE: {}, R2: {}'.format(label, rmse, variance))
    return rmse,variance;

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)

# y_train.shape

#   DataFrame Principal
Results = pd.DataFrame(columns=['Modelo', 'RMSE-Train', 'R2-Train', 'RMSE-Test', 'R2-Test'])

def add_results_to_dataframe(results_df, model_name, rmse_train, r2_train, rmse_test, r2_test):
    # Criar um novo DataFrame com os resultados do modelo atual
    new_result = pd.DataFrame({
        'Modelo': [model_name],
        'RMSE-Train': [rmse_train],
        'R2-Train': [r2_train],
        'RMSE-Test': [rmse_test],
        'R2-Test': [r2_test]
    })
    # Concatenar o novo resultado ao DataFrame global
    return pd.concat([results_df, new_result], ignore_index=True)


#       REGRESSAO LINEAR

print("\n\t\tRegressao Linear:\n")
# create and fit model
lm = LinearRegression()
lm.fit(X_train1, y_train)

# predict and evaluate
y_hat_train1 = lm.predict(X_train1)
RMSE_Train,R2_Train=evaluate(y_train, y_hat_train1,'train')
print("\n")

y_hat_test1 = lm.predict(X_test1)
test['Predicted_RUL'] = y_hat_test1
last_cycle_per_unit = test.groupby('unit_nr')['time_cycles'].idxmax()
last_predictions = test.loc[last_cycle_per_unit, 'Predicted_RUL'].values

RMSE_Test,R2_Test=evaluate(y_test.values, last_predictions,'test')

Results = add_results_to_dataframe(Results, 'Regressão Linear', RMSE_Train, R2_Train, RMSE_Test, R2_Test)


#           RANDOM FOREST

print("\n\t\tRandom Forest:\n")
rf = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=6, min_samples_leaf=5)
rf.fit(X_train1, y_train)
y_hat_train1 = rf.predict(X_train1)
# Evaluating on Train Data Set
RMSE_Train,R2_Train=evaluate(y_train, y_hat_train1, 'train')
print("\n")

# Evaluating on Test Data Set
y_hat_test1 = rf.predict(X_test1)
test['Predicted_RUL'] = y_hat_test1
last_cycle_per_unit = test.groupby('unit_nr')['time_cycles'].idxmax()
last_predictions = test.loc[last_cycle_per_unit, 'Predicted_RUL'].values

RMSE_Test,R2_Test=evaluate(y_test.values, last_predictions)

Results = add_results_to_dataframe(Results, 'Random Forest', RMSE_Train, R2_Train, RMSE_Test, R2_Test)



#           XGBOOST

print("\n\t\tXgBoost:\n")
import xgboost as xg
xgb_r = xg.XGBRegressor(objective ='reg:squarederror',
                  n_estimators = 10, seed = 123)
xgb_r.fit(X_train1, y_train)

# Evaluating on Train Data Set
y_hat_train1 = xgb_r.predict(X_train1)
RMSE_Train,R2_Train=evaluate(y_train, y_hat_train1, 'train')
print("\n")

# Evaluating on Test Data Set
y_hat_test1 = xgb_r.predict(X_test1)
test['Predicted_RUL'] = y_hat_test1
last_cycle_per_unit = test.groupby('unit_nr')['time_cycles'].idxmax()
last_predictions = test.loc[last_cycle_per_unit, 'Predicted_RUL'].values

RMSE_Test,R2_Test=evaluate(y_test.values, last_predictions)

Results = add_results_to_dataframe(Results, 'XGBoost', RMSE_Train, R2_Train, RMSE_Test, R2_Test)



#           TENSORFLOW 
print("\n\t\tTensorflow:\n")

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))
ann.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
ann.fit(X_train1, y_train, batch_size = 32, epochs = 75)

# print("y_train:",{y_train.shape})
# print("y_test:",{y_test.shape})

# Adicionar as previsões ao DataFrame original do conjunto de teste
y_hat_test1 = ann.predict(X_test1)
test['Predicted_RUL'] = y_hat_test1

# #        MEDIA DAS PREVISOES PARA CADA CICLO
# print("\n")
# print("Media da previsao da RUL de cada ciclo de cada motor")
# # Agrupar por unidade e exibir as previsões
# predictions_per_unit_mean = test[['unit_nr', 'Predicted_RUL']].groupby('unit_nr').mean().reset_index()
# predictions_per_unit_mean['True_RUL'] = y_test.values
# # Exibir as previsões
# print(predictions_per_unit_mean)


#        ULTIMA PREVISAO (ultimo ciclo)
print("\n")
print("Previsao da RUL para cada motor")
# Identificar o último ciclo registrado para cada motor
last_cycle_per_unit = test.groupby('unit_nr')['time_cycles'].idxmax()
# Selecionar as previsões correspondentes ao último ciclo
predictions_per_unit = test.loc[last_cycle_per_unit, ['unit_nr']]
predictions_per_unit['Predicted_RUL'] = y_hat_test1[last_cycle_per_unit.index]
predictions_per_unit['True_RUL'] = y_test.values
# Exibir as previsões
print(predictions_per_unit)


#EVALUATING

# Selecionar as previsões correspondentes ao último ciclo
last_predictions = test.loc[last_cycle_per_unit, 'Predicted_RUL'].values

# Evaluating on Train Data Set
y_hat_train1 = ann.predict(X_train1)
print("\n")
RMSE_Train,R2_Train=evaluate(y_train, y_hat_train1, 'train')

# Evaluating on Test Data Set
RMSE_Test,R2_Test=evaluate(y_test.values, last_predictions,'test') ##problema aq

Results = add_results_to_dataframe(Results, 'ANN', RMSE_Train, R2_Train, RMSE_Test, R2_Test)



print("\n\nResultados Finais de Todos os Modelos:\n")
print(Results)
