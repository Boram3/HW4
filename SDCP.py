import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#파일 불러오기 및 데이터 처리
df = pd.read_csv('sleep_deprivation_dataset_detailed.csv')
df = df.drop(columns=['Participant_ID', 'Stroop_Task_Reaction_Time']) #불필요 데이터 제외
print(df.head())

y_data = df['PVT_Reaction_Time']
df = df.drop(columns=['PVT_Reaction_Time'])

df = pd.get_dummies(df, columns=['Gender'])
gender_columns = [col for col in df.columns if col.startswith('Gender_')]
df[gender_columns] = df[gender_columns].astype(int)
x_data = df

#데이터 분할(train, val, test)
x_train, x_s30, y_train, y_s30 = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_s30, y_s30, test_size=0.5, random_state=42)

#데이터 정규화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#모델 구현
model = Sequential()
model.add(Dense(
    1
    , input_dim=12
    )
) #모델 생성1
model.summary() #모델 생성2

sgd_optimizer = SGD(learning_rate=0.01) # 옵티마이저 및 learning_rat
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)# EarlyStopping, patience회 적발 시 스탑
model.compile(loss='mse', optimizer=sgd_optimizer, metrics=['mae']) #모델 컴파일
history = model.fit(x_train, y_train, epochs=200, batch_size=8, validation_data=(x_val, y_val), callbacks=[early_stopping]) #모델 학습

# Test 평가
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# 예측 값
predictions = model.predict(x_test)
print(predictions[:5])


import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


#데이터 분리
x_temp, x_test, y_temp, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
x_train_kfold, x_val_holdout, y_train_kfold, y_val_holdout = train_test_split(x_temp, y_temp, test_size=0.1765, random_state=4)

#데이터 정규화
scaler = StandardScaler()
x_train_kfold = scaler.fit_transform(x_train_kfold)
x_val_holdout = scaler.transform(x_val_holdout)
x_test = scaler.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=42) #K-fold 기본 설정, 5분할, 섞기O
mae_history = [] #MAE 저장 리스트

#k-fold
for train_idx, val_idx in kfold.split(x_train_kfold):
    x_train, x_val = x_train_kfold[train_idx], x_train_kfold[val_idx]
    y_train, y_val = y_train_kfold.iloc[train_idx], y_train_kfold.iloc[val_idx]  # pandas index을 사용

    # 옵티마이저 정의
    sgd_optimizer = SGD(learning_rate=0.01)

    #모델
    model = Sequential()
    model.add(Dense(1, input_dim=x_train.shape[1])) #모델 생성
    model.compile(loss='mse', optimizer=sgd_optimizer, metrics=['mae']) #모델 컴파일
    model.fit(
        x_train         # 입력값 
        , y_train       # 상응기대값
        , epochs=200    # 200번 반복
        , batch_size=8  # 비교 8회마다 업데이트
        , verbose=0     # 모델 학습 도중 뭐 하나 말하지 말라는 거
        , validation_data=(x_val, y_val)
        ) #모델학습

    #검증 데이터 평가 돌리기
    _, val_mae = model.evaluate(x_val, y_val, verbose=0)

    # 나중에 출력하려고 추가하는 거
    mae_history.append(val_mae)

#K-Fold MAE 출력
print("각 분할 MAE: ", mae_history)
print(f"\n K-Fold 평균 MAE:  {np.mean(mae_history):.4f}")

#성능 확인
val_loss, val_mae = model.evaluate(x_val_holdout, y_val_holdout, verbose=0)
print(f"검증 세트 MAE: {val_mae:.4f}")

#최종? 모델 평가
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
print(f"평가 세트 MAE (test): {test_mae:.4f}")