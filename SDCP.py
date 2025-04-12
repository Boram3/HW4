import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfkeras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

#파일 불러오기 및 데이터 처리
df = pd.read_csv('sleep_deprivation_dataset_detailed.csv')

# 불필요 데이터 제외
df = df.drop(columns=['Participant_ID', 'Stroop_Task_Reaction_Time'])
print(df.head())

# 기대예측값 분리
y_data = df['PVT_Reaction_Time']

# 입력값에서 기대예측값 제거
df = df.drop(columns=['PVT_Reaction_Time'])

# Gender를 one-hot encoding해서 bool을 전부 int로 casting하기
df = pd.get_dummies(df, columns=['Gender']).astype(int)

# 이제 저 dataframe을 입력으로 넣는다.
x_data = df

#데이터 분할(train, val, test)
x_train, x_s30, y_train, y_s30 =    train_test_split(x_data, y_data, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test =      train_test_split(x_s30, y_s30, test_size=0.5, random_state=42)

# 데이터 정규화 객체 만들기
# x_train(즉 입력값)용으로 만들어졌으니 재사용이 가능하다.
scaler = StandardScaler()

# 최초정규화를 위해 x_train 순회 돌기
x_train = scaler.fit_transform(x_train)

# 정규화: x_val
x_val = scaler.transform(x_val)

# 정규화: x_test
x_test = scaler.transform(x_test)

# 선형모델
model = Sequential()

# 모델의 출력층 정의
model.add(Dense(
    1               # 출력층: 1
    , input_dim=12  # 입력층: 12
    )
)

# 모델 구조 출력
model.summary() 

# 옵티마이저 미리 설정
sgd_optimizer = SGD(
    learning_rate=0.01
    )

# EarlyStopping, patience회 적발 시 스탑
early_stopping = EarlyStopping(
    monitor='val_loss'              # loss출력값 확인
    , patience=30                   # 30번을 카운트로
    , restore_best_weights=True     # 가중치 예쁘게 나온 거 저장
    )

# 모델 컴파일
model.compile(
    loss='mse'                  # mse loss
    , optimizer=sgd_optimizer   # 미리 만든 sgd_optimizer를 사용
    , metrics=['mae']           # mae로 loss값 확인
    ) 

#모델 학습
history = model.fit(
    x_train                             # 입력값
    , y_train                           # 기대대응출력값
    , epochs=200                        # 반복
    , batch_size=8                      # 업데이트를 위한 비교수
    , validation_data=(x_val, y_val)    # 검증
    , callbacks=[early_stopping]        # early_stopping를 스텝마다 소환
    ) 

# Test 평가
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# 예측 값
predictions = model.predict(x_test)
print(predictions[:5])


## 


# 데이터 분리
x_temp, x_test, y_temp, y_test = train_test_split(
    x_data
    , y_data
    , test_size=0.15
    , random_state=42
    )

# 입력값에서 테스트용 입력 분리하기
x_train_kfold, x_val_holdout, y_train_kfold, y_val_holdout = train_test_split(
    x_temp
    , y_temp
    , test_size=0.1765
    , random_state=4
    )

# 이미 만들어진 sclar transform이 있으니 새로 학습시킬 거 없이 그대로 반영
x_train_kfold = scaler.transform(x_train_kfold)
x_val_holdout = scaler.transform(x_val_holdout)
x_test = scaler.transform(x_test)

# K-fold 기본 설정, 5분할, 섞기O
kfold = KFold(n_splits=5, shuffle=True, random_state=42) 

# MAE 저장 리스트
mae_history = [] 

#모델
model_kfold = Sequential()

#모델 층 추가
model_kfold.add(
    Dense(
        1                               # 출력층 1
        , input_dim=x_train.shape[1]    # 입력층: x_train의 열 수로 설정
        )
    )

# k-fold
for train_idx, val_idx in kfold.split(x_train_kfold):
    x_train, x_val = x_train_kfold[train_idx], x_train_kfold[val_idx]

    # pandas index을 사용
    y_train, y_val = y_train_kfold.iloc[train_idx], y_train_kfold.iloc[val_idx]  

    # 모델을 매번 만드는 거도 귀찮으니 그냥 미리 만든 걸 클론해 보자
    _model = tf.keras.models.clone_model(model_kfold)

    # 옵티마이저 정의
    sgd_optimizer = SGD(learning_rate=0.0092)

    #모델 컴파일
    _model.compile(
        loss='mse'
        , optimizer=sgd_optimizer
        , metrics=['mae']
        )

    # 모델학습
    _model.fit(
        x_train         # 입력값 
        , y_train       # 상응기대값
        , epochs=200    # 200번 반복
        , batch_size=8  # 비교 8회마다 업데이트
        , verbose=0     # 모델 학습 도중 뭐 하나 말하지 말라는 거
        , validation_data=(x_val, y_val)
        )

    #검증 데이터 평가 돌리기
    _, val_mae = _model.evaluate(x_val, y_val, verbose=0)

    # 나중에 출력하려고 추가하는 거
    mae_history.append(val_mae)

# K-Fold MAE 출력
print("각 분할 MAE: ", mae_history)
print(f"K-Fold 평균 MAE:  {np.mean(mae_history):.4f}")

sgd_optimizer = SGD(learning_rate=0.0092)

#모델 컴파일
model_kfold.compile(
    loss='mse'
    , optimizer=sgd_optimizer
    , metrics=['mae']
    )

# 모델학습
model_kfold.fit(
    x_train         # 입력값 
    , y_train       # 상응기대값
    , epochs=200    # 200번 반복
    , batch_size=8  # 비교 8회마다 업데이트
    , verbose=0     # 모델 학습 도중 뭐 하나 말하지 말라는 거
    , validation_data=(x_val, y_val)
    )

# 두 모델에 대해서 성능 테스트해 보기 (kfold와 기존 model 변경)
for name, mod in [("model", model), ("model_kfold", model_kfold)]:
    print(f"성능 확인: {name}")

    # 성능 확인
    val_loss, val_mae = mod.evaluate(x_val_holdout, y_val_holdout, verbose=0)
    print(f"검증 세트 MAE: {val_mae:.4f}")

    # 최종? 모델 평가
    test_loss, test_mae = mod.evaluate(x_test, y_test, verbose=0)
    print(f"평가 세트 MAE (test): {test_mae:.4f}")
    print()

    pass