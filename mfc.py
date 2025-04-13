import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import Dense
from tensorflow.keras. optimizers import RMSprop

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection, TomekLinks, CondensedNearestNeighbour

# Read
df= pd.read_csv('machine_failure_cleaned.csv')

# Unused
df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF'])
data = df.to_numpy()

print(data[:5])


X = df.drop(columns=['Machine failure']).values
y = df['Machine failure'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

# 원래 0 이랑 1이 몇개였는지 체크
print("SMOTE 전 클래스 분포:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f" {label}: {count}개")

#다수클래스가 7000개라면 소수클래스를 3500개까지 늘리겠다는 부분
sample_smote = SMOTE(sampling_strategy=0.2,
              random_state=11)

sample_rdundersamp = RandomUnderSampler(sampling_strategy=0.2,
              random_state=11)

# Autos
sample_onesidesel = OneSidedSelection()
sample_tomek = TomekLinks()
sample_cnn =  CondensedNearestNeighbour()

X_train_over_SMOTE, y_train_over_SMOTE = sample_smote.fit_resample(X_train, y_train)
X_train_over_Rand, y_train_over_Rand = sample_rdundersamp.fit_resample(X_train, y_train)
X_train_over_1SideSel, y_train_over_1SideSel = sample_onesidesel.fit_resample(X_train, y_train)
X_train_over_Tomek, y_train_over_Tomek = sample_tomek.fit_resample(X_train, y_train)
X_train_over_CNN, y_train_over_CNN = sample_cnn.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


for lbl, X_train_over, y_train_over in [
    ("SMOTE",  X_train_over_SMOTE, y_train_over_SMOTE),
    ("RandUnderSamp", X_train_over_Rand, y_train_over_Rand),
    ("Tomek", X_train_over_Tomek, y_train_over_Tomek),
    ("CNN", X_train_over_CNN, y_train_over_CNN),
    ("1SideSel", X_train_over_1SideSel, y_train_over_1SideSel),
    ]:
    unique_over, counts_over = np.unique(y_train_over, return_counts=True)
    for label, count in zip(unique_over, counts_over):
        print(f"{lbl} {label}: {count}개")
    pass

    early_stop = EarlyStopping (monitor='val_loss', patience=10, restore_best_weights=True)
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],)))

    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=0.0015735),
                metrics=['accuracy'])

    # Uncommented
    history = model.fit(
            X_train_over
            , y_train_over
            , epochs=100
            # , batch_size = 8 # Think that's too much, commented
            , validation_split=0.2
            , callbacks=[early_stop]
            , verbose= 0
            )

    # What are those
    if (0):
        model_logistic = LogisticRegression(max_iter=1000, random_state=42) # Was named model, renamed it for evade conflict
        model_logistic.fit(X_train_over, y_train_over)
        preds = model_logistic.predict(X_test)# 학습 이후 모델의 성능을 예측
        print(classification_report(y_test, preds))
        pass

    # seems working good
    test_loss, test_acc = model.evaluate(X_test, y_test) #모델 평가/검증
    print(f"Test Accuracy: {test_acc}")
