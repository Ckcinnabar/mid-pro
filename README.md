# mid pro
bike numbers and station relationship

import pandas as pd
import chardet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('202209_YouBike2.0票證刷卡資料.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('202209_YouBike2.0票證刷卡資料.csv', encoding=result['encoding'])
df['rent_time'] = pd.to_datetime(df['rent_time'])
df['return_time'] = pd.to_datetime(df['return_time'])
df['rent'] = pd.to_timedelta(df['rent'])
df['infodate'] = pd.to_datetime(df['infodate'])

df['hour'] = df['rent_time'].dt.hour
df['weekday'] = df['rent_time'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

hourly_counts = df.groupby(['rent_station', 'hour'])['rent'].count().reset_index(name='hourly_count')

def calculate_no_bike(df, threshold=0.2):
    # 計算每個車站每小時的平均租借次數
    avg_hourly_counts = df.groupby(['rent_station', 'hour'])['rent'].count().mean(level=0)
    
    # 對於每個車站，如果平均租借次數低於閾值（例如，0.2），則預測在未來一小時無車
    no_bike = avg_hourly_counts < threshold
    
    return no_bike

# 使用函式計算每個車站在未來一小時是否無車
hourly_counts['no_bike'] = calculate_no_bike(df)

# 使用 pandas 的 get_dummies 函式對 rent_station 進行獨熱編碼
rent_station_encoded = pd.get_dummies(hourly_counts['rent_station'], prefix='rent_station')

# 將獨熱編碼後的車站名稱與 hourly_counts DataFrame 結合
hourly_counts_encoded = pd.concat([hourly_counts.drop(columns=['rent_station']), rent_station_encoded], axis=1)

hourly_counts_encoded = hourly_counts_encoded.fillna(0)

X = hourly_counts_encoded.drop(columns=['no_bike'])
y = hourly_counts_encoded['no_bike']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# 預測並評估模型
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

