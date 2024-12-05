import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('Regression/Salary_dataset.csv')

x = df [['YearsExperience']]
y = df [['Salary']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
#y = W0x + W1
print('chỉ số hồi quy w0 = ', model.coef_)
print('chỉ số chặn w1 = ', model.intercept_)


# Dự đoán trên tập kiểm tra
y_pred = model.predict(x_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse) #o lường sai số trung bình bình phương giữa giá trị dự đoán của mô hình và giá trị thực tế
print("R-squared (R2):", r2) #đo lường tỷ lệ biến thiên của dữ liệu

# So sánh giá trị thực tế và dự đoán
plt.scatter(x_test, y_test, color='blue', label='Thực tế')
plt.plot(x_test, y_pred, color='red', label='Dự đoán')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
