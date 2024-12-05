import numpy as np
import matplotlib.pyplot as plt

x= np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y= np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


one = np.ones((x.shape[0], 1))
xbar = np.concatenate((one, x), axis = 1)

# tính toán trọng số w
A= np.dot(xbar.T, xbar)
b= np.dot(xbar.T, y)
w= np.dot(np.linalg.pinv(A), b)
w_0, w_1 = w[0][0], w[1][0]
print(w)


x0 = np.linspace(145, 185, 2) ## tạo 1 đường thang từ 145 đến 185
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(x.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )