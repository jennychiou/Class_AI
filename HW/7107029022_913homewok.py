# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
#plt.subplot(2,1,1)
#plt.scatter(x,y)

#訓練樣本50筆
x1 = np.linspace(0,2*np.pi,50)
y1 = np.sin(x1)+np.random.randn(len(x1))/5.0
#plt.subplot(2,1,1)
#plt.scatter(x1,y1)

#訓練樣本100筆
x2 = np.linspace(0,2*np.pi,100)
y2 = np.sin(x2)+np.random.randn(len(x2))/5.0

#x_t = np.linspace(0,2*np.pi,10)
#y_t = np.sin(x1)+np.random.randn(len(x1))/5.0
#plt.subplot(3,1,1)
#plt.scatter(x_t,y_t)

slr = LinearRegression()
x1 = x1.reshape(-1,1) #形成2維
slr.fit(x1,y1)
print("樣本數50的迴歸係數：",slr.coef_)
print("樣本數50的截距：",slr.intercept_)
print('\n')
predicted_y1 = slr.predict(x1)

slr2 = LinearRegression()
x2 = x2.reshape(-1,1) #形成2維
slr2.fit(x2,y2)
print("樣本數100的迴歸係數：",slr2.coef_)
print("樣本數100的截距：",slr2.intercept_)
print('\n')
predicted_y2 = slr2.predict(x2)
#plt.subplot(2,1,2)
#plt.plot(x1,predicted_y1)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15,15))
fig.tight_layout()
print("各種degree的多項式回歸和訓練樣本數50、訓練樣本數100的關係：")

#訓練樣本50筆，degree = 1
poly_features_1 = PolynomialFeatures(degree=1,include_bias=False)
X_poly_1 = poly_features_1.fit_transform(x1)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X_poly_1,y1)
#print(lin_reg_1.intercept_,lin_reg_1.coef_)
X_plot = np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly = poly_features_1.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly,lin_reg_1.coef_.T)+lin_reg_1.intercept_
plt.subplot(4,2,1)
plt.plot(X_plot,y_plot, 'r-', lw=2)
plt.plot(x1,y1,'b.')
plt.title("data=50 / degree=1")


#訓練樣本50筆，degree = 3
poly_features_3 = PolynomialFeatures(degree=3,include_bias=False)
X_poly_3 = poly_features_3.fit_transform(x1)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
#print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot = np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly = poly_features_3.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,3)
plt.plot(X_plot,y_plot, 'r-', lw=2)
plt.plot(x1,y1,'b.')
plt.title("data=50 / degree=3")

#訓練樣本50筆，degree = 5
poly_features_5 = PolynomialFeatures(degree=5,include_bias=False)
X_poly_5 = poly_features_5.fit_transform(x1)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(X_poly_5,y1)
#print(lin_reg_5.intercept_,lin_reg_5.coef_)
X_plot = np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly = poly_features_5.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly,lin_reg_5.coef_.T)+lin_reg_5.intercept_
plt.subplot(4,2,5)
plt.plot(X_plot,y_plot, 'r-', lw=2)
plt.plot(x1,y1,'b.')
plt.title("data=50 / degree=5")

#訓練樣本50筆，degree = 9
poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
X_poly_d = poly_features_d.fit_transform(x1)
lin_reg_d = LinearRegression()
lin_reg_d.fit(X_poly_d, y1)
#print(lin_reg_d.intercept_, lin_reg_d.coef_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_d.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
plt.subplot(4,2,7)
plt.plot(X_plot, y_plot, 'r-', lw=2)
plt.plot(x1, y1, 'b.')
plt.title("data=50 / degree=9")

#訓練樣本100筆，degree = 1
poly_features_1a = PolynomialFeatures(degree=1,include_bias=False)
X_poly_1a = poly_features_1a.fit_transform(x2)
lin_reg_1a = LinearRegression()
lin_reg_1a.fit(X_poly_1a,y2)
#print(lin_reg_1.intercept_,lin_reg_1.coef_)
X_plota = np.linspace(0,6,1000).reshape(-1,1)
X_plot_polya = poly_features_1a.fit_transform(X_plota)
y_plota = np.dot(X_plot_polya,lin_reg_1a.coef_.T)+lin_reg_1a.intercept_
plt.subplot(4,2,2)
plt.plot(X_plota,y_plota, 'r-', lw=2)
plt.plot(x2,y2,'g.')
plt.title("data=100 / degree=1")

#訓練樣本100筆，degree = 3
poly_features_3 = PolynomialFeatures(degree=3,include_bias=False)
X_poly_3 = poly_features_3.fit_transform(x2)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3,y2)
#print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot = np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly = poly_features_3.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,4)
plt.plot(X_plot,y_plot, 'r-', lw=2)
plt.plot(x2,y2,'g.')
plt.title("data=100 / degree=3")

#訓練樣本100筆，degree = 5
poly_features_5 = PolynomialFeatures(degree=5,include_bias=False)
X_poly_5 = poly_features_5.fit_transform(x2)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(X_poly_5,y2)
#print(lin_reg_5.intercept_,lin_reg_5.coef_)
X_plot = np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly = poly_features_5.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly,lin_reg_5.coef_.T)+lin_reg_5.intercept_
plt.subplot(4,2,6)
plt.plot(X_plot,y_plot, 'r-', lw=2)
plt.plot(x2,y2,'g.')
plt.title("data=100 / degree=5")

#訓練樣本100筆，degree = 9
poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
X_poly_d = poly_features_d.fit_transform(x2)
lin_reg_d = LinearRegression()
lin_reg_d.fit(X_poly_d, y2)
#print(lin_reg_d.intercept_, lin_reg_d.coef_)
X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_d.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
plt.subplot(4,2,8)
plt.plot(X_plot, y_plot, 'r-', lw=2)
plt.plot(x2, y2, 'g.')
plt.title("data=100 / degree9")

plt.show()
fig.savefig("7107029022_913homewok.jpg")