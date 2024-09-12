import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

def power_func(x,y):
    def power_law(x, a, b):
        return a * np.power(x, b)

    popt, pcov = curve_fit(power_law, x, y)

    a, b = popt

    y_pred = power_law(x, a, b)

    r2 = r2_score(y, y_pred)

    print(f"拟合参数: a = {a}, b = {b}")
    print(f"R² = {r2}")
    mse = mean_squared_error(y,  y_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    rmse = mean_squared_error(y,  y_pred, squared=False)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    mae = mean_absolute_error(y, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    plt.scatter(x, y, label='原始数据')
    plt.plot(x, y_pred, color='red', label='拟合曲线')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.title('幂函数拟合')
    plt.legend()
    plt.show()

if __name__ =='__main__':
    data_path = r'outputs_split_data_bayes_rf/predict_data_result_data240717_no_h2o.xlsx'
    data = pd.read_excel(data_path)
    filtered_df = data[abs(data['H2O (true value)'] - data['H2O (ppm)']) / data['H2O (true value)'] < 0.3]
    x = filtered_df['H2O (ppm)']
    y = filtered_df['H2O (true value)']
    power_func(x, y)