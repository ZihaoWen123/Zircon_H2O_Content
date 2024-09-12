import os
import warnings

import dill
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def save_txt(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def save_pkl(filepath, data):
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


#def standard_scaler(values, scaler_path, mode="train"):
#     if mode == "train":
#         scaler = StandardScaler()
#         scaler.fit(values)
#         save_pkl(scaler_path, scaler)
#     else:
#         scaler = load_pkl(scaler_path)
#     return scaler.transform(values)


def standard_scaler(values, scaler_path, mode="train"):
    if mode == "train":
        min_value = np.min(values)
        save_pkl(scaler_path, min_value)
    else:
        min_value = load_pkl(scaler_path)

    return np.log(values - min_value + 1)


def load_data(data_path, features_scaler_path):
    data = pd.read_excel(data_path)

    data = data[data["Dose (×1015)"] <= 3].copy().reset_index(drop=True)
    data = data[data["LREE-I"] >= 10].copy().reset_index(drop=True)

    features_columns = [
        "Age (Ma)",
        "P",
        #"Ti",
        "Y",
        "Nb",
        # "La",
        "Ce",
        # "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Th",
        "U",
        "Eu/Eu*",
        "Ce/Ce*",
        "10000×(Eu/Eu*)/YbN",
        "1000×(Ce/Nd)N/Y",
        "U/Yb",
        #"△FMQ",
        #"T(K)",
        #"Ce/(U/Ti)^0.5",
        # "La/C1 Chond",
        # "Ce/C1 Chond",
        # "Pr/C1 Chond",
        # "Nd/C1 Chond",
        # "Sm/C1 Chond",
        # "Eu/C1 Chond",
        # "Gd/C1 Chond",
        # "Tb/C1 Chond",
        # "Dy/C1 Chond",
        # "Ho/C1 Chond",
        # "Er/C1 Chond",
        # "Tm/C1 Chond",
        # "Yb/C1 Chond",
        # "Lu/C1 Chond",
        #"Ce/U",
        #"U/Ti",
        #"Th (initial)",
        #"U (initial)",
        "Dose (×1015)",
        "LREE-I",
        "Yb/Gd",
        "Ce/Sm",
        "Th/U",
    ]
    target_column = "H2O (ppm)"
    data = data[features_columns + [target_column]].copy()

    print(data)

    X = data[features_columns].values
    y = data[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = standard_scaler(X_train, features_scaler_path, mode="train")
    X_test = standard_scaler(X_test, features_scaler_path, mode="test")

    return X_train, y_train, X_test, y_test, features_columns, target_column


def regression_evaluate(y_true, y_pred, output_path):
    # 回归模型的性能指标
    evaluate_result = ""
    evaluate_result += f"评估指标为:"
    evaluate_result += f"\nMAE: {round(mean_absolute_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nMSE: {round(mean_squared_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nRMSE: {round(pow(mean_squared_error(y_true, y_pred), 0.5), 4)}"
    evaluate_result += f"\nR2: {round(r2_score(y_true, y_pred), 4)}"
    print(evaluate_result)
    save_txt(output_path, evaluate_result)


def comparison_visualization(y_real, y_pred, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)

    ax.plot(y_real, marker="", linestyle="-", linewidth=2, label="真实值")
    ax.plot(y_pred, marker="", linestyle="-", linewidth=2, label="预测值")
    ax.set_title("真实值和预测值对比图", fontsize=24)
    ax.set_xlabel("数据点", fontsize=20)
    ax.set_ylabel("值", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(loc="best", prop={"size": 20})

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def residual_visualization(y_real, y_pred, output_path, fitting=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 12), dpi=100)

    ax.text(
        min(y_real),
        max(y_real),
        f"$MAE={round(mean_absolute_error(y_real, y_pred), 4)}$"
        f"\n$MSE={round(mean_squared_error(y_real, y_pred), 4)}$"
        f"\n$RMSE={round(pow(mean_squared_error(y_real, y_pred), 0.5), 4)}$"
        f"\n$R^2={round(r2_score(y_real, y_pred), 4)}$",
        verticalalignment="top",
        fontdict={"size": 16, "color": "k"},
    )
    ax.scatter(y_real, y_pred, c="none", marker="o", edgecolors="k")  # 散点图
    if fitting:
        from sklearn.linear_model import LinearRegression

        fitting_model = LinearRegression()
        fitting_model.fit([[item] for item in y_real], y_pred)
        ax.plot(
            [min(y_real), max(y_real)],
            [
                fitting_model.predict([[min(y_real)]]).item(),
                fitting_model.predict([[max(y_real)]]).item(),
            ],
            linewidth=2,
            linestyle="--",
            color="r",
            label="拟合曲线",
        )
    ax.plot(
        [min(y_real), max(y_real)],
        [min(y_real), max(y_real)],
        linewidth=2,
        linestyle="-",
        color="r",
        label="参考曲线",
    )
    ax.set_title("真实值和预测值残差图", fontsize=24)
    ax.set_xlabel("真实值", fontsize=20)
    ax.set_ylabel("预测值", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(loc="lower right", prop={"size": 20})

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def importances_visualization(features_name, feature_importances, values_output_path, png_output_path):
    features_name, feature_importances = zip(
        *sorted(zip(features_name, feature_importances), key=lambda x: x[1], reverse=True)
    )

    pd.DataFrame({"features_name": features_name, "feature_importances": feature_importances}).to_csv(
        values_output_path, index=False
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)

    ax.bar(features_name, feature_importances, width=0.5)
    for x, y in zip(range(len(features_name)), [round(item, 3) for item in feature_importances]):
        ax.text(x=x, y=y, s=y, ha="center", va="bottom", fontdict={"size": 12, "color": "k"})

    ax.set_xticks(range(len(features_name)), features_name, rotation=0)
    ax.set_title("特征重要程度分析", fontsize=24)
    ax.set_xlabel("特征", fontsize=20)
    ax.set_ylabel("重要程度", fontsize=20)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(png_output_path)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    for model_name in ["lr", "lasso", "svm", "knn", "ann", "rf", "gbdt", "xgb", "lgb"]:
        print(f"\n\n训练{model_name}模型")

        root_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, "data")
        data_path = os.path.join(data_dir, "data240717.xlsx")
        outputs_dir = os.path.join(root_dir, f"outputs_split_data_{model_name}")
        os.makedirs(outputs_dir, exist_ok=True)

        X_train, y_train, X_test, y_test, features_columns, target_column = load_data(
            data_path,
            os.path.join(outputs_dir, "features_scaler.pkl"),
        )

        if model_name == "lr":
            model = LinearRegression(n_jobs=-1)
        elif model_name == "lasso":
            model = Lasso(random_state=42)
        elif model_name == "svm":
            model = SVR()
        elif model_name == "knn":
            model = KNeighborsRegressor(n_jobs=-1)
        elif model_name == "ann":
            model = MLPRegressor(random_state=42)
        elif model_name == "rf":
            model = RandomForestRegressor(n_jobs=-1, random_state=42)
        elif model_name == "gbdt":
            model = GradientBoostingRegressor(random_state=42)
        elif model_name == "xgb":
            model = XGBRegressor(n_jobs=-1, random_state=42)
        elif model_name == "lgb":
            model = LGBMRegressor(n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"模型名称{model_name}错误!")

        model.fit(X_train, y_train)
        save_pkl(os.path.join(outputs_dir, "model.pkl"), model)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        regression_evaluate(
            y_train, y_train_pred, os.path.join(outputs_dir, "evaluate_train.txt")
        )
        regression_evaluate(y_test, y_test_pred, os.path.join(outputs_dir, "evaluate_test.txt"))
        comparison_visualization(
            y_test, y_test_pred, os.path.join(outputs_dir, "comparison.png")
        )
        residual_visualization(
            y_test, y_test_pred, os.path.join(outputs_dir, "residual.png"), fitting=True
        )

        if model_name in ["rf", "gbdt", "xgb", "lgb"]:
            importances_visualization(
                features_columns,
                model.feature_importances_,
                os.path.join(outputs_dir, "importances.csv"),
                os.path.join(outputs_dir, "importances.png"),
            )

        if model_name in ["lr"]:
            pd.DataFrame(
                {"features_name": features_columns + ["intercept"], "coef_": model.coef_.tolist() + [model.intercept_]}
            ).to_csv(
                os.path.join(outputs_dir, "coef.csv"), index=False
            ) 
