import os
import warnings

import dill
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
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


def standard_scaler(values, scaler_path, mode="train"):
    if mode == "train":
        scaler = StandardScaler()
        scaler.fit(values)
        save_pkl(scaler_path, scaler)
    else:
        scaler = load_pkl(scaler_path)
    return scaler.transform(values)


def load_data(data_path, features_scaler_path):
    data = pd.read_excel(data_path)

    data = data[data["Dose (×1015)"] <= 3].copy().reset_index(drop=True)
    # data = data[data["(H2O+P)/(REE+Y)"] >= 1].copy().reset_index(drop=True)
    data = data[data["LREE-I"] >= 10].copy().reset_index(drop=True)

    features_columns = [
        "Age (Ma)",
        "P",
        "Ti",
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
        "1000×(Eu/Eu*)/YN",
        "1000×(Eu/Eu*)/YbN",
        "Dy/Yb",
        "U/Yb",
        "△FMQ",
        "T(K)",
        "logfO2",
        "Dose (×1015)",
        "LREE-I",
    ]
    target_column = "H2O (ppm)"
    data = data[features_columns + [target_column]].copy()

    print(data)

    X = data[features_columns].values
    y = data[target_column].values

    X = standard_scaler(X, features_scaler_path, mode="train")

    return X, y, features_columns, target_column


def regression_evaluate(scores, _key, output_path):
    evaluate_result = ""
    evaluate_result += f"评估指标为:"
    if _key == "train":
        evaluate_result += f"\nMAE: {round(-scores[f'train_neg_mean_absolute_error'].mean(), 4)} (+/-) {round(scores[f'train_neg_mean_absolute_error'].std(), 4)}"
        evaluate_result += f"\nMSE: {round(-scores[f'train_neg_mean_squared_error'].mean(), 4)} (+/-) {round(scores[f'train_neg_mean_squared_error'].std(), 4)}"
        evaluate_result += f"\nRMSE: {round(-scores[f'train_neg_root_mean_squared_error'].mean(), 4)} (+/-) {round(scores[f'train_neg_root_mean_squared_error'].std(), 4)}"
        evaluate_result += (
            f"\nR2: {round(scores[f'train_r2'].mean(), 4)} (+/-) {round(scores[f'train_r2'].std(), 4)}"
        )
    else:
        evaluate_result += f"\nMAE: {round(-scores[f'test_neg_mean_absolute_error'].mean(), 4)} (+/-) {round(scores[f'test_neg_mean_absolute_error'].std(), 4)}"
        evaluate_result += f"\nMSE: {round(-scores[f'test_neg_mean_squared_error'].mean(), 4)} (+/-) {round(scores[f'test_neg_mean_squared_error'].std(), 4)}"
        evaluate_result += f"\nRMSE: {round(-scores[f'test_neg_root_mean_squared_error'].mean(), 4)} (+/-) {round(scores[f'test_neg_root_mean_squared_error'].std(), 4)}"
        evaluate_result += (
            f"\nR2: {round(scores[f'test_r2'].mean(), 4)} (+/-) {round(scores[f'test_r2'].std(), 4)}"
        )
    print(evaluate_result)
    save_txt(output_path, evaluate_result)


if __name__ == "__main__":
    for model_name in ["lr", "lasso", "svm", "knn", "ann", "rf", "gbdt", "xgb", "lgb"]:
        print(f"\n\n训练{model_name}模型")

        root_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, "data")
        data_path = os.path.join(data_dir, "data240601-1.xlsx")
        outputs_dir = os.path.join(root_dir, f"outputs_cross_val_{model_name}")
        os.makedirs(outputs_dir, exist_ok=True)

        X, y, features_columns, target_column = load_data(
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

        scoring = ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"]
        scores = cross_validate(
            model,
            X,
            y,
            scoring=scoring,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
            return_train_score=True,
        )

        pd.DataFrame(scores).to_excel(os.path.join(outputs_dir, "scores.xlsx"), index=False)

        regression_evaluate(scores, "train", os.path.join(outputs_dir, "evaluate_train.txt"))
        regression_evaluate(scores, "test", os.path.join(outputs_dir, "evaluate_test.txt"))
