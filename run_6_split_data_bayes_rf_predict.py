import os
import warnings

import dill
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

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


def load_predict_data(predict_data_path, features_scaler_path):
    predict_data = pd.read_excel(predict_data_path)
    # 筛选数据
    predict_data = predict_data[predict_data["Dose (×1015)"] <= 3].copy().reset_index(drop=True)
    predict_data = predict_data[predict_data["LREE-I"] >= 10].copy().reset_index(drop=True)
    predict_data = predict_data[predict_data['P'] <= 10000]

    features_columns = [
        "Age (Ma)",
        #"P",
        #"Ti",
        #"Y",
        "Nb",
        # "La",
        "Ce",
        # "Pr",
        #"Nd",
        "Sm",
        #"Eu",
        #"Gd",
        #"Tb",
        #"Dy",
        #"Ho",
        #"Er",
        #"Tm",
        #"Yb",
        "Lu",
        #"Hf",
        #"Th",
        #"U",
        #"Eu/Eu*",
        #"Ce/Ce*",
        "10000×(Eu/Eu*)/YbN",
        #"1000×(Ce/Nd)N/Y",
        #"U/Yb",
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
        #"LREE-I",
        "Yb/Gd",
        #"Ce/Sm",
        #"Th/U",
    ]
    for col in features_columns:
        predict_data.loc[pd.isna(predict_data[col]), col] = (
            predict_data.loc[~pd.isna(predict_data[col]), col].astype(float).mean()
        )
    # target_column = "H2O (ppm)"

    print(predict_data)

    X_pred = predict_data[features_columns].values
    X_pred = standard_scaler(X_pred, features_scaler_path, mode="test")

    return X_pred, predict_data


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    predict_data_path = os.path.join(data_dir, "Zircon Dataset for Case Study.xlsx")
    outputs_dir = os.path.join(root_dir, f"outputs_split_data_bayes_rf")

    X_pred, predict_data = load_predict_data(
        predict_data_path,
        os.path.join(outputs_dir, "features_scaler.pkl"),
    )

    model = load_pkl(os.path.join(outputs_dir, "model.pkl"))
    y_pred = model.predict(X_pred)
    predict_data["H2O (ppm)"] = y_pred
    print(predict_data)

    predict_data.to_excel(os.path.join(outputs_dir, "predict_data_result_Zircon Dataset for Case Study240719-1.xlsx"), index=False)