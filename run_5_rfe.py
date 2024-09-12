import os
import warnings

import dill
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


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

    data[features_columns] = standard_scaler(
        data[features_columns], features_scaler_path, mode="train"
    )

    X = data[features_columns]
    y = data[target_column].values

    return X, y, features_columns, target_column


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    data_path = os.path.join(data_dir, "data240717.xlsx")
    outputs_dir = os.path.join(root_dir, f"outputs_rfe")
    os.makedirs(outputs_dir, exist_ok=True)

    X, y, features_columns, target_column = load_data(
        data_path,
        os.path.join(outputs_dir, "features_scaler.pkl"),
    )

    model = RandomForestRegressor(random_state=42)
    rfecv = RFECV(
        estimator=model,
        step=1,
        min_features_to_select=1,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="r2",
        verbose=100,
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    save_pkl(os.path.join(outputs_dir, "rfecv.pkl"), rfecv)

    # rfecv = load_pkl(os.path.join(outputs_dir, "rfecv.pkl"))

    print("features num: ", rfecv.n_features_in_)
    print("best features num: ", rfecv.n_features_)

    rfe_infos = pd.DataFrame(
        {
            "feature_names": rfecv.feature_names_in_,
            "ranking": rfecv.ranking_,
            "support": rfecv.support_,
        }
    )
    rfe_infos.to_csv(os.path.join(outputs_dir, "rfe_infos.csv"), index=False)

    rfe_results = pd.DataFrame(rfecv.cv_results_)
    rfe_results.to_csv(os.path.join(outputs_dir, "rfe_results.csv"), index=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)
    ax.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"],
        marker="",
        linestyle="-",
        linewidth=2,
        label=f"mean_test_score",
    )
    for cv_idx in range(len(rfecv.cv_results_) - 2):
        ax.plot(
            range(1, len(rfecv.cv_results_[f"split{cv_idx}_test_score"]) + 1),
            rfecv.cv_results_[f"split{cv_idx}_test_score"],
            marker="",
            linestyle="-",
            linewidth=1,
            label=f"split{cv_idx}_test_score",
        )
    ax.set_title("RFT特征递归筛选流程图", fontsize=24)
    ax.set_xlabel("特征数量", fontsize=20)
    ax.set_ylabel("R2", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(loc="best", prop={"size": 16})
    plt.savefig(os.path.join(outputs_dir, "rfe_graph.png"))
    # plt.show()
    plt.close()
