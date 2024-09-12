import os
import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(data_path):

    data = pd.read_excel(data_path)
    print(data.columns)

    data = data[data["Dose (×1015)"] <= 3].copy().reset_index(drop=True)
    data = data[data["LREE-I"] >= 10].copy().reset_index(drop=True)

    features_columns = [
        "P",
        "Ti",
        "Y",
        "Nb",
        "La",
        "Ce",
        "Pr",
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

    return data, features_columns, X, y


def plot_boxplot(data, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=100)
    sns.boxplot(data, ax=ax)
    ax.set_title("箱型图可视化", fontsize=24)
    ax.set_xlabel("Feature", fontsize=20)
    ax.set_ylabel("Values", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def plot_corr(data, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 18), dpi=100)
    sns.heatmap(
        data.corr().round(2),
        annot=True,
        cmap="RdBu",
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("相关性可视化", fontsize=24)
    ax.set_xlabel("Feature", fontsize=20)
    ax.set_ylabel("Feature", fontsize=20)
    ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def pca_fit(X, output_path):
    model = PCA(random_state=42)
    model.fit(X)
    features_variance = pd.DataFrame(
        {
            "Features": features_columns,
            "Variance": model.explained_variance_,
            "Variance Ratio": model.explained_variance_ratio_,
        }
    )
    print(features_variance)
    features_variance.to_excel(output_path, index=False)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    data_path = os.path.join(data_dir, "data240601-1.xlsx")
    outputs_dir = os.path.join(root_dir, "outputs_data_analysis")
    os.makedirs(outputs_dir, exist_ok=True)

    data, features_columns, X, y = load_data(data_path)

    plot_boxplot(data, os.path.join(outputs_dir, "boxplot.png"))
    plot_corr(data, os.path.join(outputs_dir, "corr.png"))
    pca_fit(X, os.path.join(outputs_dir, "features_variance.xlsx"))
