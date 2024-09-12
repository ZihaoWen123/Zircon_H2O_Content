import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    original_data_path = os.path.join(data_dir, "Dataset - Zircon H2O content - 240531.xlsx")
    data_path = os.path.join(data_dir, "data240717.xlsx")

    data = pd.read_excel(original_data_path, sheet_name="Zircon Composition")

    print(data.isnull().sum())
    # H2O (ppm)              8
    # Ti                   870
    # Nb                   667
    # Hf                   309
    # Th                   273
    # U                    273
    # U/Yb                 273
    # δ18O (‰)               8
    # Dose (×1015)         273

    data = data.dropna(subset=["H2O (ppm)"]).reset_index(drop=True)
    # data = data.drop(columns=['δ18O (‰)'])
    for col in ["Ti", "Nb", "Hf", "Th", "U"]:
        data.loc[pd.isna(data[col]), col] = (
            data.loc[~pd.isna(data[col]), col].astype(float).mean()
        )
    # data = data[data['P']<=10000]


    data["1000×(Ce/Nd)N/Y"] = [np.nan] * len(data)
    data["10000×(Eu/Eu*)/YbN"] = [np.nan] * len(data)
    data["Eu/Eu*"] = [np.nan] * len(data)
    data["Ce/Ce*"] = [np.nan] * len(data)
    data["Yb/Gd"] = [np.nan] * len(data)
    data["Ce/Sm"] = [np.nan] * len(data)
    data["Th/U"] = [np.nan] * len(data)
    data["U/Yb"] = [np.nan] * len(data)
    data["Dose (×1015)"] = [np.nan] * len(data)
    # data["(H2O+P)/(REE+Y)"] = [np.nan] * len(data)
    data["LREE-I"] = [np.nan] * len(data)
    data["△FMQ"] = [np.nan] * len(data)
    data["T(K)"] = [np.nan] * len(data)
    data["Th (initial)"] = [np.nan] * len(data)
    data["U (initial)"] = [np.nan] * len(data)
    data["Ce/(U/Ti)^0.5"] = [np.nan] * len(data)
    data["La/C1 Chond"] = [np.nan] * len(data)
    data["Ce/C1 Chond"] = [np.nan] * len(data)
    data["Pr/C1 Chond"] = [np.nan] * len(data)
    data["Nd/C1 Chond"] = [np.nan] * len(data)
    data["Sm/C1 Chond"] = [np.nan] * len(data)
    data["Eu/C1 Chond"] = [np.nan] * len(data)
    data["Gd/C1 Chond"] = [np.nan] * len(data)
    data["Tb/C1 Chond"] = [np.nan] * len(data)
    data["Dy/C1 Chond"] = [np.nan] * len(data)
    data["Ho/C1 Chond"] = [np.nan] * len(data)
    data["Er/C1 Chond"] = [np.nan] * len(data)
    data["Tm/C1 Chond"] = [np.nan] * len(data)
    data["Yb/C1 Chond"] = [np.nan] * len(data)
    data["Lu/C1 Chond"] = [np.nan] * len(data)
    data["Ce/U"] = [np.nan] * len(data)
    data["U/Ti"] = [np.nan] * len(data)



    print(data)
    data.to_excel(data_path, index=False)
