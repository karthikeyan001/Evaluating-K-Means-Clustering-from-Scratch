import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    c1 = np.random.randn(100,2) + [0,0]
    c2 = np.random.randn(100,2) + [5,5]
    c3 = np.random.randn(100,2) + [-5,5]
    c4 = np.random.randn(100,2) + [5,-5]

    X = np.vstack([c1, c2, c3, c4])
    np.save("features.npy", X)
    pd.DataFrame(X, columns=["x","y"]).to_csv("data.csv", index=False)

if __name__ == "__main__":
    generate_data()
