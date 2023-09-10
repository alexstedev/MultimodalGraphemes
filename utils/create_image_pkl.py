import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("inputs/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop("image_id", axis=1)
        image_array = df.values
        for j, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"inputs/image_pickles/{image_id}.pkl")
