import os
import pandas as pd

DATASET_PATH = "datasets/Audio_Lyrics_Dataset/Audio"

def create_csv(dataset_path, output="dataset.csv"):
    data = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".mp3"):

                file_path = os.path.join(root, file)

                # genre = parent folder name
                genre = os.path.basename(root)

                data.append([file_path, genre])

    df = pd.DataFrame(data, columns=["file_path", "genre"])
    df.to_csv(output, index=False)

    print(f"CSV created → {output}")
    print(f"Total samples → {len(df)}")

create_csv(DATASET_PATH)