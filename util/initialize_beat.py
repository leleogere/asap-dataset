""" 
This script is useful if you want to perform audio beat tracking.
It takes the asap_annotations.json file and the metadata.csv file and creates the .beats files for the asap dataset.
"""

from pathlib import Path
import os
import json
import pandas as pd
import soundfile as sf
from collections import OrderedDict
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from collections import Counter
from sklearn.preprocessing import LabelEncoder

BASE_PATH = Path("./")


with open(Path(BASE_PATH,'asap_annotations.json')) as json_file:
    json_data = json.load(json_file)
print("Total performances", len(json_data.keys()))

df = pd.read_csv(Path(BASE_PATH,"metadata.csv"))
# add a columns which is the concatenation of title and composer columns, which we will use to identify unique pieces
df["title_composer"] = df["title"] + "_" + df["composer"]

print("Total performances", len(df), "Unique pieces", len(df["title_composer"].unique()))

# take only the tracks with audio
df = df[df["audio_performance"].notna()]

json_data = {k:json_data[k] for k in df["midi_performance"].values}
print("with audio", len(json_data.keys()), "Unique pieces", len(df["title_composer"].unique()))

# remove pieces that have bR in the beat annotations
no_br_pieces = {k:v for k,v in json_data.items() if not any([beat_type == "bR" for time,beat_type in v["performance_beats_type"].items()])}
df = df[df["midi_performance"].isin(no_br_pieces.keys())]
print("No bR", len(no_br_pieces), "Unique pieces", len(df["title_composer"].unique()) )

# get the remaining unique time signatures
ts = set()
for k,v in no_br_pieces.items():
    # print(v["perf_time_signatures"])
    # print(list(v["perf_time_signatures"].values()))
    for time_signature in v["perf_time_signatures"].values():
        ts.add(time_signature[1])
print("Time signatures numerators", ts)


def anndict_to_beats(anndict, output_path, number_of_beats):
    # write a tsv file with beats.times as first column and beats.positions as second column
    ordered_ann = OrderedDict(sorted(anndict.items(), key = lambda x: float(x[0])))
    with open(output_path, 'w') as f:
        # find how many beats before the first downbeat
        first_downbeat_index = [i for i, x in enumerate(ordered_ann.values()) if x == "db"][0]
        upmeasure = [number_of_beats - i for i in range(first_downbeat_index)][::-1]
        # find and write the beats
        counter = 1
        for i, (time,type) in enumerate(ordered_ann.items()):
            if type == "db":
                counter = 1
                pos = 1
            elif type == "b":
                if i < first_downbeat_index:
                    pos = upmeasure[0]
                    upmeasure.pop(0)
                elif i>= first_downbeat_index:
                    pos = counter
                else:
                    raise ValueError("Something went wrong")
            else:
                raise ValueError("Something went wrong")
            counter += 1
            f.write(str(time) + '\t' + str(pos) + '\n')

# save audio and beat annotations in tsv .beats format
for k,v in no_br_pieces.items():
    audio_performance_path = df[df["midi_performance"]==k]["audio_performance"].values[0]
    path_flattened = audio_performance_path.replace('/','_')[:-4]
    ann_path = str(Path("asap_beat","annotations","beats",f"{path_flattened}.beats"))
    audio_path = str(Path("asap_beat","audio",f"{path_flattened}.wav"))
    print("Processing", audio_performance_path, "to", audio_path, ann_path)
    if not Path(audio_path).exists():
        # # copy the file to the new location and rename it, with cp command
        os.makedirs(Path(audio_path).parent, exist_ok=True)
        os.system(f"cp {audio_performance_path} {audio_path}")
        # audio, sr = sf.read(audio_performance_path)
        # sf.write(audio_path, audio, sr)
    if not Path(ann_path).exists():
        # save beat annotations
        anndict_to_beats(v["performance_beats_type"], ann_path, list(v["perf_time_signatures"].values())[0][1])

# this will be used to balance across composers
# make this categorical integers
le = LabelEncoder()
groups = le.fit_transform(df["title_composer"].to_list())
y = le.fit_transform(df["composer"].to_list())

# create 7 splits so that there are no duplicates and the number of pieces is almost the same
# 7 splits because we want to have a validation set of 15% of the unique pieces
kf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
splits = list(kf.split(df["midi_performance"],y,groups))

# take the split with the ratio closest to 0.15
ratios = [len(val_idx)/len(train_idx) for train_idx, val_idx in splits]
print("Ratio for the produced splits", ratios)
min_ratio = min(ratios, key=lambda x: abs(x-0.15))
idx = ratios.index(min_ratio)
print("Split number", idx, " was selected")
train_idx, val_idx = splits[idx]
print("Len train", len(train_idx), "Len val", len(val_idx), "Ratio", len(val_idx)/len(train_idx))

# buil the train and val dataframes
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
trainval_df = pd.concat([train_df,val_df])
print("Len unique pieces train", len(train_df["title_composer"].unique()), "Len unique pieces val", len(val_df["title_composer"].unique()))
print("Composer train", Counter(train_df["composer"].tolist()))
print("Composer val", Counter(val_df["composer"].tolist()))

# add a column in the dataframe with the flattened path
trainval_df["piece"] = trainval_df["audio_performance"].apply(lambda x: x.replace('/','_')[:-4])
# add a column with the split
trainval_df["split"] = "train"
trainval_df.loc[val_df.index, "split"] = "val"
# save the split in a csv file with two columns: name and split, ordered by name
trainval_df = trainval_df.sort_values(by="piece")
trainval_df[["piece","split"]].to_csv(Path(BASE_PATH,"asap_beat","asap_split.csv"), index=False)

# check if there is an audio file for each row
for i, row in trainval_df.iterrows():
    if not Path(BASE_PATH,"asap_beat","audio",f"{row['piece']}.wav").exists():
        print("Missing piece", row['piece'])
print("Done")

