import os
import pandas as pd
from data_process import dti_process

def files():
    folder = "../park4\\NITRC_PD_DATA"
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    rows = []
    for i in subfolders:
        files = [ os.path.join(i,file) for file in os.listdir(i) if "1000" in file]
        rows.append(files)
    return rows

def pd_group(string):
    if string == "Control":
        return 0
    return 1

def processed_data():
    csv = pd.read_csv("PDClinicalData.csv")
    rows = files()
    # print(rows)
    data = []
    for i in rows:
        row = i[0].split("\\")[-2]
        i.append(pd_group(csv.loc[csv['Subject'] == row]['Group'].to_list()[0]))
        i.append(csv.loc[csv['Subject'] == row]['HADS_depression'].to_list()[0])
        data.append(i)
        # print(i)
    return data

def calculate_dtic():
    data = processed_data()
    final_data = []
    for row in data:
        nii_file,bval_file,bvec_file = row[0],row[1],row[2]
        fa,md,rd,ad = dti_process(nii_file,bval_file,bvec_file)
        pd_status, depression = row[3],row[4]
        final_row = [[fa,md,rd,ad],[pd_status,depression]]
        final_data.append(final_row)
    return final_data

# print(calculate_dtic())

# print(len(rows))
# print(csv)
