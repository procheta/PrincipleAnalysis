import pandas as pd

df=pd.read_csv("pref.csv")


principle_dict={}


for i in range(len(df)):
    x=df["principle"][i]
    principle_dict[df["name"][i]]=x




with open("principle.csv","w") as f:
    for key in principle_dict.keys():
        f.write(key)
        f.write("\t")
        f.write(principle_dict[key])
        f.write("\n")


f.close()
