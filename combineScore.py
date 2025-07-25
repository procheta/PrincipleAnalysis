import pandas as pd
import os

folder=os.listdir("/Users/prochetasen/Downloads/imp_edges_principle12/")

edge_dict={}


for file in folder:
	df=pd.read_csv("/Users/prochetasen/Downloads/imp_edges_principle12/"+file,sep=",")
	for i in range(len(df)):
		x=0
		if df["Unnamed: 0"][i] in edge_dict.keys():
			x=edge_dict[df["Unnamed: 0"][i]]
		if x!=0:
			edge_dict[df["Unnamed: 0"][i]]=(x+df["score"][i])/2
		else:
			edge_dict[df["Unnamed: 0"][i]]=df["score"][i]


with open("final_edge_principal12.csv","w") as f:
	for key in edge_dict.keys():
		f.write(key)
		f.write("\t")
		f.write(str(edge_dict[key]))
		f.write("\n")


f.close()
