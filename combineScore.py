import pandas as pd
import os
import sys
folder=os.listdir("/users/psen/PrincipleAnalysis/EAP/imp_edges_principle"+sys.argv[1]+"/")

edge_dict={}


for file in folder:
	df=pd.read_csv("/users/psen/PrincipleAnalysis/EAP/imp_edges_principle"+sys.argv[1]+"/"+file,sep=",")
	for i in range(len(df)):
		x=0
		if df["Unnamed: 0"][i] in edge_dict.keys():
			x=edge_dict[df["Unnamed: 0"][i]]
		if x!=0:
			edge_dict[df["Unnamed: 0"][i]]=(x+df["score"][i])/2
		else:
			edge_dict[df["Unnamed: 0"][i]]=df["score"][i]


with open("/users/psen/PrincipleAnalysis/final_edge_principal"+sys.argv[1]+".csv","w") as f:
    f.write("edge\tscore\n")
    for key in edge_dict.keys():
        f.write(key)
        f.write("\t")
        f.write(str(edge_dict[key]))
        f.write("\n")


f.close()
