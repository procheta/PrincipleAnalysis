
import pandas as pd

df=pd.read_csv("pp.csv",sep="\t",on_bad_lines='skip',index_col=False)

size=len(df)

print(size)
ar=[]
x=[]
for i in range(len(df)):
    if len(x) !=3:
        try:
            x.append(df["clean"][i]+"###"+df["corrupted"][i]+"###"+str(df["label"][i]))
        except e:
            c=0
            print(e)
    else:
        ar.append(x)
        x=[]


for i in range(len(ar)):
    x=ar[i]
    with open("part/"+str(i)+".csv","w") as f:
        f.write("clean\tcorrupted\tlabel\n")
        print("here")
        for j in range(len(x)):
            s=x[j].split("###")
            for s1 in s:
                f.write(s1)
                f.write("\t")
            f.write("\n")
    f.close()

