import pandas as pd

df=pd.read_csv("pref.csv",sep=",")

print(len(df))

print(df.columns)

print(df["convo"][1])
print(df["response_a_text"][1])


principle_dict={}
for i in range(len(df)):
    try:
        prompt= "Consider the following conversation: User: "+ df["convo"][i]+" "+df["principle"][i] + " Options A. Assistant: "+ df["response_a_text"][i] + " Options B. Assistant: " +df["response_b_text"][i]+ " Only answer A or B. The answer is:"
        prompt=prompt.replace("\n","")
        principle=df["principle"][i]
        x=[]
        if principle in principle_dict.keys():
            x= principle_dict[principle]
            #if len(x) > 300:
                #continue
        text=prompt+"###"+df["response"][i]+"###"+principle
        x.append(text)
        principle_dict[principle]=x
    except:
        print("error")



with open("principle.txt","w") as f:
    for key in principle_dict.keys():
        f.write(key)
        f.write("\n")
    f.close()


print(len(principle_dict.keys()))

count=1
for key in principle_dict.keys():
    with open("principles/Principle"+str(count)+".csv","w") as f:
        texts=principle_dict[key]
        f.write("Id\tPrompt\tResponse\tPrinciple\n")
        for i in range(len(texts)):
            x=texts[i]
            st=x.split("###")
            f.write(str(i))
            f.write("\t")
            f.write(st[0])
            f.write("\t")
            f.write(st[1])
            f.write("\t")
            f.write(st[2])
            f.write("\n")
    f.close()
    count=count+1


