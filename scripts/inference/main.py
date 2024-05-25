import pickle
import os

current_dir = os.getcwd()
pickle_model = os.path.abspath(current_dir+"/resources/model.pkl")
input_file = os.path.abspath(current_dir+"/data/inference/input_file.dat")
#output_file = os.path.abspath(current_dir+"/data/inference/output_file.dat")
output_file = os.path.abspath(current_dir+"/data/inference/combined_csv.csv")


def business_rule_anomaly(row,feature1_bounds,feature2_bounds):
    f1 = row['column1']
    f2 = row['column2']

    if(f1>feature1_bounds[1] or f1<feature1_bounds[0] or f2>feature2_bounds[1] or f2<feature2_bounds[0]):
        return -1
    else:
        return 1
    
def business_rule_nonanomaly(row,feature1_bounds,feature2_bounds):
    f1 = row['column1']
    f2 = row['column2']

    if(f1<=feature1_bounds[1] and f1>=feature1_bounds[0] and f2<=feature2_bounds[1] and f2>=feature2_bounds[0]):
        return -1
    else:
        return 1
    
def populate_final_result(row):
    if(row['business_rule_anomaly']==-1):
        return "Anomaly: Business"
    elif(row['business_rule_nonanomaly']==-1):
        return "NonAnomaly: Business"
    elif(row['predictions']==-1):
        return "Anomaly: ML"
    elif(row['predictions']==1):
        return "NonAnomaly: ML"

with open(pickle_model, 'rb') as f:
    loaded_model = pickle.load(f)
print(loaded_model.get_params())
with open(input_file,"r") as f:
    numbers = [line.strip().split(",") for line in f]

import pandas as pd

#create a daraframe
df = pd.DataFrame(numbers, columns=['column1','column2'])
df['column1'] = df['column1'].astype('float')
df['column2'] = df['column2'].astype('float')

#Apply abs anomaly business logic
feature1_anomaly_bounds=[0,8]
feature2_anomaly_bounds=[0,8]
feature1_nonanomaly_bounds=[0,1.5]
feature2_nonanomaly_bounds=[0,1.5]

df['business_rule_anomaly'] = df.apply(business_rule_anomaly,axis=1,feature1_bounds=feature1_anomaly_bounds, feature2_bounds=feature2_anomaly_bounds)
df['business_rule_nonanomaly'] = df.apply(business_rule_nonanomaly,axis=1,feature1_bounds=feature1_nonanomaly_bounds, feature2_bounds=feature2_nonanomaly_bounds)

# Apply ML logic
predictions = loaded_model.predict(df.loc[:,['column1','column2']].values).tolist()
df['predictions'] = predictions

df['final'] = df.apply(populate_final_result,axis=1)
df.to_csv(output_file, index=False)

# with open(output_file,"w") as f:
#     for result in predictions:
#         f.write(str(result)+"\n")
    