import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from kneed import KneeLocator
import os
import json
from sklearn.neighbors import NearestNeighbors
import pickle

current_dir = os.getcwd()
print("Current Directory: ",current_dir)

input_file=os.path.abspath(current_dir+"/data/training/data.csv")
training_logs_file=os.path.abspath(current_dir+"/data/training/training_logs.json")
training_data_path=os.path.abspath(current_dir+"/data/training/")
plots_path=os.path.abspath(current_dir+"/plots/")
pickle_file_location = os.path.abspath(current_dir+"/resources/model.pkl")

print("Reading Data...")
df = pd.read_csv(input_file)
print("Data Reading Complete.")
print("Data Shape: ",df.shape)
print("Data Columns: ",df.columns)

print("Defining business rules...")
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


feature1_max=df['column1'].max()
feature1_min=df['column1'].min()
feature2_max=df['column2'].max()
feature2_min=df['column2'].min()

print("fetching business rules requirements...")
feature1_anomaly_bounds=[0,8]
feature2_anomaly_bounds=[0,8]
feature1_nonanomaly_bounds=[0,1.5]
feature2_nonanomaly_bounds=[0,1.5]

print("Applying Business rules...")
df['business_rule_anomaly'] = df.apply(business_rule_anomaly,axis=1,feature1_bounds=feature1_anomaly_bounds, feature2_bounds=feature2_anomaly_bounds)
df['business_rule_nonanomaly'] = df.apply(business_rule_nonanomaly,axis=1,feature1_bounds=feature1_nonanomaly_bounds, feature2_bounds=feature2_nonanomaly_bounds)
print("done.")

print("Making data ready for ML")
df_with_business_results = df.copy()
df = df.drop(df.loc[df['business_rule_anomaly']==-1].index,axis=0)
print(df.shape)
df = df.reset_index(drop=True)
df = df.drop(columns=['business_rule_anomaly','business_rule_nonanomaly'])
print("done.")


print("Parameter Optimizer Functions Loading...")

def calculate_separation_distances(df_to_fit, df_to_calculate):
    n_neighbors = 3
    result=0 if (len(df_to_fit)==0 or len(df_to_calculate)==0) else 1

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',n_jobs=-1).fit(df_to_fit.values)
    nbrs.kneighbors(df_to_calculate.values)
    distances = nbrs.kneighbors(df_to_calculate.values)[0]
    required_distances = np.min(distances, axis=1)
    result = np.min(required_distances)
    return result
print("Parameter Optimizer Functions Loaded.")

print("Creating Parameter Grid...")
main_start_point=0.0001
main_end_point=0.2
main_intervals=5
step=(main_end_point-main_start_point)/main_intervals
points_in_each_interval=10
small_step=step/points_in_each_interval

range_=[]
start_point=main_start_point
for _ in range(main_intervals):
    for _ in range(points_in_each_interval):
        range_.append(start_point+small_step)
        start_point=start_point+small_step

range_.sort()
print("Parameter Grid Population Completed.")

training_logs = {}

print("Starting Iterations Through Random Numbers...")
random_numbers = [1]

for i, random_number in zip(range(len(random_numbers)),random_numbers):
    print("Iteration: ",i)
    separation_distances=[]
    for parameter in range_:
        print("Training Model for Contaminantion: ",parameter)
        model = IsolationForest(contamination=parameter,max_features=df.shape[1],max_samples=0.8, random_state=random_number,)
        model.fit(df.values)
        outliers = model.predict(df.values)
        anomaly_indices = np.where(outliers == -1)[0]
        df_anomaly = df.iloc[anomaly_indices]
        df_nonanomaly = df.drop(anomaly_indices)
        separation_distance = calculate_separation_distances(df_to_fit=df_nonanomaly, df_to_calculate=df_anomaly)
        separation_distances.append(separation_distance)
        separation_distances = [item for item in separation_distances if item!=None]
        separation_distances = separation_distances + [None]*(len(range_)-len(separation_distances))

    training_logs.update({"{}".format(i):
                    {
                    'range_':range_,
                    'scores': separation_distances                   
                    }})
    print("Training Logs Updated.")

print("Dumping Training Logs...")
with open(training_logs_file, 'w') as f:
    json.dump(training_logs, f)
print("Done.")

print("Dumping Granular Training Logs...")
# Each iteration separately
for i in range(len(random_numbers)):
    range_ = training_logs.get("{}".format(i)).get("range_")
    neg_range = [-i for i in range_]
    scores = training_logs.get("{}".format(i)).get("scores")
    pd.DataFrame({
        'parameter':range_,
        'scores':scores      
        }).to_csv(training_data_path+"/iteration-{}.csv".format(i),index=False)
print("Done.")

print("Reading Training Logs...")
with open(training_logs_file, 'r') as f:
    training_logs = json.load(f)

print("Creating Plots...")
print("Parameter Grid...")
for i in range(len(random_numbers)):
    range_ = training_logs.get("{}".format(i)).get("range_")
    neg_range = [-i for i in range_]
    avg_anomaly_scores = training_logs.get("{}".format(i)).get("avg_anomaly_scores")
    plt.figure(figsize=(20,5))
    plt.plot(range(len(range_)),range_)
    plt.xlabel("index")
    plt.ylabel("paramter")
    plt.title("parameter range")
    plt.xticks(range(len(range_)))
    plt.grid()
    plt.savefig(plots_path+"/training-parameter_generation.png")
    plt.close()

print("Parameter Grid And Optimizer Function Output...")
# Look for a sudden drop, a knee. Choose the point at the edge of the sudden drop
for i in range(len(random_numbers)):
    range_ = training_logs.get("{}".format(i)).get("range_")
    scores = training_logs.get("{}".format(i)).get("scores")
    plt.figure(figsize=(20,5))
    plt.plot(range_,scores,label="iteration-{}".format(i))
    plt.xlabel("parameter")
    plt.ylabel("optimizer function output")
    plt.title("optimizer function output VS parameter")
    plt.xticks(range_, rotation='vertical')
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(plots_path+"/training-parameter_and_output-iteration-{}.png".format(i))
    plt.close()

print("Done.")

print("Getting Optimized Parameter Through Algorithm...")
knee_points=[]
for i in range(len(random_numbers)):
    range_ = training_logs.get("{}".format(i)).get("range_")
    scores = training_logs.get("{}".format(i)).get("scores")
    knee_locator = KneeLocator(range_, scores, curve='convex', direction='decreasing')
    knee_point = knee_locator.knee
    knee_point = None if knee_point==None else knee_point
    knee_points.append(knee_point)

while None in knee_points:
    knee_points.remove(None)
final_optimized_parameter = np.mean(knee_points)

print("\n-----\n")
print("The final_optimized_parameter chosen by algorithm: ",final_optimized_parameter)
print("Please note that there is no guarantee that the above value is the best.")
print("Please view plots and training logs to select the best value for the parameter")
final_optimized_parameter = input("Please enter the parameter value for the final ML model: ")
final_optimized_parameter = float(final_optimized_parameter)
print("You have selected: ",final_optimized_parameter)
print("Thank You!")

print("Training Final Model...")
# Create and fit the Isolation Forest model
model = IsolationForest(contamination=final_optimized_parameter,max_features=df.shape[1], random_state=random_numbers[0])
model.fit(df.values)
print("Final Model Training Complete.")

print("Saving Model...")
with open(pickle_file_location, 'wb') as f:
    pickle.dump(model, f)
print("Done.")