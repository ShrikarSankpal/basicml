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

print("Parameter Optimizer Functions Loading...")
def calculate_avg_anomaly_score(model, df_anomaly):
    anomaly_scores = model.decision_function(df_anomaly.values)
    avg_anomaly_score = np.mean(anomaly_scores)
    return avg_anomaly_score

def calculate_separation_distances(df_to_fit, df_to_calculate):
    n_neighbors = 3
    result=0 if (len(df_to_fit)==0 or len(df_to_calculate)==0) else 1

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',n_jobs=-1).fit(df_to_fit.values)
    nbrs.kneighbors(df_to_calculate.values)
    distances = nbrs.kneighbors(df_to_calculate.values)[0]
    required_distances = np.min(distances, axis=1)
    result = np.mean(required_distances)
    return result
print("Parameter Optimizer Functions Loaded.")

print("Creating Parameter Grid...")
iterations = 5
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
        #range_.append(random.uniform(start_point,start_point+step))
        range_.append(start_point+small_step)
        start_point=start_point+small_step

range_.sort()
neg_range_ = [-i for i in range_]

training_logs = {}
print("Parameter Grid Population Completed.")

print("Starting Iterations Through Random Numbers...")
random_numbers = [1]
for random_number in random_numbers:
    print("random_number: ",random_number)
    separation_distances=[]
    print("Generating Models For Parameter Grid...")
    for parameter in range_:
        model = IsolationForest(contamination=parameter,max_features=df.shape[1],max_samples=0.8, random_state=random_number)
        model.fit(df.values)
        outliers = model.predict(df.values)
        anomaly_indices = np.where(outliers == -1)[0]
        df_anomaly = df.iloc[anomaly_indices]
        df_nonanomaly = df.drop(anomaly_indices)
        separation_distances.append(calculate_separation_distances(df_to_fit=df_nonanomaly, df_to_calculate=df_anomaly))
    print("Updating Traininig Logs...")
    training_logs.update({"{}".format(random_number):
                    {
                    'range_':range_,
                    'neg_range_': neg_range_,
                    'scores': separation_distances                   
                    }})
    print("Training Logs Updated.")

print("Dumping Training Logs...")
with open(training_logs_file, 'w') as f:
    json.dump(training_logs, f)
print("Done.")

print("Dumping Granular Training Logs...")
# Each iteration separately
for i in random_numbers:
    range_ = training_logs.get("{}".format(i)).get("range_")
    neg_range = [-i for i in range_]
    scores = training_logs.get("{}".format(i)).get("scores")
    pd.DataFrame({
        'parameter':range_,
        'scores':scores      
        }).to_csv(training_data_path+"/random_number-{}.csv".format(i),index=False)
print("Done.")

print("Reading Training Logs...")
with open(training_logs_file, 'r') as f:
    training_logs = json.load(f)

print("Creating Plots...")
print("Parameter Grid...")
for i in [random_numbers[0]]:
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
for i in random_numbers:
    range_ = training_logs.get("{}".format(i)).get("range_")
    scores = training_logs.get("{}".format(i)).get("scores")
    plt.figure(figsize=(20,5))
    plt.plot(range_,scores,label="random_number-{}".format(i))
    plt.xlabel("parameter")
    plt.ylabel("optimizer function output")
    plt.title("optimizer function output VS parameter")
    plt.xticks(range_, rotation='vertical')
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(plots_path+"/training-parameter_and_output-iteration-{}.png".format(i))
    plt.close()

print("Negative Parameter Grid And Optimizer Function Output...")
for i in random_numbers:
    range_ = training_logs.get("{}".format(i)).get("range_")
    neg_range = [-i for i in range_]
    scores = training_logs.get("{}".format(i)).get("scores")
    plt.figure(figsize=(20,5))
    plt.plot(neg_range,scores,label="random_number-{}".format(i))
    plt.xlabel("negative parameter")
    plt.ylabel("optimizer function output")
    plt.title("optimizer function output VS negative parameter")
    plt.xticks(neg_range,rotation='vertical')
    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(plots_path+"/negative_training-parameter_and_output-iteration-{}.png".format(i))
    plt.close()

print("Done.")

print("Getting Optimized Parameter Through Algorithm...")
knee_points=[]
for i in random_numbers:
    range_ = training_logs.get("{}".format(i)).get("range_")
    scores = training_logs.get("{}".format(i)).get("scores")
    knee_locator = KneeLocator(range_, scores, curve='concave', direction='increasing')
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
print("Thank You!")

print("Training Final Model...")
# Create and fit the Isolation Forest model
model = IsolationForest(contamination=final_optimized_parameter,max_features=df.shape[1])
model.fit(df.values)
print("Final Model Training Complete.")

print("Saving Model...")
with open(pickle_file_location, 'wb') as f:
    pickle.dump(model, f)
print("Done.")