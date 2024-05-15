import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from kneed import KneeLocator
import random
import os
import pickle

current_dir = os.getcwd()
input_file=os.path.abspath(current_dir+"/data/training/data.csv")
df = pd.read_csv(input_file)

pickle_file_location = os.path.abspath(current_dir+"/resources/model.pkl")

def calculate_avg_anomaly_score(model, df_anomaly):
    anomaly_scores = model.decision_function(df_anomaly.values)
    avg_anomaly_score = np.mean(anomaly_scores)
    return avg_anomaly_score

def calculate_max_anomaly_score(model, df_anomaly):
    anomaly_scores = model.decision_function(df_anomaly.values)
    max_anomaly_score = max(anomaly_scores)
    return max_anomaly_score

def calculate_adv1_avg_anomaly_score(model, df_anomaly):
    anomaly_scores = model.decision_function(df_anomaly.values)
    avg_anomaly_score = np.e**(-(np.mean(anomaly_scores)-0.1)**2)
    return avg_anomaly_score

iterations = 5
main_start_point=0.0001
main_end_point=0.2
main_intervals=4
step=(main_end_point-main_start_point)/main_intervals
points_in_each_interval=5
knee_points=[]
logging = {}

for iteration in range(iterations):
    print("iteration: ",iteration)
    range_=[]

    start_point=main_start_point
    for _ in range(main_intervals):
        for _ in range(points_in_each_interval):
            range_.append(random.uniform(start_point,start_point+step))
        start_point=start_point+step

    range_.sort()
    neg_range_ = [-i for i in range_]
 
    avg_anomaly_scores=[]
    #max_anomaly_scores=[]
    for parameter in range_:
        model = IsolationForest(contamination=parameter,max_features=df.shape[1],max_samples=0.8)
        model.fit(df.values)
        outliers = model.predict(df.values)
        anomaly_indices = np.where(outliers == -1)[0]
        df_anomaly = df.iloc[anomaly_indices]
        avg_anomaly_scores.append(calculate_avg_anomaly_score(model, df_anomaly))
        #max_anomaly_scores.append(calculate_max_anomaly_score(model, df_anomaly))

    knee_locator = KneeLocator(neg_range_, avg_anomaly_scores, curve='concave', direction='increasing')
    knee_point = knee_locator.knee
    knee_point = None if knee_point==None else -knee_point
    knee_points.append(knee_point)

    logging.update({"{}".format(iteration):
                    {
                    'range_':range_,
                    'neg_range_': neg_range_,
                    'avg_anomaly_scores': avg_anomaly_scores,
                    'knee_point':knee_point                    
                    }})

while None in knee_points:
    knee_points.remove(None)
final_optimized_parameter = np.mean(knee_points)
print("optimized parameter: ",final_optimized_parameter)

# Create and fit the Isolation Forest model
model = IsolationForest(contamination=final_optimized_parameter,max_features=df.shape[1])
model.fit(df.values)

# Predict outliers/anomalies
outliers = model.predict(df.values)

# Anomalies will be labeled as -1, normal points as 1
anomaly_indices = np.where(outliers == -1)[0]

with open(pickle_file_location, 'wb') as f:
    pickle.dump(model, f)