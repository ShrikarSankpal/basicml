import pickle
import os

current_dir = os.getcwd()
pickle_model = os.path.abspath(current_dir+"/resources/model.pkl")
input_file = os.path.abspath(current_dir+"/data/inference/input_file.dat")
output_file = os.path.abspath(current_dir+"/data/inference/output_file.dat")

with open(pickle_model, 'rb') as f:
    loaded_model = pickle.load(f)
    print(loaded_model.get_params())
    with open(input_file,"r") as f:
        numbers = [line.strip().split(",") for line in f]
    predictions = loaded_model.predict(numbers).tolist()
    with open(output_file,"w") as f:
        for result in predictions:
            f.write(str(result)+"\n")
    