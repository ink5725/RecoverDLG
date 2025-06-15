import pickle

with open("./task/model/logo.pkl", "rb") as f:
    param_shapes = pickle.load(f)

for item in param_shapes:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        name, shape = item
        print(f"Parameter {name}: Shape = {shape}")
    else:
        print("Unexpected item format:", item)

