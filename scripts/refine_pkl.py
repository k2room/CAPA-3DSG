import pickle

file_path = "/home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph/dev/421254/42444754/capa_1/cfslam_funcgraph_edges.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data) # list of tuples

arr = []
for i in data:
    a, b, c = list(i)[:3]
    pred = list(i)[3]
    if '_' in pred:
        pred = pred.replace('_', ' ')
    tuple_data = (a, b, c, pred)
    arr.append(tuple_data)

print(arr)

with open(file_path, 'wb') as f:
    pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
