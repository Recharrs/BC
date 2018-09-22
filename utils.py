import pickle

def get_batch(data, batch_size, step):
    num_batch = len(data) // batch_size
    batch_idx = step % num_batch 
    
    start = (batch_idx) * batch_size
    end = (batch_idx + 1) * batch_size

    return zip(*data[start:end])

def load_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data