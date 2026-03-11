import os
import urllib.request
import h5py
import numpy as np
import argparse

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"File {filename} already exists. Skipping download.")

def convert_hdf5_to_binary(hdf5_path, out_dir):
    print(f"Opening {hdf5_path}...")
    f = h5py.File(hdf5_path, 'r')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Dump train (base vectors)
    print("Dumping base vectors...")
    train = np.array(f['train'])
    print(f"Shape: {train.shape}, Type: {train.dtype}")
    with open(os.path.join(out_dir, "base.bin"), "wb") as out_f:
        # Write dimensions as 32-bit integers at the start to make C++ reading easier
        out_f.write(np.array([train.shape[0], train.shape[1]], dtype=np.int32).tobytes())
        out_f.write(train.astype(np.float32).tobytes())

    # Dump test (query vectors)
    print("Dumping query vectors...")
    test = np.array(f['test'])
    print(f"Shape: {test.shape}, Type: {test.dtype}")
    with open(os.path.join(out_dir, "query.bin"), "wb") as out_f:
        out_f.write(np.array([test.shape[0], test.shape[1]], dtype=np.int32).tobytes())
        out_f.write(test.astype(np.float32).tobytes())

    # Dump neighbors (ground truth)
    print("Dumping ground truth neighbors...")
    neighbors = np.array(f['neighbors'])
    print(f"Shape: {neighbors.shape}, Type: {neighbors.dtype}")
    with open(os.path.join(out_dir, "ground_truth.bin"), "wb") as out_f:
        out_f.write(np.array([neighbors.shape[0], neighbors.shape[1]], dtype=np.int32).tobytes())
        out_f.write(neighbors.astype(np.int32).tobytes())
        
    f.close()
    print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean", help="Dataset name e.g., sift-128-euclidean")
    args = parser.parse_args()
    
    dataset = args.dataset
    url = f"http://ann-benchmarks.com/{dataset}.hdf5"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    hdf5_file = os.path.join(data_dir, f"{dataset}.hdf5")
    download_file(url, hdf5_file)
    convert_hdf5_to_binary(hdf5_file, os.path.join(data_dir, dataset))

if __name__ == "__main__":
    main()
