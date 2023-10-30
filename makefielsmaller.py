import pandas as pd

# Define the number of rows for each chunk
chunk_size = 50000
sample_fraction = 1/60
samples = []

# Use the 'chunksize' parameter of 'read_csv' to read the file in chunks
for chunk in pd.read_csv('wildfiredb.csv', chunksize=chunk_size):
    # Sample a fraction of the chunk
    samples.append(chunk.sample(frac=sample_fraction))


# Combine all the sampled chunks
sampled_data = pd.concat(samples, axis=0)

sampled_data.to_csv('wildfire100.csv', index = False)