from pyspark import SparkContext

# Initialize Spark Context
sc = SparkContext("local", "BigDataApp")

# Read the CSV file as text
rdd = sc.textFile("path/to/dataset.csv")

# function to parse each line
def parse_csv(line):
    return line.split(",")

# Apply the function to each line
parsed_rdd = rdd.map(parse_csv)
#If we want to cache the rdd to use it again
#parsed_rdd.persist()

def map_function(row):
    # Suppose you want to count occurrences of a value in the second column
    return (row[1], 1)

def reduce_function(a, b):
    return a + b

# Apply the map function
mapped_rdd = parsed_rdd.map(map_function)

# Reduce by key to count occurrences
reduced_rdd = mapped_rdd.reduceByKey(reduce_function)
rdd = sc.textFile("path/to/dataset.csv", minPartitions=100)
parsed_rdd.persist()

# Collect the results
results = reduced_rdd.collect()

# Process results or save them
for result in results:
    print(result)
