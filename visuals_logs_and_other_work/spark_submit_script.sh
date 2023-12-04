#!/bin/bash
#SBATCH --job-name=spark_job_project          # Job name
#SBATCH --nodes=1                     # Number of nodes to request
#SBATCH --ntasks-per-node=12           # Number of processes per node
#SBATCH --mem=20G                      # Memory per node
#SBATCH --time=8:00:00                # Maximum runtime in HH:MM:SS
#SBATCH --account=open 	      # Queue

# Load necessary modules (if required)
module load anaconda3
source activate ds410_f23
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Run PySpark
# Record the start time
start_time=$(date +%s)
spark-submit --deploy-mode client cluster_decision_tree.py

#python cluster_decision_tree.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
