#!/bin/bash -l
# Sample script for tensorflow job

## Scheduler parameters ##

#BSUB -J LeNet             # job name
#BSUB -o LeNet.%J.stdout   # optional: have output written to specific file
#BSUB -e LeNet.%J.stderr   # optional: have errors written to specific file
# #BSUB -q rb_highend               # optional: use highend nodes w/ Volta GPUs (default: Geforce GPUs)
#BSUB -W 4:00                       # fill in desired wallclock time [hours,]minutes (hours are optional)
# #BSUB -n 1                          # min CPU cores,max CPU cores (max cores is optional)
#BSUB -n 32                          # min CPU cores,max CPU cores (max cores is optional)
#BSUB -M 16384                       # fill in required amount of RAM (in Mbyte)
# #BSUB -R "span[hosts=1]"          # optional: run on single host (if using more than 1 CPU cores)
# #BSUB -R "span[ptile=28]"         # optional: fill in to specify cores per node (max 28)
# #BSUB -P myProject                # optional: fill in cluster project
#BSUB -R "rusage[ngpus_excl_p=10]"   # use 1 GPU (in explusive process mode)

## Job parameters ##

# Anaconda virtualenv to be used
# Create before running the job with e.g.
# conda create -n tensorflow-3.5 python=3.5 tensorflow-gpu
#vEnv=tensorflow-3.5 # (please change)
vEnv=carnd-term1

# Source environment (optional)
#. /fs/applications/lsf/latest/conf/profile.lsf
#. /fs/applications/modules/current/init/bash

# Load modules
module purge
module load conda/4.4.8-readonly cuda/8.0.0 cudnn/8.0_v5.1  
#module load conda/4.3.33-readonly cudnn/8.0_v7.0

# Activate environment
source activate $vEnv

# Run your code here (please change, this is only an example)
jupyter nbconvert --to notebook --execute LeNet-Lab-Solution.ipynb --ExecutePreprocessor.timeout=18000
