# Autojob v1.0
# Automatically retrieves code from source control and executes via SLURM
# Written by Xander Neuwirth, January 2020

# Create and descend hierarchy
mkdir autojobs
cd autojobs || exit

# Clear out previous duplicate jobs
rm -rf {{ID}}

# Pull down code
git clone {{REMOTE}} {{ID}}
cd {{ID}} || exit
git checkout {{BRANCH}}

# Schedule with SLURM
sbatch singularity shell --nv /data/containers/msoe-tensorflow-20.07-tf2-py3.sif sh run.sh

# Give slurm time to create outfile
sleep 15

# Hook into updates
tail -f pong_job.out