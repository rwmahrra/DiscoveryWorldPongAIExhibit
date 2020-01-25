mkdir autojobs
cd autojobs || exit
mkdir {{ID}}
cd {{ID}} || exit
rm -rf job
git clone {{REMOTE}} job
cd job || exit
git checkout {{BRANCH}}
sbatch run.sh
tail -f pong_job.out