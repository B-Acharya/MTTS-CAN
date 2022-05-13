import subprocess

# specify the number of jobs

nums_jobs = 1
dataset = "cohface"
gpus = 1
data_path ="/home/bacharya/rPPG_Deeplearning/src/features/PURE/"
save_chekpoints ="/home/bacharya/PURE/checkpoints/"
temp = "Hybrid_CAN"
exp_name = "short"+"-"+dataset+"-"+temp+"-"+str(gpus)
batch_id = 000000

for i in range(nums_jobs):
	print(f"submiting job {i}")
	if i == 0:
		instruction = ["sbatch",f"--output={dataset}-{i}-multistep.log",
			      f"--gpus={gpus}",f"--export=expname={exp_name}-{i},data={data_path},save_dir={save_chekpoints},temporal={temp},initial=1",
			      "-D","pwd",
                              "train-multistep.sh"]
	else:
		instruction = ["sbatch",f"--output={dataset}-{i}-multistep.log",
			      f"--gpus={gpus}",f"--export=expname={exp_name}-{i},data={data_path},save_dir={save_chekpoints},temporal={temp},initial=0",
			      f"--dependency:afterok:{batch_id}"
			      "-D","pwd",
                              "train-multistep.sh"]
	output = subprocess.run(instruction,capture_output=True)
	print(output)
	if output.returncode == 0:
		print(f"submitted {i} job successefully")
		batch_id = int(output.stdout.strip().split()[3].decode('UTF-8'))
