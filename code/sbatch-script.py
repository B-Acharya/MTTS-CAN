import subprocess

# specify the number of jobs

nums_jobs = 4
dataset = "cohface"
gpus = 4
data_path ="/home/bacharya/rPPG_Deeplearning/src/features/PURE/"
save_chekpoints ="/home/bacharya/PURE/checkpoints/"
temp = "Hybrid_CAN"
exp_name = "short"+"-"+dataset+"-"+temp+"-"+str(gpus)
batch_id = 000000

for i in range(nums_jobs):
	print(f"submiting job {i}")
	if i == 0:
		instruction = ["sbatch",f"--output={dataset}-{i}-multistep.log",
					   f"--gpus={gpus}",f"--expname={exp_name}-{i}",
					   f"--data={data_path}",
					   f"--save_dir={save_chekpoints}",
					   f"--temporal={temp}",
					   f"--initial=0",
					   f"train-multistep.sh"]
	else:
		instruction = ["sbatch",f"--output={dataset}-{i}-multistep.log",
					   f"--gpus={gpus}",f"--expname={exp_name}-{i}",
					   f"--data={data_path}",
					   f"--save_dir={save_chekpoints}",
					   f"--temporal={temp}",
					   f"--initial=0",
					   f"--dependency=afterok:{batch_id}",
					   f"train-multistep.sh"]
	output = subprocess.run(instruction,capture_output=True)
	print(output)
	if output.returncode == 0:
		print("submitted job successefully")
		batch_id = int(output.stdout.strip().split()[3].decode('UTF-8'))
