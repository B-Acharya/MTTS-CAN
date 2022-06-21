import subprocess
import time

# specify the number of jobs

nums_jobs = 40
dataset = "cohface"
#dataset = "PURE"
gpus = 4
data_path ="/work/data/bacharya/cohface/"
#data_path ="/home/bacharya/rPPG_Deeplearning/src/features/PURE/"
save_chekpoints ="/home/bacharya/cohface/checkpoints/"
#save_chekpoints ="/home/bacharya/PURE/checkpoints/"
temp = "Hybrid_CAN"
#temp = "TS_CAN"
#temp = "CAN_3D"
exp_name = "short"+"-"+dataset+"-"+temp+"-"+str(gpus)
batch_id = 000000
intermediate_path = "/home/bacharya/cohface/models"
#intermediate_path = "/home/bacharya/PURE/models"

for i in range(nums_jobs):
	print(f"submiting job {i}")
	if i == 0:
		instruction = ["sbatch",f"--output=Split-{dataset}-{temp}-{i}-multistep.log",
				f"--error=Split-{dataset}-{i}-errors.log",
			      f"--gpus={gpus}",f"--export=exp_name={exp_name}-{i},data={data_path},save_dir={save_chekpoints},temporal={temp},initial=1,inter={intermediate_path}",
                              "train-multistep.sh"]
	else:
		instruction = ["sbatch",f"--output=Split-{dataset}-{temp}-{i}-multistep.log",
				f"--error=Split-{dataset}-{i}-errors.log",
			      f"--gpus={gpus}",f"--export=exp_name={exp_name}-{i},data={data_path},save_dir={save_chekpoints},temporal={temp},initial=0,inter={intermediate_path}",
			      f"--dependency=afterok:{batch_id}",
                              "train-multistep.sh"]
	output = subprocess.run(instruction,capture_output=True)
	print(output)
	if output.returncode == 0:
		print(f"submitted {i} job successefully")
		batch_id = int(output.stdout.strip().split()[3].decode('UTF-8'))
	else:
		print(f"Job {i} failled")
		while(output.stderr.split(b"\n")[0].split()[-1] == b"QOSMaxSubmitJobPerUserLimit"):
			print("waiting for a job to finish")
			time.sleep(60)
			output = subprocess.run(instruction,capture_output=True)
			if output.returncode == 0:
				print(f"submitted {i} job successefully")
				batch_id = int(output.stdout.strip().split()[3].decode('UTF-8'))
				break
