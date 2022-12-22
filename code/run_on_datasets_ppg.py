from model_evaluation import run_on_dataset
import subprocess
from config import leaveOneOut_PURE


if __name__ == "__main__":

	subprocess.run(['python','model_evaluaiton.py',
					'--database_name', leaveOneOut_PURE.database_name,
					'--save_dir', leaveOneOut_PURE.save_dir,
					'--data_dir', leaveOneOut_PURE.data_dir,
					'--split_path', leaveOneOut_PURE.split_path,
					'--test_name', leaveOneOut_PURE.test_name,
					'--model_name', leaveOneOut_PURE.model_name,
					'--model_path', leaveOneOut_PURE.model_path])
