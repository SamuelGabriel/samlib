from submitit import *


def get_executor(folder="log_test",timeout_min=60, slurm_partition="testdlc_gpu-rtx2080", slurm_gres='gpu:1',
                 slurm_setup=['export MKL_THREADING_LAYER=GNU'], **kwargs):
    # executor is the submission interface (logs are dumped in the folder)
    executor = AutoExecutor(folder=folder)
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=timeout_min, slurm_partition=slurm_partition, slurm_gres=slurm_gres,
                               slurm_setup=slurm_setup, **kwargs)
    return executor


def print_job_out(job, only_stdout=False, only_stderr=False):
    assert not (only_stderr and only_stdout)
    if not only_stderr:
        print("STD OUT")
        print(job.stdout().replace('\\n', '\n'))
    if not only_stdout:
        print("STD ERR")
        print(job.stderr().replace('\\n', '\n'))
