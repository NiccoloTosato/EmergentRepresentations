import yaml
from pynvml import *
import numpy as np
import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster
from distributed import Client
from dask import delayed
from pprint import pprint
import time
from ModelConfig import ExperimentConfigurator,expandExperiments
import os
import sys, traceback

def select_device():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    device_process=list()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        device_process.append(len(nvmlDeviceGetComputeRunningProcesses_v3(handle)))
    nvmlShutdown()
    return np.argmin(device_process)

def run_experiment(config):
    config,base_path,increment=config
    time.sleep(increment*10)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(select_device())
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    model_config=ExperimentConfigurator(config,base_path)
    print(model_config,flush=True)
    model_config.run()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trigger a series of experiments.')
    parser.add_argument('--config', dest='config_file',required=True,
                        help='File with experiments list')
    parser.add_argument('--dry-run', dest='dry_run',default=False,
                        help='Perform a dry-run')
    args = parser.parse_args()
    #init the cluster
    cluster = SLURMCluster(cores=8,processes=8,memory="210GB",
                               account="mgm",
                               queue="DGX",
                               walltime="24:00:00",job_extra_directives=['--exclusive','--gpus=8',],
                               job_script_prologue=['source /u/area/ntosato/scratch/pippo/ForwardForward-machiavelli/dask_dgx/bin/activate', ], )
   cluster.scale(8*2)
   client = Client(cluster)
    print(client)
    with open(args.config_file, "r") as stream:
        try:
            #load the YAML config file
            configuration=yaml.safe_load(stream)
            base_path=configuration["ExperimentBasePath"]
            pprint(configuration)
            experiments_list=expandExperiments(configuration)
            if args.dry_run:
                pprint(experiments_list)
            else:
                print("Submitting job..")
                results=[client.submit(run_experiment, (config,base_path,increment)) for increment,config in enumerate(experiments_list)]
                client.gather(results)
                print("Done...")
        except Exception as error:
            print(error)
            traceback.print_exc(file=sys.stdout)

