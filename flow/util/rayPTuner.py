import argparse
import hashlib
import json
import os
from os.path import abspath
import re
import sys
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run
import numpy as np
import time

import uuid
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import Experiment
from ray.tune import get_trial_name
from ray import get, put


import nevergrad as ng
from ax.service.ax_client import AxClient

ORFS_URL = 'https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts'
AUTOTUNER_BEST = 'autotuner-best.json'
FASTROUTE_TCL = 'fastroute.tcl'
CONSTRAINTS_SDC = 'constraint.sdc'

DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# experiment = f'test-tune-{DATE}-{uuid.uuid4()}'
# platform = ""
# design = ""
# verbose = 0
# LAST_STEP = ""
# EVAL_FUNC = None
# CONTINUE_FROM = ""
# LOCAL_DIR = ""
# INSTALL_PATH = ""
# FR_ORIGINAL = ""
# SDC_ORIGINAL = ""


class AutoTunerBase(tune.Trainable):
    '''
    AutoTuner base class for experiments.
    '''

    def setup(self, config, data=None):
        '''
        Setup current experiment step.
        '''
        # We create the following directory structure:
        #      1/     2/         3/       4/                5/   6/
        # <repo>/<logs>/<platform>/<design>/<experiment>-DATE/<id>/<cwd>
        repo_dir = os.getcwd() + '/../' * 6
        self.repo_dir = abspath(repo_dir)
        self.parameters = parse_config(config, path=os.getcwd())
        # print(f'Main Repo: {repo_dir}, Repo Dir: {self.repo_dir}, params: {self.parameters}\n')
        self.step_ = 0

        self.experiment = data.get("exp")
        self.last = data.get("last")
        self.eval = data.get("eval")
        self.prev = data.get("cont")
        
        print(f'Running on pid {os.getpid()} with experiment {self.experiment}, last step {self.last}\n')

    def step(self):
        '''
        Run step experiment and compute its score.
        '''
        metrics_file, flow_variant = openroad(self.repo_dir, self.parameters, self.experiment, self.last, self.prev)
        with open(f'../{self.trial_id}', 'w') as file:
            file.write(flow_variant)
            file.flush()
            file.close()
        self.step_ += 1

        with open(metrics_file) as file:
            data = json.load(file) 

        score = self.eval(data)

        return {"minimum": score}

    def evaluate(self, metrics):
        '''
        User-defined evaluation function.
        It can change in any form to minimize the score (return value).
        Default evaluation function optimizes effective clock period.
        '''
        error = 'ERR' in metrics.values()
        not_found = 'N/A' in metrics.values()
        if error or not_found:
            return (99999999999) * (self.step_ / 100)**(-1)
        gamma = (metrics['clk_period'] - metrics['worst_slack']) / 10
        score = metrics['clk_period'] - metrics['worst_slack']
        score = score * (self.step_ / 100)**(-1) + gamma * metrics['num_drc']
        return score

    @classmethod
    def read_metrics(cls, file_name):
        '''
        Collects metrics to evaluate the user-defined objective function.
        '''
        with open(file_name) as file:
            data = json.load(file)
        clk_period = 9999999
        worst_slack = 'ERR'
        wirelength = 'ERR'
        num_drc = 'ERR'
        total_power = 'ERR'
        core_util = 'ERR'
        final_util = 'ERR'
        for stage, value in data.items():
            if stage == 'constraints' and len(value['clocks__details']) > 0:
                clk_period = float(value['clocks__details'][0].split()[1])
            if stage == 'floorplan' \
                    and 'design__instance__utilization' in value:
                core_util = value['design__instance__utilization']
            if stage == 'detailedroute' and 'route__drc_errors' in value:
                num_drc = value['route__drc_errors']
            if stage == 'detailedroute' and 'route__wirelength' in value:
                wirelength = value['route__wirelength']
            if stage == 'finish' and 'timing__setup__ws' in value:
                worst_slack = value['timing__setup__ws']
            if stage == 'finish' and 'power__total' in value:
                total_power = value['power__total']
            if stage == 'finish' and 'design__instance__utilization' in value:
                final_util = value['design__instance__utilization']
        ret = {
            "clk_period": clk_period,
            "worst_slack": worst_slack,
            "wirelength": wirelength,
            "num_drc": num_drc,
            "total_power": total_power,
            "core_util": core_util,
            "final_util": final_util
        }
        return ret


def read_config(file_name):
    '''
    Please consider inclusive, exclusive
    Most type uses [min, max)
    But, Quantization makes the upper bound inclusive.
    e.g., qrandint and qlograndint uses [min, max]
    step value is used for quantized type (e.g., quniform). Otherwise, write 0.
    When min==max, it means the constant value
    '''
    def read(path):
        with open(abspath(path), 'r') as file:
            ret = file.read()
        return ret

    def read_sweep(this):
        return [*this['minmax'], this['step']]

    def read_tune(this):
        min_, max_ = this['minmax']
        if min_ == max_:
            # Returning a choice of a single element allow pbt algorithm to
            # work. pbt does not accept single values as tunable.
            return tune.choice([min_])
        if this['type'] == 'int':
            # if min_ == 0 and args.algorithm == 'nevergrad':
            #     print('[WARNING TUN-0011] NevergradSearch may not work '
            #           'with lowerbound value 0.')
            if this['step'] == 1:
                return tune.randint(min_, max_)
            return tune.qrandint(min_, max_, this['step'])
        if this['type'] == 'float':
            if this['step'] == 0:
                return tune.uniform(min_, max_)
            return tune.quniform(min_, max_, this['step'])
        return None



    with open(file_name) as file:
        data = json.load(file)
    sdc_file = ''
    fr_file = ''
    config = dict()
    for key, value in data.items():
        if key == '_SDC_FILE_PATH' and value != '':
            if sdc_file != '':
                print('[WARNING TUN-0004] Overwriting SDC base file.')
            sdc_file = read(f'{os.path.dirname(file_name)}/{value}')
            continue
        if key == '_FR_FILE_PATH' and value != '':
            if fr_file != '':
                print('[WARNING TUN-0005] Overwriting FastRoute base file.')
            fr_file = read(f'{os.path.dirname(file_name)}/{value}')
            continue
        if not isinstance(value, dict):
            config[key] = value
        else:
            config[key] = read_tune(value)
        # elif args.mode == 'sweep':
        #     config[key] = read_sweep(value)
        # elif args.mode == 'tune' and args.algorithm != 'ax':
        #     config[key] = read_tune(value)
        # elif args.mode == 'tune' and args.algorithm == 'ax':
        #     config.append(read_tune_ax(key, value))
    # Copy back to global variables
    return config, sdc_file, fr_file


def parse_config(config, path=os.getcwd()):
    '''
    Parse configuration received from tune into make variables.
    '''
    options = ''
    sdc = {}
    fast_route = {}
    for key, value in config.items():
        # Keys that begin with underscore need special handling.
        if key.startswith('_'):
            # Variables to be injected into fastroute.tcl
            if key.startswith('_FR_'):
                fast_route[key.replace('_FR_', '', 1)] = value
            # Variables to be injected into constraints.sdc
            elif key.startswith('_SDC_'):
                sdc[key.replace('_SDC_', '', 1)] = value
            # Special substitution cases
            elif key == "_PINS_DISTANCE":
                options += f' PLACE_PINS_ARGS="-min_distance {value}"'
            elif key == "_SYNTH_FLATTEN" and value == 1:
                options += ' SYNTH_ARGS=""'
        # Default case is VAR=VALUE
        else:
            options += f' {key}={value}'
    if bool(sdc):
        write_sdc(sdc, path)
        options += f' SDC_FILE={path}/{CONSTRAINTS_SDC}'
    if bool(fast_route):
        write_fast_route(fast_route, path)
        options += f' FASTROUTE_TCL={path}/{FASTROUTE_TCL}'
    return options


def write_sdc(variables, path):
    '''
    Create a SDC file with parameters for current tuning iteration.
    '''
    # TODO: handle case where the reference file does not exist
    new_file = SDC_ORIGINAL
    for key, value in variables.items():
        if key == 'CLK_PERIOD':
            new_file = re.sub(r'-period [0-9\.]+ (.*)',
                              f'-period {value} \\1',
                              new_file)
            new_file = re.sub(r'-waveform [{}\s0-9\.]+[\s|\n]',
                              '',
                              new_file)
        elif key == 'UNCERTAINTY':
            if new_file.find('set uncertainty') != -1:
                new_file = re.sub(r'set uncertainty .*\n(.*)',
                                  f'set uncertainty {value}\n\\1',
                                  new_file)
            else:
                new_file += f'\nset uncertainty {value}\n'
        elif key == "IO_DELAY":
            if new_file.find('set io_delay') != -1:
                new_file = re.sub(r'set io_delay .*\n(.*)',
                                  f'set io_delay {value}\n\\1',
                                  new_file)
            else:
                new_file += f'\nset io_delay {value}\n'
    file_name = path + f'/{CONSTRAINTS_SDC}'
    with open(file_name, 'w') as file:
        file.write(new_file)
    return file_name


def write_fast_route(variables, path):
    '''
    Create a FastRoute Tcl file with parameters for current tuning iteration.
    '''
    # TODO: handle case where the reference file does not exist
    layer_cmd = 'set_global_routing_layer_adjustment'
    new_file = FR_ORIGINAL
    for key, value in variables.items():
        if key.startswith('LAYER_ADJUST'):
            layer = key.lstrip('LAYER_ADJUST')
            # If there is no suffix (i.e., layer name) apply adjust to all
            # layers.
            if layer == '':
                new_file += '\nset_global_routing_layer_adjustment'
                new_file += ' $::env(MIN_ROUTING_LAYER)'
                new_file += '-$::env(MAX_ROUTING_LAYER)'
                new_file += f' {value}'
            elif re.search(f'{layer_cmd}.*{layer}', new_file):
                new_file = re.sub(f'({layer_cmd}.*{layer}).*\n(.*)',
                                  f'\\1 {value}\n\\2',
                                  new_file)
            else:
                new_file += f'\n{layer_cmd} {layer} {value}\n'
        elif key == 'GR_SEED':
            new_file += f'\nset_global_routing_random -seed {value}\n'
    file_name = path + f'/{FASTROUTE_TCL}'
    with open(file_name, 'w') as file:
        file.write(new_file)
    return file_name


def get_flow_variant(param, experiment_):
    '''
    Create a hash based on the parameters. This way we don't need to re-run
    experiments with the same configuration.
    '''
    print("Experiment is: " + experiment_)
    variant_hash = hashlib.md5(f"{param}".encode('utf-8')).hexdigest()
    with open(os.path.join(os.getcwd(), 'variant_hash.txt'), 'w') as file:
        file.write(variant_hash)
    return f'{experiment_}/variant-{variant_hash}'


def run_command(cmd, stderr_file=None, stdout_file=None, fail_fast=False, time_limit=28800):
    '''
    Wrapper for subprocess.run
    Allows to run shell command, control print and exceptions.
    '''

    # print(f'Running command: {cmd}\n')
    if stderr_file is not None and stdout_file is not None:
        with open(stderr_file, 'a') as err_file, open(stdout_file, 'a') as out_file :
            process = run(cmd, capture_output=True, text=True, check=False, shell=True, timeout=time_limit, stdout=out_file, stderr=stderr_file)
    else:
        process = run(cmd, capture_output=True, text=True, check=False, shell=True, timeout=time_limit)
        
    # if stderr_file is not None and process.stderr != '':
    #     with open(stderr_file, 'a') as file:
    #         file.write(f'\n\n{cmd}\n{process.stderr}')
    # if stdout_file is not None and process.stdout != '':
    #     with open(stdout_file, 'a') as file:
    #         file.write(f'\n\n{cmd}\n{process.stdout}')
    # if verbose >= 1:
    #     print(process.stderr)
    # if verbose >= 2:
    #     print(process.stdout)

    # if fail_fast and process.returncode != 0:
    #     raise RuntimeError



def openroad(base_dir, parameters, experiment_, last='finish', prev='', path='',):
    '''
    Run OpenROAD-flow-scripts with a given set of parameters.
    '''
    # Make sure path ends in a slash, i.e., is a folder
    flow_variant = get_flow_variant(parameters, experiment_)
    if path != '':
        log_path = f'{path}/{flow_variant}/'
        report_path = log_path.replace('logs', 'reports')
        os.system(f'mkdir -p {log_path}')
        os.system(f'mkdir -p {report_path}')
    else:
        log_path = report_path = os.getcwd() + '/'

   
    if prev != "":
        os.system(f'mkdir -p {base_dir}/flow/results/{platform}/{design}/{flow_variant}')
        os.system(f'mkdir -p {base_dir}/flow/objects/{platform}/{design}/{flow_variant}')
        os.system(f'mkdir -p {base_dir}/flow/logs/{platform}/{design}/{flow_variant}')
        os.system(f'mkdir -p {base_dir}/flow/reports/{platform}/{design}/{flow_variant}')
        os.system(f'cp -rp {base_dir}/flow/results/{platform}/{design}/{prev}/* {base_dir}/flow/results/{platform}/{design}/{flow_variant}/')
        os.system(f'cp -rp {base_dir}/flow/objects/{platform}/{design}/{prev}/* {base_dir}/flow/objects/{platform}/{design}/{flow_variant}/')
        os.system(f'cp -rp {base_dir}/flow/logs/{platform}/{design}/{prev}/* {base_dir}/flow/logs/{platform}/{design}/{flow_variant}/')
        os.system(f'cp -rp {base_dir}/flow/reports/{platform}/{design}/{prev}/* {base_dir}/flow/reports/{platform}/{design}/{flow_variant}/')

    export_command = f'export PATH={INSTALL_PATH}/OpenROAD/bin'
    export_command += f':{INSTALL_PATH}/yosys/bin'
    export_command += f':{INSTALL_PATH}/LSOracle/bin:$PATH'
    export_command += ' && '

    make_command = export_command
    make_command += f'make -C {base_dir}/flow DESIGN_CONFIG=designs/'
    make_command += f'{platform}/{design}/config.mk'
    make_command += f' FLOW_VARIANT={flow_variant} {parameters}'
    make_command += f' NPROC=16 {last}'
    run_command(make_command,
                stderr_file=f'{log_path}error-make-finish.log',
                stdout_file=f'{log_path}make-finish-stdout.log', time_limit=10800)

    metrics_file = os.path.join(report_path, 'metrics.json')
    metrics_command = export_command
    metrics_command += f'{base_dir}/flow/util/genMetrics.py -x'
    metrics_command += f' -v {flow_variant}'
    metrics_command += f' -d {design}'
    metrics_command += f' -p {platform}'
    metrics_command += f' -o {metrics_file}'
    run_command(metrics_command,
                stderr_file=f'{log_path}error-metrics.log',
                stdout_file=f'{log_path}metrics-stdout.log')

    return metrics_file, flow_variant

def set_best_params(platform, design):
    '''
    Get current known best parameters if it exists.
    '''
    params = []
    best_param_file = f'designs/{platform}/{design}/{AUTOTUNER_BEST}'
    if os.path.isfile(best_param_file):
        with open(best_param_file) as file:
            params = json.load(file)
    return params


@ray.remote
def save_best(best_config):
    '''
    Save best configuration of parameters found.
    '''

    new_best_path = f'{LOCAL_DIR}/{ray.get(experiment)}/{AUTOTUNER_BEST}'
    with open(new_best_path, 'w') as new_best_file:
        json.dump(best_config, new_best_file, indent=4)
    print(f'[INFO TUN-0003] Best parameters written to {new_best_path}')




def evaluate_floorplan(metrics):
    floorplan_metrics = metrics.get("floorplan")
    return floorplan_metrics.get("timing__setup__ws") if floorplan_metrics != None else sys.maxsize

def evaluate_placement(metrics):
    floorplan_metrics = metrics.get("floorplan")
    return floorplan_metrics.get("timing__setup__ws") if floorplan_metrics != None else sys.maxsize

def evaluate_groute(metrics): 
    constraints = metrics.get("constraints")
    clock_details = constraints.get("clocks__details") if constraints is not None and len(constraints.get("clocks__details")) > 0 else None 
    if clock_details is None:
        return sys.maxsize
    
    clock_details = float(clock_details[0].split()[1])

    groute_metrics = metrics.get("globalroute")
    worst_slack = groute_metrics.get("timing__setup__ws") if groute_metrics is not None and groute_metrics.get("timing__setup__ws") != "ERR" and groute_metrics.get("timing__setup__ws") != "N/A" else float(sys.maxsize * -1)

    # Performance
    performance = (clock_details - worst_slack)
    
    # Violations
    drvs = 0
    drvs += groute_metrics.get("timing__drv__max_slew")  if groute_metrics is not None and groute_metrics.get("timing__drv__max_slew") != "ERR"  and groute_metrics.get("timing__drv__max_slew") != "N/A"else 0
    drvs += groute_metrics.get("timing__drv__max_fanout")  if groute_metrics is not None and groute_metrics.get("timing__drv__max_fanout") != "ERR" and groute_metrics.get("timing__drv__max_fanout") != "N/A" else 0
    drvs += groute_metrics.get("timing__drv__max_cap")  if groute_metrics is not None and groute_metrics.get("timing__drv__max_cap") != "ERR" and groute_metrics.get("timing__drv__max_cap") != "N/A" else 0
    
    # Clock Skew
    skew = groute_metrics.get("clock__skew__worst")  if groute_metrics is not None and groute_metrics.get("clock__skew__worst") != "ERR" else sys.maxsize


    print(f'performance: {performance}, drvs: {drvs}, skew: {skew}\n')
    return performance + drvs + skew


def read_metrics(data):
    '''
    Collects metrics to evaluate the user-defined objective function.
    '''
    clk_period = 9999999
    worst_slack = 'ERR'
    wirelength = 'ERR'
    num_drc = 'ERR'
    total_power = 'ERR'
    core_util = 'ERR'
    final_util = 'ERR'
    for stage, value in data.items():
        if stage == 'constraints' and len(value['clocks__details']) > 0:
            clk_period = float(value['clocks__details'][0].split()[1])
        if stage == 'floorplan' \
                and 'design__instance__utilization' in value:
            core_util = value['design__instance__utilization']
        if stage == 'detailedroute' and 'route__drc_errors' in value:
            num_drc = value['route__drc_errors']
        if stage == 'detailedroute' and 'route__wirelength' in value:
            wirelength = value['route__wirelength']
        if stage == 'finish' and 'timing__setup__ws' in value:
            worst_slack = value['timing__setup__ws']
        if stage == 'finish' and 'power__total' in value:
            total_power = value['power__total']
        if stage == 'finish' and 'design__instance__utilization' in value:
            final_util = value['design__instance__utilization']
    ret = {
        "clk_period": clk_period,
        "worst_slack": worst_slack,
        "wirelength": wirelength,
        "num_drc": num_drc,
        "total_power": total_power,
        "core_util": core_util,
        "final_util": final_util
    }
    return ret
        
def evaluate_end(raw_metrics): 
    metrics = read_metrics(raw_metrics)
    error = 'ERR' in metrics.values()
    not_found = 'N/A' in metrics.values()
    if error or not_found:
        return (99999999999)
    gamma = (metrics['clk_period'] - metrics['worst_slack']) / 10
    score = metrics['clk_period'] - metrics['worst_slack']
    score = score + gamma * metrics['num_drc']

    print(f'Final score is {score}')
    return score



if __name__ == '__main__':
# def run_experiment(config_path_, platform_, design_, last_step, eval_func, continue_from="", verbose_=0):
    stage_evals = {"floorplan": evaluate_floorplan, "place": evaluate_placement, "globalroute": evaluate_groute, "finish": evaluate_end}
    runs = [("globalroute", "../designs/sky130hd/ibex/autotuner.json",5), 
            ("finish", "../designs/sky130hd/ibex/autotuner_rem.json",5)]

    platform = "sky130hd"
    design = "ibex"
    verbose = 0

    dd = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.stdout = open(f'logs-{dd}.log', 'w')
    sys.stderr = open(f'logs-{dd}.err', 'w')

    LOCAL_DIR = f'../logs/{platform}/{design}'
    INSTALL_PATH = abspath('../../tools/install')

    CONTINUE_FROM = ray.put("")

    for stage in runs:
        DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_str =  f'test-tune-{DATE}-{str(uuid.uuid4())[:5]}'
        print("Running a new experiment with id " + experiment_str + " on pid " + str(os.getpid()))
        config_path = stage[1]
        LAST_STEP = ray.put(stage[0])
        # EVAL_FUNC = ray.put(stage_evals[LAST_STEP])
        experiment = ray.put(experiment_str)
        # Read config and original files before handling where to run in case we
        # need to upload the files.
        config_dict, SDC_ORIGINAL, FR_ORIGINAL = read_config(abspath(config_path))
        search_algo = BasicVariantGenerator(max_concurrent=int(np.floor(cpu_count() / 2)))
        TrainClass = AutoTunerBase

        data = {
            "exp" : experiment_str,
            "last" : stage[0],
            "cont" : ray.get(CONTINUE_FROM),
            "eval" : stage_evals[stage[0]]
        }

        tune_args = dict(
            name=f'{experiment_str}',
            metric='minimum',
            mode='min',
            num_samples=stage[2],
            fail_fast=True,
            local_dir=LOCAL_DIR,
            resume="",
            stop={"training_iteration": 1},
        )

        tune_args['search_alg'] = search_algo
        # tune_args['scheduler'] = AsyncHyperBandScheduler()
        tune_args['config'] = config_dict
        start_time = time.time()


        analysis = tune.run(tune.with_parameters(TrainClass, data=data), **tune_args)
        end_time = time.time()
        # task_id = save_best.remote(analysis.best_config)
        # _ = ray.get(task_id)
        f = open(f'{LOCAL_DIR}/{experiment_str}/{analysis.best_trial.trial_id}', 'r')
        CONTINUE_FROM = ray.put(f.read())
        print(f'Best Variant for {ray.get(LAST_STEP)} is {ray.get(CONTINUE_FROM)} with params {analysis.best_result} achieved in {(end_time - start_time)} seconds.') 
        sys.stdout.flush()
        time.sleep(5)


    sys.stdout.close()
    sys.stderr.close()

    # DATE = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # experiment = f'test-tune-{DATE}-{uuid.uuid4()}'
    # experiment_path = f'../logs/{platform_}/{design_}/{experiment}'
    
    # config_path = config_path_
    # platform = platform_
    # design = design_
    # verbose = verbose_
    # LAST_STEP = last_step
    # EVAL_FUNC = eval_func
    # CONTINUE_FROM = continue_from

    # # For local runs, use the same folder as other ORFS utilities.
    # # os.chdir(os.path.dirname(abspath(__file__)) + '/../')

    # LOCAL_DIR = f'../logs/{platform}/{design}'
    # INSTALL_PATH = abspath('../../tools/install')

    # # Read config and original files before handling where to run in case we
    # # need to upload the files.
    # config_dict, SDC_ORIGINAL, FR_ORIGINAL = read_config(abspath(config_path))

    # print(FR_ORIGINAL)
    # best_params = set_best_params(platform, design)
    # search_algo = BasicVariantGenerator(max_concurrent=int(np.floor(cpu_count() / 2)))
    # TrainClass = AutoTunerBase

    # tune_args = dict(
    #     name=f'{experiment}',
    #     metric='minimum',
    #     mode='min',s
    #     num_samples=10,
    #     fail_fast=True,
    #     local_dir=LOCAL_DIR,
    #     resume="",
    #     stop={"training_iteration": 1},
    # )

    # tune_args['search_alg'] = search_algo
    # tune_args['scheduler'] = AsyncHyperBandScheduler()
    # tune_args['config'] = config_dict

    # analysis = tune.run(TrainClass, **tune_args)

    # task_id = save_best.remote(analysis.best_config)
    # _ = ray.get(task_id)
    # # print(f'[INFO TUN-0002] Best parameters found: {analysis.best_config}')
    # # print(f'Best trial id is {analysis.best_trial.trial_id} ')

    # f = open(f'{LOCAL_DIR}/{experiment}/{analysis.best_trial.trial_id}', 'r')
    # best_variant = f.read()
    # print(f'Best Variant is:{best_variant}') 



# def evaluate_floorplan(metrics):
#     floorplan_metrics = metrics.get("floorplan")
#     return floorplan_metrics.get("design__instance__utilization") if floorplan_metrics != None else sys.maxsize

# def evaluate_placement(metrics):
#     floorplan_metrics = metrics.get("floorplan")
#     return floorplan_metrics.get("design__instance__utilization") if floorplan_metrics != None else sys.maxsize


# if __name__ == '__main__':
    
#      # For local runs, use the same folder as other ORFS utilities.
#     os.chdir(os.path.dirname(abspath(__file__)) + '/../')


#     config_path = "util/rayconfigs/autotuner.json"
    
    
#     # Run until floorplan
#     best_variant = run_experiment(config_path, "sky130hd", "gcd", "floorplan", evaluate_floorplan)

#     print(f'floorplan best variant is: {best_variant}\n')

#     # Run until placement
#     best_variant = run_experiment(config_path, "sky130hd", "gcd", "place", evaluate_placement, best_variant)

#     print(f'placement best Variant is: {best_variant}\n')
    
#     # # Run until routing
#     # best_variant = run_experiment(config_path, "sky130hd", "gcd", "route", evaluate_placement, best_variant)

#     # print(f'routing best variant is: {best_variant}\n')
    
#     # # Run until placement
#     # best_variant = run_experiment(config_path, "sky130hd", "gcd", "finish", evaluate_placement, best_variant)

#     # print(f'final best variant is: {best_variant}\n')
