import json
import os
from argparse import ArgumentParser
import copy

import torch
from torch.cuda import is_available as cuda_available

from data import Data
from pretrain import trainer as PreTrainer
import pretrain.losses as TrainLoss
import model as Model
import utils


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-c', '--config', help='name of the config file to use', type=str, required=True)
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = "1"
device = f'cuda:0' if cuda_available() else 'cpu'
datetime_key = utils.get_datetime_key()
torch.autograd.set_detect_anomaly(True)

print(f'Datetime key: {datetime_key}')
# Load config file
with open(f'config/{args.config}.json', 'r') as fp:
    config = json.load(fp)

# Each config file can contain multiple entries. Each entry is a different set of configuration.
for num_entry, entry in enumerate(config):
    print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')

    # Load dataset.
    data_entry = entry['data']
    data = Data(data_entry['name'])
    data.load_stat()
    num_roads = data.data_info['num_road']

    conf_save_dir = os.path.join(data.base_path, 'config')
    utils.create_if_noexists(conf_save_dir)
    with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.json'), 'w') as fp:
        json.dump(entry, fp)

    # Each entry can be repeated for several times.
    num_repeat = entry.get('repeat', 1)
    for repeat_i in range(num_repeat):
        print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')

        models = []
        for model_entry in entry['models']:
            # Create models.
            model_name = model_entry['name']
            model_config = model_entry.get('config', {})
            if model_name == 'dualpos_transformer':
                models.append(Model.transformer.DualPosTransformer(**model_config))
            else:
                raise NotImplementedError(f'No encoder called "{model_name}".')

        for model in models:
            print(model.name, 'size', utils.cal_model_size(model), 'MB')

        # Create pre-training loss function.
        if 'pretrain' in entry:
            pretrain_entry = entry['pretrain']
            loss_entry = pretrain_entry['loss']
            loss_name = loss_entry['name']

            loss_param = loss_entry.get('config', {})
            extra_param_names = ''
            if 'extra_params' in loss_entry:
                extra_param_arrs = []
                for meta_type in loss_entry['extra_params']:
                    extra_param_arrs += list(data.load_meta(meta_type, 0))
                extra_param_names = ','.join(loss_entry['extra_params'])
            if loss_name == 'uvtm':
                loss_func = TrainLoss.uvtm.UVTM(num_tokens=num_roads, **loss_param,
                                                extra_params=extra_param_arrs,
                                                model_name_suf=extra_param_names)
            else:
                raise NotImplementedError(f'No loss function called "{loss_name}".')

            print(loss_func.name, 'size', utils.cal_model_size(loss_func), 'MB')

            # Create pre-trainer.
            pretrainer_entry = pretrain_entry['trainer']
            pretrainer_name = pretrainer_entry['name']
            pretrainer_comm_param = {"data": data, "models": models, "loss_func": loss_func,
                                     "device": device, "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'}
            pretrainer_config = pretrainer_entry.get('config', {})
            if pretrainer_name == 'generative':
                pre_trainer = PreTrainer.GenerativeTrainer(**pretrainer_comm_param, **pretrainer_config)
            else:
                raise NotImplementedError(f'No loss function called "{pretrainer_name}".')

            # Pre-training on the training set, or load from trained cache.
            if pretrain_entry.get('load', False):
                pre_trainer.load_models()
            else:
                pre_trainer.train(pretrain_entry.get('resume', -1))

            if "finetune" in pretrain_entry:
                finetune_entry = pretrain_entry['finetune']
                pre_trainer.finetune(**finetune_entry.get('config', {}))

            if "generation" in pretrain_entry:
                generation_entry = pretrain_entry['generation']
                pre_trainer.generate(generation_entry['eval_set'],
                                     **generation_entry.get('config', {}))

            models = pre_trainer.get_models()
        else:
            print('No pre-training.')
            pre_trainer = None
