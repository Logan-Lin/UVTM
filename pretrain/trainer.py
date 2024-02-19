import math
from time import time
from abc import abstractmethod

import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from utils import create_if_noexists, next_batch, cal_model_size, gather_all_param
from preparer import BatchPreparer
from data import SET_NAMES


class Trainer:
    """
    Base class of the pre-training helper class.
    Implements most of the functions shared by all types of pre-trainers.
    """

    def __init__(self, data, models, trainer_name,
                 loss_func, batch_size, num_epoch, lr, device,
                 log_name_key, train_prop=1.0, cache_epoches=False,
                 meta_types=['trip'], suffix='', **kwargs):
        """
        :param data: dataset object.
        :param models: list of models. Depending on the type of pretext task, they can be encoders or decoders.
        :param trainer_name: name of the pre-trainer.
        :param loss_func: loss function module defined by specific pretext task.
        :param log_name_key: the name key for saving training logs. All saved log will use this key as their file name.
        :param cache_epoches: whether to save all models after every training epoch.
        :param meta_types: list of meta types used for pre-training, corresponds to the meta names in dataset object.
        """
        self.data = data
        # The list of models may have different usage in different types of trainers.
        self.models = [model.to(device) for model in models]
        self.trainer_name = trainer_name

        self.loss_func = loss_func.to(device)
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.train_prop = train_prop
        self.cache_epoches = cache_epoches

        self.meta_types = meta_types
        model_name = '_'.join([model.name for model in models])
        meta_name = '_'.join(self.meta_types)
        self.BASE_KEY = f'{trainer_name}_b{batch_size}-lr{lr}-tp{train_prop}{suffix}/{loss_func.name}/{self.data.name}_{meta_name}/{model_name}'
        self.model_cache_dir = f'{data.base_path}/model_cache/{self.BASE_KEY}'
        self.model_save_dir = f'{data.base_path}/model_save/{self.BASE_KEY}'
        self.log_save_dir = f'{data.base_path}/log/{self.BASE_KEY}'

        self.optimizer = torch.optim.Adam(gather_all_param(*self.models, self.loss_func), lr=lr)
        self.log_name_key = log_name_key

    def prepare_batch_iter(self, select_set, data_prop=1.0):
        """ Prepare iterations of batches for one training epoch. """
        metas = []
        meta_lengths = [0]
        for meta_type in self.meta_types:
            meta = self.data.load_meta(meta_type, select_set)
            metas += meta
            meta_lengths.append(len(meta))
        self.meta_split_i = np.cumsum(meta_lengths)
        self.batch_iter = list(zip(*metas))
        if data_prop < 1.0:
            self.batch_iter = shuffle(self.batch_iter)[:math.ceil(data_prop * len(self.batch_iter))]
        self.num_iter = math.ceil((len(self.batch_iter) - 1) / self.batch_size)

    def prepare_batch_meta(self, batch_meta):
        """ Prepare seperated meta arguments for one batch. """
        zipped = list(zip(*batch_meta))
        prepared_metas = []
        for i, meta_type in enumerate(self.meta_types):
            meta_prepare_func = BatchPreparer.fetch_prepare_func(meta_type)
            prepared_metas += meta_prepare_func(zipped[self.meta_split_i[i]:self.meta_split_i[i+1]], self.device)
        return prepared_metas

    def train(self, start=-1):
        """
        Finish the full training process.

        :param start: if given a value of 0 or higher, will try to load the trained model 
            cached after the start-th epoch training, and resume training from there.
        """
        self.prepare_batch_iter(0, self.train_prop)
        self.train_epoches(start)
        self.save_models()

    def train_epoches(self, start=-1, name='Pretraining'):
        """ Train the models for multiple iterations (denoted by num_epoch). """
        self.train_state()

        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        train_logs = []
        desc_text = f'{name}, avg loss %.4f'
        with trange(start+1, self.num_epoch, desc=desc_text % 0.0) as tbar:
            for epoch_i in tbar:
                s_time = time()
                epoch_avg_loss = self.train_epoch()
                e_time = time()
                tbar.set_description(desc_text % epoch_avg_loss)
                train_logs.append([epoch_i, e_time - s_time, epoch_avg_loss])

                if self.cache_epoches and epoch_i < self.num_epoch - 1:
                    self.save_models(epoch_i)

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key=f'{name}_log')

    def train_epoch(self):
        """ Train the models for one epoch. """
        loss_log = []
        for batch_meta in tqdm(next_batch(shuffle(self.batch_iter), self.batch_size),
                               desc=f'-->Traverse batches', total=self.num_iter, leave=False):
            self.optimizer.zero_grad()
            loss = self.forward_loss(batch_meta)
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def finetune(self, ft_meta_types=None, **ft_params):
        if ft_meta_types is not None:
            self.meta_types = ft_meta_types
        for key, value in ft_params.items():
            setattr(self, key, value)
        self.prepare_batch_iter(0, ft_params.get('ft_prop', 1.0))
        self.train_epoches(name='Finetuning')

    @abstractmethod
    def forward_loss(self, batch_meta):
        """
        Controls how the trainer forward models and meta datas to the loss function.
        Might be different depending on specific type of pretex task.
        """
        return self.loss_func(self.models, *self.prepare_batch_meta(batch_meta))

    def save_models(self, epoch=None):
        """ Save learnable parameters in the models as pytorch binaries. """
        for model in (*self.models, self.loss_func):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the model. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        return model

    def load_models(self, epoch=None):
        """ 
        Load all models from file. 
        """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model, epoch)
        self.loss_func = self.load_model(self.loss_func, epoch)

    def get_models(self):
        """ Obtain all models in the trainer in evluation state. """
        self.eval_state()
        return self.models

    def train_state(self):
        for model in self.models:
            model.train()
        self.loss_func.train()

    def eval_state(self):
        for model in self.models:
            model.eval()
        self.loss_func.eval()


class GenerativeTrainer(Trainer):
    """
    Trainer for generative pre-training.
    Contains a generate function for evaluating the recovered input.
    """

    def __init__(self, enc_meta_i=[], rec_meta_i=[], con_meta_i=[], **kwargs):
        """
        :param enc_meta_i: list of meta indices indicating which meta data to fed into the encoders.
        :param rec_meta_i: list of meta indices indicating which meta data to use as recovery target.
        """
        meta_i_name = [','.join(map(str, meta_i)) for meta_i in [enc_meta_i, rec_meta_i, con_meta_i]]
        super().__init__(trainer_name='generative',
                         suffix=f'_enc{meta_i_name[0]}-rec{meta_i_name[1]}-con{meta_i_name[2]}',
                         **kwargs)
        self.generation_save_dir = f'{self.data.base_path}/generation/{self.BASE_KEY}'
        self.enc_meta_i = enc_meta_i
        self.rec_meta_i = rec_meta_i
        self.con_meta_i = con_meta_i

    def _prepare_enc_rec(self, batch_meta):
        metas = self.prepare_batch_meta(batch_meta)
        enc_meta, rec_meta, con_meta = ([] for _ in range(3))
        for i in self.enc_meta_i:
            enc_meta += metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
        for i in self.rec_meta_i:
            rec_meta += metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
        for i in self.con_meta_i:
            con_meta += metas[self.meta_split_i[i]:self.meta_split_i[i+1]]
        return enc_meta, rec_meta, con_meta

    def forward_loss(self, batch_meta):
        """ For generative training, the batch is split into encode and recovery meta, then fed into the loss function. """
        enc_meta, rec_meta, con_meta = self._prepare_enc_rec(batch_meta)
        return self.loss_func(self.models, enc_meta, rec_meta, con_meta)

    def generate(self, set_index, gen_meta_types=None,
                 enc_meta_i=None, rec_meta_i=None, con_meta_i=None, **gen_params):
        """ Generate and save recovered meta data. """
        if gen_meta_types is not None:
            self.meta_types = gen_meta_types
        if enc_meta_i is not None:
            self.enc_meta_i = enc_meta_i
        if rec_meta_i is not None:
            self.rec_meta_i = rec_meta_i
        if con_meta_i is not None:
            self.con_meta_i = con_meta_i

        self.prepare_batch_iter(set_index)
        self.eval_state()

        gen_dicts = []
        with tqdm(next_batch(self.batch_iter, self.batch_size),
                  desc='Generating', total=self.num_iter) as bar:
            s_time = time()
            for batch_meta in bar:
                enc_meta, rec_meta, con_meta = self._prepare_enc_rec(batch_meta)
                gen_dict, gen_save_name = self.loss_func.generate(self.models, enc_meta, rec_meta, con_meta,
                                                                  stat=self.data.stat, gen_type=self.meta_types[self.rec_meta_i[0]],
                                                                  **gen_params)
                gen_dicts.append(gen_dict)
            e_time = time()
        print(f'Generation time: {round(e_time - s_time, 3)} seconds')
        numpy_dict = {key: np.concatenate([gen_dict[key] for gen_dict in gen_dicts], 0) for key in gen_dicts[0].keys()}

        gen_save_dir = f'{self.generation_save_dir}/{gen_save_name}'
        create_if_noexists(gen_save_dir)
        gen_save_path = f'{gen_save_dir}/{SET_NAMES[set_index][1]}_{self.log_name_key}.npz'
        np.savez(gen_save_path, **numpy_dict)
