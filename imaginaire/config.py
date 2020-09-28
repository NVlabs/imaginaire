# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
"""Config utilities for yml file."""

import collections
import functools
import os
import re

import yaml
from imaginaire.utils.distributed import master_only_print as print


class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()
        # Set default parameters.
        # Logging.
        large_number = 1000000000
        self.snapshot_save_iter = large_number
        self.snapshot_save_epoch = large_number
        self.snapshot_save_start_iter = 0
        self.snapshot_save_start_epoch = 0
        self.image_save_iter = large_number
        self.image_display_iter = large_number
        self.max_epoch = large_number
        self.max_iter = large_number
        self.logging_iter = 100

        # Trainer.
        self.trainer = AttrDict(
            model_average=False,
            model_average_beta=0.9999,
            model_average_start_iteration=1000,
            model_average_batch_norm_estimation_iteration=30,
            model_average_remove_sn=True,
            image_to_tensorboard=False,
            hparam_to_tensorboard=False,
            distributed_data_parallel='pytorch',
            delay_allreduce=True,
            gan_relativistic=False,
            gen_step=1,
            dis_step=1)

        # Networks.
        self.gen = AttrDict(type='imaginaire.generators.dummy')
        self.dis = AttrDict(type='imaginaire.discriminators.dummy')

        # Optimizers.
        self.gen_opt = AttrDict(type='adam',
                                fused_opt=True,
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        self.dis_opt = AttrDict(type='adam',
                                fused_opt=True,
                                lr=0.0001,
                                adam_beta1=0.0,
                                adam_beta2=0.999,
                                eps=1e-8,
                                lr_policy=AttrDict(iteration_mode=False,
                                                   type='step',
                                                   step_size=large_number,
                                                   gamma=1))
        # Data.
        self.data = AttrDict(name='dummy',
                             type='imaginaire.datasets.images',
                             num_workers=0)
        self.test_data = AttrDict(name='dummy',
                                  type='imaginaire.datasets.images',
                                  num_workers=0,
                                  test=AttrDict(is_lmdb=False,
                                                roots='',
                                                batch_size=1))


# Cudnn.
        self.cudnn = AttrDict(deterministic=False,
                              benchmark=True)

        # Others.
        self.pretrained_weight = ''
        self.inference_args = AttrDict()

        # Update with given configurations.
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader=loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        recursive_update(self, cfg_dict)

        # Put common opts in both gen and dis.
        if 'common' in cfg_dict:
            self.common = AttrDict(**cfg_dict['common'])
            self.gen.common = self.common
            self.dis.common = self.common

        if verbose:
            print(' imaginaire config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def rsetattr(obj, attr, val):
    """Recursively find object and set value"""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find object and return value"""

    def _getattr(obj, attr):
        r"""Get attribute."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d
