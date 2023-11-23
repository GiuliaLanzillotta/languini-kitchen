# Copyright 2023 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from munch import Munch  # Munch is a dictionary that supports attribute-style access


config_names = [
    'mini',
    'tiny',
    'small',
    'medium',
    'large',
    'XL',
]


def add_exp_name(config):
    """Constructs the name of the log folder used to easily identify the experiment. """
    c = config
    c.exp_name = ("GPT{}_{}_bsz{}{}_sl{}_coslr{}to{}_h{}_ff{}_nH{}_dH{}_nl{}_clip{}_decay{}k_workers{}{}_fp16_seed{}{}"
                  .format("_flash" if c.use_flash else "",
                          c.dataset.replace("_", ""),
                          c.train_batch_size,
                          "" if c.gradient_accumulation_steps == 1 else f"_micro{c.gradient_accumulation_steps}",
                          c.seq_len,
                          c.max_lr,
                          c.min_lr,
                          c.h_dim,
                          c.mlp_dim,
                          c.n_heads,
                          c.head_dim,
                          c.n_layers,
                          c.grad_clip_norm,
                          c.decay_steps // 1_000,
                          c.n_workers,
                          "" if c.compile == "None" else f"_{c.compile}Compile",
                          c.seed,
                          "_debug" if c.debug else "")                          
                 )


## Add experiment configs

def add_distil_configs(config):
    """ Adding the config parameters used in a distillation experiment."""
    config.alpha = 0
    """ this is a GPT-medium size model trained for 96h RTX3090 hours which resulted in about 5.7B training tokens. 
    The training data is about 24B tokens in total. Achieves 2.032 nppl. It has about 330M parameters"""
    config.teacher_checkpoint_path = "/media/hofmann-scratch/glanzillo/languini-models/gpt-medium"
    config.mse = False # flag: whether to use the mse loss or KL 
    return config

def load_config(name=None, distil_experiment=False):

    c = Munch(
        # data
        data_root = "/local/home/stuff/languini_data/books",
        relative_log_path = "logs",         # Relative path to the log folder within the project folder languini-kitchen/projects/gpt/logs/
        dataset = "books_16384",
        vocab_size = 16384,
        debug = False,                      # simply adds a "_debug" suffix so logs are easily distinguishable

        # optimiser
        seed = 0,
        gradient_accumulation_steps = 1,    # number of batches before doing a gradient step
        train_batch_size = 16,              # make sure batch sizes are an integer multiple of the number of workers
        eval_batch_size = 16,
        test_batch_size = 16,
        seq_len = 512,
        max_eval_steps = 500,
        max_train_steps = 500_000,          # total number of training steps
        decay_steps = 500_000,              # number of steps over which we will decay the learning rate
        max_lr = 0.0006,                    # starting learning rate
        min_lr = 0.000006,                  # final learning rate
        grad_clip_norm = 0.0,               # gradient norm clipping
        tokens_per_second = 0,              # tokens per second throughput of this config on the hardware run; used for logging over gpuhours
        sgd=False,                          # whether to use SGD as optimizer

        # perform certain tasks every N steps
        eval_every = 1_000,                 # perform a fast evaluation
        log_terminal_every = 100,           # print the current loss to terminal
        log_metrics_every = 100,            # log accuracy and loss metrics
        log_grads_every = 1_000,            # log gradients and step sizes
        log_activations_every = -1,         # log gradients and step sizes
        log_ckpt_every = 5_000,             # save model checkpoint to disk

        # logging
        logger_type = 'wandb',  # can be 'tb', 'wandb' or 'all'
        wandb_project_name = 'DataEfficientDistillation', 
        wandb_entity_name = 'continually',
    )
    # default model
    if not name or name == 'default':
        name = 'mini'

    # model
    c.use_flash = False
    if name == 'mini':
        c.n_layers = 4
        c.h_dim = 512
        c.mlp_dim = 2048
        c.head_dim = 32
        c.n_heads = 8
    if name == 'mini2':
        c.n_layers = 4
        c.h_dim = 1024
        c.mlp_dim = 2048
        c.head_dim = 64
        c.n_heads = 16
    elif name == 'tiny':
        c.n_layers = 4
        c.h_dim = 768
        c.mlp_dim = 3072
        c.head_dim = 64
        c.n_heads = 12
    elif name == 'small':
        c.n_layers = 12
        c.h_dim = 768
        c.mlp_dim = 3072
        c.head_dim = 64
        c.n_heads = 12
    elif name == 'medium':
        c.n_layers = 24
        c.h_dim = 1024
        c.mlp_dim = 4096
        c.head_dim = 64
        c.n_heads = 16
    elif name == 'large':
        c.n_layers = 24
        c.h_dim = 1536
        c.mlp_dim = 6144
        c.head_dim = 96
        c.n_heads = 16
    elif name == 'XL':
        c.n_layers = 24
        c.h_dim = 2048
        c.mlp_dim = 8192
        c.head_dim = 128
        c.n_heads = 24
    else:
        raise ValueError(f"Config name {name} is an invalid name. ")
    
    if distil_experiment: c = add_distil_configs(c)

    return c
