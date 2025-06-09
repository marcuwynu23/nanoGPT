"""
finetune_shakespeare.py
------------------------

This is a configuration file for fine-tuning a pre-trained GPT-2 model on a small dataset 
like Shakespeare. It’s designed for use with nanoGPT’s `train.py` script.

Key points:
- Starts from the pre-trained GPT-2 XL model (`init_from = 'gpt2-xl'`).
- Uses small batch size (1) with gradient accumulation to simulate larger batches.
- Runs for only 20 iterations (short demonstration of fine-tuning).
- Uses a small learning rate (3e-5) without decay, good for fine-tuning.
- Only saves checkpoints if validation loss improves (to save disk space).
- Adjust `out_dir`, `dataset`, and `wandb_log` as needed.

Usage:
------
$ python train.py config/finetune_shakespeare.py

Adjust this file for your own fine-tuning tasks (like Barangay ordinances!) 
by setting:
- dataset = 'barangay_data'
- out_dir = 'out-barangay'
- init_from = 'gpt2-xl' or 'scratch'

Happy fine-tuning!
"""

import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
