"""
train_shakespeare.py
---------------------

This is a configuration file for training a small character-level GPT model on the
Shakespeare dataset. It’s a great starting point for experimenting with character-level 
modeling in nanoGPT.

Key points:
- Uses a small "baby" GPT architecture (6 layers, 6 heads, 384 embedding size).
- Designed for CPU or small GPU environments (e.g. MacBooks).
- Only saves checkpoints if the validation loss improves (to avoid clutter).
- Tiny dataset (Shakespeare), so overfitting is expected quickly.
- Uses a high learning rate for fast convergence.

Usage:
------
$ python train.py config/train_shakespeare.py

To adapt for your own dataset (e.g. Barangay ordinances):
- Replace `dataset = 'shakespeare_char'` with your dataset name.
- Adjust `block_size` and `max_iters` based on your dataset’s length and training goals.
- Update `out_dir` to a unique output directory.

Example:
dataset = 'barangay_char'
out_dir = 'out-barangay'
block_size = 256  # or a value that works with your text

Have fun training!
"""

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 500
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
