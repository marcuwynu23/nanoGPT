# barangay_ai_sampler.py
import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

class Sampler:
    def __init__(
        self,
        out_dir='out',
        device='cpu',
        dtype='float32',  # default to float32 for CPU
        compile_model=False
    ):
        self.out_dir = out_dir
        self.device = device
        self.compile = compile_model
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[dtype]
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=self.ptdtype)
        self.model = self._load_model()

        # Load meta if available
        meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')  # default fallback
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi, self.itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [self.stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            # Use GPT-2 encoding by default
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)

    def _load_model(self):
        # Load checkpoint
        ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        if self.compile:
            model = torch.compile(model)
        return model

    def generate_text(
        self,
        prompt="\n",
        num_samples=1,
        max_new_tokens=100,
        temperature=0.8,
        top_k=200
    ):
        start_ids = self.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        generated_texts = []
        with torch.no_grad():
            with self.ctx:
                for _ in range(num_samples):
                    y = self.model.generate(
                        x,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                    generated_texts.append(self.decode(y[0].tolist()))
        return generated_texts
