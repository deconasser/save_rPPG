# # main.py — train/test + visualize MSTmap + realtime GPU metrics + infer latency
# import os
# import yaml
# import numpy as np
# from argparse import ArgumentParser
# from functools import partial
# from time import perf_counter

# import torch
# import torch.nn as nn
# import lightning.pytorch as pl
# from lightning.pytorch.loggers.wandb import WandbLogger
# from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint

# from torch_geometric.nn import RGCNConv, TransformerConv  # cho FLOPs hook
# from model import Model  # model.py đã được cập nhật với infer_and_visualize_mstmap

# # ==================== FLOPs hooks (như bạn có sẵn) ====================
# def conv_flops_counter_hook(module, input, output):
#     batch_size = input[0].shape[0]
#     output_dims = list(output.shape[2:])
#     kernel_dims = list(module.kernel_size)
#     in_channels = module.in_channels
#     out_channels = module.out_channels
#     groups = module.groups
#     filters_per_channel = out_channels // groups
#     conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel
#     active_elements_count = batch_size * np.prod(output_dims)
#     overall_flops = conv_per_position_flops * active_elements_count
#     bias_flops = out_channels * active_elements_count if module.bias is not None else 0
#     module.__flops__ += int(overall_flops + bias_flops)

# def linear_flops_counter_hook(module, input, output):
#     x = input[0]
#     out_last = output.shape[-1]
#     bias_flops = out_last if module.bias is not None else 0
#     module.__flops__ += int(np.prod(x.shape) * out_last + bias_flops)

# def bn_flops_counter_hook(module, input, output):
#     batch_flops = np.prod(input[0].shape)
#     if module.affine:
#         batch_flops *= 2
#     module.__flops__ += int(batch_flops)

# def relu_flops_counter_hook(module, input, output):
#     module.__flops__ += int(output.numel())

# def pooling_flops_counter_hook(module, input, output):
#     module.__flops__ += int(np.prod(input[0].shape))

# def rgcn_conv_flops_counter_hook(module, input, output):
#     edge_index = input[1]
#     num_edges = edge_index.size(1)
#     module.__flops__ += int(num_edges * module.out_channels)

# def transformer_conv_flops_counter_hook(module, input, output):
#     edge_index = input[1]
#     num_edges = edge_index.size(1)
#     num_heads = module.heads
#     head_dim = module.out_channels // num_heads
#     extra_flops = num_heads * (4 * num_edges * head_dim + 5 * num_edges)
#     module.__flops__ += int(extra_flops)

# flops_counter_hooks = {
#     nn.Conv1d: conv_flops_counter_hook,
#     nn.Conv2d: conv_flops_counter_hook,
#     nn.Conv3d: conv_flops_counter_hook,
#     nn.Linear: linear_flops_counter_hook,
#     nn.BatchNorm1d: bn_flops_counter_hook,
#     nn.BatchNorm2d: bn_flops_counter_hook,
#     nn.BatchNorm3d: bn_flops_counter_hook,
#     nn.ReLU: relu_flops_counter_hook,
#     nn.ReLU6: relu_flops_counter_hook,
#     nn.LeakyReLU: relu_flops_counter_hook,
#     nn.MaxPool1d: pooling_flops_counter_hook,
#     nn.MaxPool2d: pooling_flops_counter_hook,
#     nn.MaxPool3d: pooling_flops_counter_hook,
#     nn.AvgPool1d: pooling_flops_counter_hook,
#     nn.AvgPool2d: pooling_flops_counter_hook,
#     nn.AvgPool3d: pooling_flops_counter_hook,
#     nn.MultiheadAttention: linear_flops_counter_hook,
#     RGCNConv: rgcn_conv_flops_counter_hook,
#     TransformerConv: transformer_conv_flops_counter_hook,
# }

# def add_flops_counting_to_model(model):
#     def add_hooks(m):
#         if type(m) in flops_counter_hooks:
#             m.register_buffer('__flops__', torch.zeros(1, dtype=torch.int64))
#             m.register_forward_hook(flops_counter_hooks[type(m)])
#     model.apply(add_hooks)

# def compute_macs(model, input_tensor):
#     add_flops_counting_to_model(model)
#     model.eval()
#     with torch.no_grad():
#         _ = model(input_tensor)
#     total = 0
#     for m in model.modules():
#         if hasattr(m, '__flops__'):
#             total += m.__flops__.item()
#     return total / 1e9  # GMACs

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # ==================== Lightning callbacks & helpers ====================
# def get_callback_cls(name: str):
#     name = name.lower()
#     if name == 'devicestatsmonitor':
#         return partial(
#             DeviceStatsMonitor,
#             cpu_stats=True,
#         )
#     elif name == 'learningratemonitor':
#         return LearningRateMonitor
#     elif name == 'modelcheckpoint':
#         return partial(
#             ModelCheckpoint,
#             save_last=True,
#             every_n_epochs=5,
#         )
#     elif name == 'valckpt':
#         return partial(
#             ModelCheckpoint,
#             save_top_k=1,
#             monitor='val/loss',
#             mode='min',  
#         )
#     raise ValueError(f'Unknown callback: {name}')

# from time import perf_counter
# import torch
# import lightning.pytorch as pl

# class GPULatencyMemoryMonitor(pl.Callback):
#     def __init__(self, log_every_n_steps: int = 1, stage_prefix: str = ""):
#         self.log_every_n_steps = log_every_n_steps
#         self.stage = stage_prefix
#         self._t0 = None

#     def _device(self, pl_module):
#         try:
#             return next(pl_module.parameters()).device
#         except StopIteration:
#             return torch.device("cpu")

#     def _start(self, pl_module):
#         self._t0 = perf_counter()
#         dev = self._device(pl_module)
#         if dev.type == "cuda":
#             torch.cuda.reset_peak_memory_stats(dev)

#     def _end(self, pl_module, tag_prefix: str):
#         t_ms = (perf_counter() - self._t0) * 1000.0
#         pl_module.log(f"{tag_prefix}/batch_latency_ms", t_ms, on_step=True, on_epoch=False, prog_bar=False)
#         dev = self._device(pl_module)
#         if dev.type == "cuda":
#             alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
#             reserv = torch.cuda.max_memory_reserved(dev) / (1024**2)
#             pl_module.log(f"{tag_prefix}/gpu_mem_alloc_mb", alloc, on_step=True, on_epoch=False, prog_bar=False)
#             pl_module.log(f"{tag_prefix}/gpu_mem_reserved_mb", reserv, on_step=True, on_epoch=False, prog_bar=False)

#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         self._start(pl_module)

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if (batch_idx + 1) % self.log_every_n_steps == 0:
#             self._end(pl_module, "train")

#     def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
#         self._start(pl_module)

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         self._end(pl_module, "val")

#     def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
#         self._start(pl_module)

#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         self._end(pl_module, "test")



# def benchmark_inference(pl_model, test_loader=None, iters: int = 20):
#     pl_model.eval()
#     dev = next(pl_model.parameters()).device if any(True for _ in pl_model.parameters()) else torch.device("cpu")

#     if test_loader is None:
#         test_loader = pl_model.test_dataloader()[0]
#     frames, _, _ = next(iter(test_loader))
#     frames = frames.to(dev, non_blocking=True)

#     with torch.no_grad():
#         for _ in range(3):
#             _ = pl_model(frames)
#             _ = pl_model.predict(frames)
#             if dev.type == "cuda":
#                 torch.cuda.synchronize(dev)

#     if dev.type == "cuda":
#         torch.cuda.reset_peak_memory_stats(dev)

#     times = []
#     with torch.no_grad():
#         for _ in range(iters):
#             t0 = perf_counter()
#             _ = pl_model(frames)
#             _ = pl_model.predict(frames)
#             if dev.type == "cuda":
#                 torch.cuda.synchronize(dev)
#             times.append((perf_counter() - t0) * 1000.0)

#     lat_ms = float(sum(times) / len(times))
#     lat_std = float((sum((t - lat_ms) ** 2 for t in times) / len(times)) ** 0.5)

#     if dev.type == "cuda":
#         gpu_name = torch.cuda.get_device_name(dev)
#         alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
#         reserv = torch.cuda.max_memory_reserved(dev) / (1024**2)
#     else:
#         gpu_name, alloc, reserv = "cpu", 0.0, 0.0

#     print(f"[Benchmark] device={gpu_name} | latency_ms/seq={lat_ms:.2f} ± {lat_std:.2f} | "
#           f"alloc={alloc:.1f}MB reserved={reserv:.1f}MB")

#     return {"device": gpu_name, "latency_ms": lat_ms, "latency_std_ms": lat_std,
#             "alloc_mb": alloc, "reserved_mb": reserv}


# # ==================== Main ====================
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help='Path to config file')
#     parser.add_argument('--split_idx', type=int, required=True, help='Index of split in 5-fold cross validation')
#     args = parser.parse_args()

#     run_name = os.path.split(args.config)[-1].split('.')[0]
#     proj_name = run_name.split('_')[-1]
#     save_dir = os.path.join('logs', run_name)
#     os.makedirs(save_dir, exist_ok=True)

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
#     config['split_idx'] = args.split_idx

#     # seed
#     pl.seed_everything(config['seed'], workers=True)

#     # build model
#     model = Model(config)

#     # logger
#     logger = WandbLogger(
#         name=run_name + f'_fold{args.split_idx}',
#         save_dir=save_dir,
#         project=proj_name,
#         log_model=True,
#     )

#     # callbacks
#     callbacks = []
#     for name in config['trainer']['callbacks']:
#         callbacks.append(get_callback_cls(name)())
#     # thêm realtime GPU monitor (log mỗi batch)
#     # callbacks.append(GPULatencyMemoryMonitor(log_every_n_steps=1))

#     # trainer
#     trainer = pl.Trainer(
#         precision='16-mixed',
#         max_epochs=config['trainer']['max_epochs'],
#         deterministic='warn',
#         logger=logger,
#         callbacks=callbacks,
#         default_root_dir=save_dir,
#     )

#     # train + test
#     # trainer.fit(model)
#     # trainer.test(model, ckpt_path='best')
#     # ==== Auto visualize theo config ====
#     # viz_cfg = config.get('visualize', {})
#     # if viz_cfg.get('enable', False):
#     #     # BVP (GT vs Pred)
#     #     if viz_cfg.get('bvp', {}).get('enable', False):
#     #         eval_cfg = config.get('eval', {})
#     #         fs     = eval_cfg.get('fs', 30)
#     #         bpmmin = eval_cfg.get('bpmmin', 40)
#     #         bpmmax = eval_cfg.get('bpmmax', 180)

#     #         out_dir_bvp   = os.path.join(save_dir, viz_cfg['bvp'].get('out_dir', 'waves'))
#     #         max_frames    = viz_cfg['bvp'].get('max_frames', 1000)
#     #         do_processing = viz_cfg['bvp'].get('do_processing', True)

#     #         os.makedirs(out_dir_bvp, exist_ok=True)
#     #         model.plot_test_waves(
#     #             out_dir=out_dir_bvp,
#     #             max_frames=max_frames,
#     #             fs=fs, bpmmin=bpmmin, bpmmax=bpmmax,
#     #             do_processing=do_processing
#     #         )

#     #     # Heatmap attention của Swin
#     #     if viz_cfg.get('attn', {}).get('enable', False):
#     #         out_dir_attn = os.path.join(save_dir, viz_cfg['attn'].get('out_dir', 'attn'))
#     #         os.makedirs(out_dir_attn, exist_ok=True)
#     #         # model.visualize_attention_test dùng self.config['attention_output_dir'] nếu có
#     #         model.config['attention_output_dir'] = out_dir_attn
#     #         model.visualize_attention_test()

#     _ = benchmark_inference(model, iters=20)
# main.py — latency + GPU mem realtime + GFLOPs (robust CPU/GPU)
import os
import yaml
import numpy as np
from argparse import ArgumentParser
from functools import partial
from time import perf_counter

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint

from torch_geometric.nn import RGCNConv, TransformerConv
from model import Model

# ---------------- FLOPs hooks ----------------
def _conv_hook(module, inp, out):
    b = inp[0].shape[0]
    out_dims = list(out.shape[2:])
    k = list(module.kernel_size)
    in_ch, out_ch, g = module.in_channels, module.out_channels, module.groups
    filters_per_channel = out_ch // g
    conv_per_pos = np.prod(k) * in_ch * filters_per_channel
    active = b * np.prod(out_dims)
    bias = out_ch * active if module.bias is not None else 0
    module.__flops__ += int(conv_per_pos * active + bias)

def _linear_hook(module, inp, out):
    x = inp[0]
    out_last = out.shape[-1]
    bias = out_last if module.bias is not None else 0
    module.__flops__ += int(np.prod(x.shape) * out_last + bias)

def _bn_hook(module, inp, out):
    n = np.prod(inp[0].shape)
    if module.affine: n *= 2
    module.__flops__ += int(n)

def _relu_hook(module, inp, out):
    module.__flops__ += int(out.numel())

def _pool_hook(module, inp, out):
    module.__flops__ += int(np.prod(inp[0].shape))

def _rgcn_hook(module, inp, out):
    edge_index = inp[1]
    num_edges = edge_index.size(1)
    module.__flops__ += int(num_edges * module.out_channels)

def _tconv_hook(module, inp, out):
    edge_index = inp[1]
    num_edges = edge_index.size(1)
    heads = module.heads
    head_dim = module.out_channels // heads
    extra = heads * (4 * num_edges * head_dim + 5 * num_edges)
    module.__flops__ += int(extra)

FLOP_HOOKS = {
    nn.Conv1d: _conv_hook, nn.Conv2d: _conv_hook, nn.Conv3d: _conv_hook,
    nn.Linear: _linear_hook,
    nn.BatchNorm1d: _bn_hook, nn.BatchNorm2d: _bn_hook, nn.BatchNorm3d: _bn_hook,
    nn.ReLU: _relu_hook, nn.ReLU6: _relu_hook, nn.LeakyReLU: _relu_hook,
    nn.MaxPool1d: _pool_hook, nn.MaxPool2d: _pool_hook, nn.MaxPool3d: _pool_hook,
    nn.AvgPool1d: _pool_hook, nn.AvgPool2d: _pool_hook, nn.AvgPool3d: _pool_hook,
    nn.MultiheadAttention: _linear_hook,  # gần đúng
    RGCNConv: _rgcn_hook,
    TransformerConv: _tconv_hook,
}

def _add_flops_hooks(model):
    handles = []
    def _add(m):
        h = FLOP_HOOKS.get(type(m))
        if h is not None:
            if not hasattr(m, "__flops__"):
                m.register_buffer("__flops__", torch.zeros(1, dtype=torch.int64))
            handles.append(m.register_forward_hook(h))
    model.apply(_add)
    return handles

def _reset_flops(model):
    for m in model.modules():
        if hasattr(m, "__flops__"):
            m.__flops__.zero_()

def _remove_hooks(handles):
    for h in handles:
        h.remove()

def compute_gmacs(model, example_input):
    model.eval()
    handles = _add_flops_hooks(model)
    _reset_flops(model)
    with torch.no_grad():
        _ = model(example_input)
        dev = example_input.device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    total = 0
    for m in model.modules():
        if hasattr(m, "__flops__"):
            total += m.__flops__.item()
    _remove_hooks(handles)
    return total / 1e9

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- Realtime callback ----------------
class GPULatencyMemoryMonitor(pl.Callback):
    def __init__(self, log_every_n_steps: int = 1):
        self.log_every_n_steps = log_every_n_steps
        self._t0 = None

    @staticmethod
    def _device(m):
        try: return next(m.parameters()).device
        except StopIteration: return torch.device("cpu")

    def _start(self, pl_module):
        self._t0 = perf_counter()
        dev = self._device(pl_module)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

    def _end(self, pl_module, tag):
        t_ms = (perf_counter() - self._t0) * 1000.0
        pl_module.log(f"{tag}/batch_latency_ms", t_ms, on_step=True, on_epoch=False)
        dev = self._device(pl_module)
        if dev.type == "cuda":
            alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
            reserv = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
            pl_module.log(f"{tag}/gpu_mem_alloc_mb", alloc, on_step=True, on_epoch=False)
            pl_module.log(f"{tag}/gpu_mem_reserved_mb", reserv, on_step=True, on_epoch=False)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): self._start(pl_module)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.log_every_n_steps == 0: self._end(pl_module, "train")
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0): self._start(pl_module)
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0): self._end(pl_module, "val")
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0): self._start(pl_module)
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0): self._end(pl_module, "test")

# ---------------- Helpers ----------------
def get_callback_cls(name: str):
    name = name.lower()
    if name == 'devicestatsmonitor':
        return partial(
            DeviceStatsMonitor,
            cpu_stats=True,
        )
    elif name == 'learningratemonitor':
        return LearningRateMonitor
    elif name == 'modelcheckpoint':
        return partial(
            ModelCheckpoint,
            save_last=True,
            save_top_k=-1,         
            every_n_epochs=1,     
        )
    elif name == 'valckpt':
        return partial(
            ModelCheckpoint,
            save_top_k=1,
            monitor='val/0/bpm/RMSE',
            mode='min',  
        )
    raise ValueError(f'Unknown callback: {name}')


def _model_device(m):
    try: return next(m.parameters()).device
    except StopIteration: return torch.device("cpu")

def _build_example_frames(cfg, device):
    # Ưu tiên đọc từ config; fallback mặc định
    T = int(cfg.get('data', {}).get('chunk_length', 180))
    res = int(cfg.get('model', {}).get('hparams', {}).get('input_resolution', 128))
    C = 3
    x = torch.randn(1, T, C, res, res, device=device)
    return x

def benchmark_inference(pl_model, frames=None, iters: int = 20):
    pl_model.eval()
    dev = _model_device(pl_model)
    amp = torch.cuda.is_available() and dev.type == "cuda"

    if frames is None:
        try:
            loader = pl_model.test_dataloader()[0]
            frames, _, _ = next(iter(loader))
            frames = frames.to(dev, non_blocking=True)
        except Exception:
            frames = _build_example_frames(pl_model.config, dev)

    with torch.no_grad():
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _ = pl_model(frames)
                _ = pl_model.predict(frames)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _ = pl_model(frames)
                _ = pl_model.predict(frames)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times.append((perf_counter() - t0) * 1000.0)

    lat_ms = float(sum(times) / len(times))
    lat_std = float((sum((t - lat_ms) ** 2 for t in times) / len(times)) ** 0.5)

    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
        alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        reserv = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)
    else:
        gpu_name, alloc, reserv = "cpu", 0.0, 0.0

    print(f"[Benchmark] device={gpu_name} | latency_ms/seq={lat_ms:.2f} ± {lat_std:.2f} | alloc={alloc:.1f}MB reserved={reserv:.1f}MB")
    return {"device": gpu_name, "latency_ms": lat_ms, "latency_std_ms": lat_std, "alloc_mb": alloc, "reserved_mb": reserv, "frames": frames}

# ---------------- Main ----------------
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split_idx', type=int, required=True)
    args = parser.parse_args()

    run_name = os.path.split(args.config)[-1].split('.')[0]
    proj_name = run_name.split('_')[-1]
    save_dir = os.path.join('logs', run_name); os.makedirs(save_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['split_idx'] = args.split_idx

    pl.seed_everything(config['seed'], workers=True)

    model = Model(config)
    dev = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model.to(dev)

    logger = WandbLogger(name=run_name + f'_fold{args.split_idx}', save_dir=save_dir, project=proj_name, log_model=True)

    callbacks = [get_callback_cls(n)() for n in config['trainer']['callbacks']]
    callbacks.append(GPULatencyMemoryMonitor(log_every_n_steps=1))

    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=config['trainer']['max_epochs'],
        deterministic='warn',
        logger=logger,
        callbacks=callbacks,
        default_root_dir=save_dir,
    )

    # Nếu muốn train/test, mở 2 dòng dưới:
    trainer.fit(model)
    trainer.test(model, ckpt_path='best')

    # ==== Auto visualize theo config ====
    viz_cfg = config.get('visualize', {})

    if viz_cfg.get('enable', False):

        # === BVP (GT vs Pred) ===
        if viz_cfg.get('bvp', {}).get('enable', False):
            eval_cfg = config.get('eval', {})
            fs = eval_cfg.get('fs', 30)
            bpmmin = eval_cfg.get('bpmmin', 40)
            bpmmax = eval_cfg.get('bpmmax', 180)
            out_dir_bvp = os.path.join(save_dir, viz_cfg['bvp'].get('out_dir', 'waves'))
            max_frames = viz_cfg['bvp'].get('max_frames', 1000)
            do_processing = viz_cfg['bvp'].get('do_processing', True)
            os.makedirs(out_dir_bvp, exist_ok=True)

            model.plot_test_waves(
                out_dir=out_dir_bvp,
                max_frames=max_frames,
                fs=fs,
                bpmmin=bpmmin,
                bpmmax=bpmmax,
                do_processing=do_processing
            )

        # === Heatmap attention của Swin ===
        if viz_cfg.get('attn', {}).get('enable', False):
            out_dir_attn = os.path.join(save_dir, viz_cfg['attn'].get('out_dir', 'attn'))
            os.makedirs(out_dir_attn, exist_ok=True)

            # model.visualize_attention_test dùng self.config['attention_output_dir'] nếu có
            model.config['attention_output_dir'] = out_dir_attn
            model.visualize_attention_test()
    # # Benchmark (tự fallback dummy nếu không có loader)
    # bench = benchmark_inference(model, iters=20)
    # frames = bench["frames"]

    # # GFLOPs (dùng đúng input ví dụ, cùng device)
    # gmacs = compute_gmacs(model, frames)
    # params_m = count_parameters(model) / 1e6
    # print(f"[Model] Params={params_m:.2f}M | MACs={gmacs:.2f} G")
