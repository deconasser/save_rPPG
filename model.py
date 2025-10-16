import cv2
import gc
import os
import random
import torch
import numpy as np
import networkx as nx
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from datasets import get_dataset_cls, collate_fn
from models import get_model_cls
from losses import get_loss_cls
from metrics import calculate_hr_and_hrv_metrics
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from metrics import preprocess, calculate_hr_and_hrv  # Dùng preprocess trong hr.py

def get_wd_params(module):
    """Weight decay is only applied to a part of the params.
    https://github.com/karpathy/minGPT   

    Args:
        module (Module): torch.nn.Module

    Returns:
        optim_groups: Separated parameters
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.MultiheadAttention)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif 'time_mix' in pn:
                decay.add(fpn)
            else:
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad], "weight_decay": 0.0},
    ]
    
    return optim_groups

def show_mask_on_image(img, mask):
    """
    Hàm trực quan hóa heatmap lên ảnh gốc.
    Args:
        img (numpy.ndarray): Ảnh gốc với giá trị pixel từ 0 đến 255.
        mask (numpy.ndarray): Heatmap đã được chuẩn hóa về [0, 1].
    Returns:
        cam (numpy.ndarray): Ảnh overlay (đã chuyển sang uint8).
        heatmap (numpy.ndarray): Heatmap màu theo colormap HSV.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap
def show_mask_on_image(
    img, 
    mask, 
    alpha: float = 0.55,            # độ hòa trộn heatmap
    cmap: str = "jet",              # "jet" (blue→red) hoặc "turbo"
    gamma: float = 0.85,            # tăng tương phản vùng nóng
    sharpen_amount: float = 1.1,    # làm nét nhẹ
    use_percentile: bool = True     # co giãn theo percentile để tránh outlier
):
    """
    Trực quan hóa heatmap lên ảnh gốc với màu sắc đậm & nét kiểu paper.
    - img: H x W x 3, dtype uint8 hoặc float, trong [0,255] hoặc [0,1]
    - mask: H x W hoặc h x w, float, bất kỳ dải; sẽ được normalize về [0,1]
    Trả về:
    - overlay_rgb_uint8: ảnh overlay (uint8)
    - heatmap_rgb_uint8: heatmap màu (uint8)
    """

    # 1) Ảnh về [0,1]
    img01 = img.astype(np.float32)
    if img01.max() > 1.5:  # nếu là [0,255]
        img01 /= 255.0
    img01 = np.clip(img01, 0.0, 1.0)

    # 2) Chuẩn hóa mask về [0,1] (có percentile để tránh outlier)
    m = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if m.size == 0:
        m = np.zeros_like(img01[..., 0], dtype=np.float32)
    m -= m.min()
    if use_percentile:
        lo, hi = np.percentile(m, 5), np.percentile(m, 99)
        m = (m - lo) / (hi - lo + 1e-8)
    else:
        m = m / (m.max() + 1e-8)
    m = np.clip(m, 0.0, 1.0)

    # 3) Làm nét + gamma cho “nổi khối”
    if sharpen_amount > 0:
        blur = cv2.GaussianBlur(m, (0, 0), 0.8)
        m = np.clip(m * (1 + sharpen_amount) - blur * sharpen_amount, 0.0, 1.0)
    if gamma != 1.0:
        m = np.power(m, gamma)

    # 4) Ánh xạ màu (JET = xanh đậm → đỏ). Fallback TURBO nếu muốn.
    if cmap.lower() == "turbo" and hasattr(cv2, "COLORMAP_TURBO"):
        cmap_id = cv2.COLORMAP_TURBO
    else:
        cmap_id = cv2.COLORMAP_JET
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cmap_id)   # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 5) Overlay chuẩn theo alpha
    overlay = (1 - alpha) * img01 + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)

    return (overlay * 255).astype(np.uint8), (heat * 255).astype(np.uint8)


def get_optimizer_cls(name: str):
    name = name.lower()
    if name == 'adamw':
        return optim.AdamW
    raise ValueError(f'Unknown optimizer: {name}')

# CUTMIX
def temporal_cutmix(x, y, cutmix_prob=0.5, cutmix_ratio_range=(0.25, 0.5)):
    if random.random() > cutmix_prob:
        return x, y

    n, d, c, h, w = x.shape
    idx = torch.randperm(n)
    x_other = x[idx]
    y_other = y[idx]

    min_ratio, max_ratio = cutmix_ratio_range
    cut_length = random.randint(int(d * min_ratio), int(d * max_ratio))
    start_idx = random.randint(0, d - cut_length)

    # Tạo mask cho dữ liệu video
    mask = torch.ones_like(x)
    mask[:, start_idx:start_idx + cut_length, :, :, :] = 0

    x_cutmix = x * mask + x_other * (1 - mask)

    # Áp dụng CutMix chỉ cho phần nhãn tương ứng
    y_cutmix = y.clone()
    y_cutmix[:, start_idx:start_idx+cut_length, :] = y_other[:, start_idx:start_idx+cut_length, :]

    return x_cutmix, y_cutmix

# MIXUP
def temporal_mixup(x, y, mixup_prob=0.5, mixup_ratio_range=(0.25, 0.5), mixup_alpha=0.2):

    if random.random() > mixup_prob:
        return x, y

    n, d, c, h, w = x.shape
    # Lấy một permutation ngẫu nhiên của batch
    idx = torch.randperm(n)
    x_other = x[idx]
    y_other = y[idx]

    # Xác định độ dài đoạn mixup
    min_ratio, max_ratio = mixup_ratio_range
    mix_length = random.randint(int(d * min_ratio), int(d * max_ratio))
    start_idx = random.randint(0, d - mix_length)

    # Lấy hệ số mixup từ phân phối Beta (hoặc có thể dùng random.random() cho đơn giản)
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    
    # Thực hiện mixup trên đoạn được chọn
    x_mixup = x.clone()
    x_mixup[:, start_idx:start_idx + mix_length, :, :, :] = (
        lam * x[:, start_idx:start_idx + mix_length, :, :, :] +
        (1 - lam) * x_other[:, start_idx:start_idx + mix_length, :, :, :]
    )
    
    # Tương tự với nhãn
    y_mixup = y.clone()
    y_mixup[:, start_idx:start_idx + mix_length, :] = (
        lam * y[:, start_idx:start_idx + mix_length, :] +
        (1 - lam) * y_other[:, start_idx:start_idx + mix_length, :]
    )
    
    return x_mixup, y_mixup

# from scipy.signal import savgol_filter

# def smooth_signal(signal, window_length=21, polyorder=3):
#     import numpy as np
#     from scipy.signal import savgol_filter
    
#     # Đảm bảo signal là mảng 1D
#     signal = np.ravel(signal)

#     # Nếu signal quá ngắn hoặc polyorder >= window_length => bỏ qua smoothing
#     if len(signal) < window_length or polyorder >= window_length:
#         return signal

#     # Nếu đủ điều kiện, áp dụng Savitzky-Golay filter
#     smoothed = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
#     return smoothed



class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        hparams = config["model"]["hparams"]
        self.max_epochs = config['trainer']['max_epochs']
        model_cls = get_model_cls(config['model']['name'])
        self.model = model_cls(**hparams)
        
        self.loss_names = [params['name'] for params in config['loss']]
        self.loss_weight_bases = [params['weight'] for params in config['loss']]
        self.loss_weight_exps = [params.get('exp', 1.0) for params in config['loss']]
        self.losses = nn.ModuleList([get_loss_cls(params['name'])() for params in config['loss']])
        self.test_predictions_final = None
        self.test_ground_truths_final = None

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def on_train_epoch_start(self) -> None:
        self.loss_weights = [base * (exp ** (self.current_epoch / self.max_epochs)) for base, exp in zip(self.loss_weight_bases, self.loss_weight_exps)]
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx):
        frames, waves, data = batch
        # # Áp dụng Temporal CutMix chỉ trong quá trình training 
        if self.training:
            # Đọc từ config (đã lưu trong self.config ở __init__)
            cm_prob  = self.config["model"]["hparams"].get("cutmix_prob", 0.4)
            cm_range = tuple(self.config["model"]["hparams"].get("cutmix_ratio_range", (0.25, 0.5)))
            frames, waves = temporal_cutmix(
                frames, waves,
                cutmix_prob=cm_prob,
                cutmix_ratio_range=cm_range
            )
    
        predictions = self(frames)
        loss = 0.
        for loss_name, crit, weight in zip(self.loss_names, self.losses, self.loss_weights):
            loss_value = crit(predictions, waves)
            self.log(f'train/{loss_name}', loss_value, prog_bar=True)
            loss = loss_value * weight + loss
        
        self.log('train/loss', loss, prog_bar=True)
        return loss

    # ------------------- VALIDATION CODE BẮT ĐẦU -------------------

    def on_validation_epoch_start(self):
        self.loss_weights = [base * (exp ** (self.current_epoch / self.max_epochs)) for base, exp in zip(self.loss_weight_bases, self.loss_weight_exps)]
        # Tạo bộ nhớ tích lũy cho dự đoán và ground truth của validation
        self.validation_predictions = {}
        self.validation_ground_truths = {}
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        frames, waves, data = batch
        # Bước 1: Tính dự đoán dưới dạng tensor để tính loss
        predictions_tensor = self.predict(frames)
        
        # Tính loss với tensor (không chuyển sang numpy)
        loss = 0.
        for loss_name, crit, weight in zip(self.loss_names, self.losses, self.loss_weights):
            loss_value = crit(predictions_tensor, waves)
            loss += loss_value * weight
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        # Bước 2: Chuyển kết quả dự đoán sang numpy để lưu lại cho metric
        predictions = predictions_tensor.detach().cpu().numpy()
        batch_size = frames.shape[0]
        
        # Bước 3: Đảm bảo dictionary cho dataloader_idx đã được khởi tạo
        if dataloader_idx not in self.validation_predictions:
            self.validation_predictions[dataloader_idx] = {}
            self.validation_ground_truths[dataloader_idx] = {}
        
        # Lặp qua từng phần tử trong batch
        for i in range(batch_size):
            metadata = data[i]
            subject = metadata['subject']
            record = metadata['record']
            idx = metadata['idx']
            
            # Đảm bảo key subject tồn tại bên trong dictionary của dataloader_idx
            if subject not in self.validation_predictions[dataloader_idx]:
                self.validation_predictions[dataloader_idx][subject] = {}
                self.validation_ground_truths[dataloader_idx][subject] = {}
            # Đảm bảo key record tồn tại bên trong dictionary của subject
            if record not in self.validation_predictions[dataloader_idx][subject]:
                self.validation_predictions[dataloader_idx][subject][record] = {}
                self.validation_ground_truths[dataloader_idx][subject][record] = {}
            
            self.validation_predictions[dataloader_idx][subject][record][idx] = predictions[i]
            # Giả sử dữ liệu ground truth cho validation cũng có key 'waves'
            self.validation_ground_truths[dataloader_idx][subject][record][idx] = data[i]['waves'].detach().cpu().numpy()
        
        return loss


    def on_validation_epoch_end(self):
        for dataloader_id in self.validation_predictions.keys():
            predictions = []
            ground_truths = []
            dataloader_predictions = self.validation_predictions[dataloader_id]
            dataloader_ground_truths = self.validation_ground_truths[dataloader_id]

            for subject in dataloader_predictions.keys():
                pred_subj = dataloader_predictions[subject]
                gt_subj = dataloader_ground_truths[subject]
                for record in pred_subj.keys():
                    pred_rec = pred_subj[record]
                    gt_rec = gt_subj[record]
                    pred_ = []
                    gt_ = []
                    for i, idx in enumerate(sorted(pred_rec.keys())):
                        pred = pred_rec[idx]
                        gt = gt_rec[idx]
                        if i > 0:
                            pred = pred[-self.config['data']['chunk_interval']:]
                            gt = gt[-self.config['data']['chunk_interval']:]
                        pred_.append(pred)
                        gt_.append(gt)
                    pred_ = np.concatenate(pred_, axis=0)
                    gt_ = np.concatenate(gt_, axis=0)
                    predictions.append(pred_)
                    ground_truths.append(gt_)

            metrics = calculate_hr_and_hrv_metrics(
                predictions, 
                ground_truths, 
                diff='diff' in self.config['data']['wave_type'][0]
            )
            for metric_name, metric_value in metrics.items():
                self.log(f'val/{dataloader_id}/{metric_name}', metric_value, prog_bar='bpm' in metric_name)
        
        log_file ="14_10_2025.log"

        with open(log_file, "a") as f:  # "a" để ghi tiếp vào file mà không ghi đè
            f.write(f"\n=== Name: {self.config['model']['name']} - Epoch {self.current_epoch} Validation Metrics ===\n")
            for key, value in self.trainer.callback_metrics.items():
                if key.startswith("val/"):
                    f.write(f"{key}: {value}\n")
                
        self.validation_predictions = {}
        self.validation_ground_truths = {}
        gc.collect()
        return super().on_validation_epoch_end()

    # ------------------- VALIDATION CODE KẾT THÚC -------------------

    # ------------------- TEST CODE BẮT ĐẦU -------------------

    def on_test_epoch_start(self):
        self.predictions = {}
        self.ground_truths = {}
        return super().on_test_epoch_start()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        frames, waves, data = batch
        predictions = self.predict(frames).detach().cpu().numpy()
        batch_size = frames.shape[0]
        for i in range(batch_size):

            metadata = data[i]
            subject = metadata['subject']
            record = metadata['record']
            idx = metadata['idx']
            
            if dataloader_idx not in self.predictions:
                self.predictions[dataloader_idx] = {}
                self.ground_truths[dataloader_idx] = {}
            
            if subject not in self.predictions[dataloader_idx]:
                self.predictions[dataloader_idx][subject] = {}
                self.ground_truths[dataloader_idx][subject] = {}
            
            if record not in self.predictions[dataloader_idx][subject]:
                self.predictions[dataloader_idx][subject][record] = {}
                self.ground_truths[dataloader_idx][subject][record] = {}
            
            self.predictions[dataloader_idx][subject][record][idx] = predictions[i]
            self.ground_truths[dataloader_idx][subject][record][idx] = data[i]['waves'].detach().cpu().numpy()
            
        return

    def on_test_epoch_end(self):
        """
        Sau khi test xong:
        1) Ghép các chunk theo subject/record -> tín hiệu liên tục
        2) Tính HR_pred & HR_gt cho từng subject/record
        3) Ghi tất cả (HR + toàn bộ chuỗi BVP pred/gt) vào CÙNG MỘT FILE LOG
        4) Tính metrics tổng như cũ
        """
        from datetime import datetime

        hr_log_file = self.config.get('hr_log_file', 'test_hr_subjects.log')
        os.makedirs(os.path.dirname(hr_log_file) if os.path.dirname(hr_log_file) else ".", exist_ok=True)

        eval_cfg = self.config.get('eval', {})
        fs      = eval_cfg.get('fs', 30)
        bpmmin  = eval_cfg.get('bpmmin', 40)
        bpmmax  = eval_cfg.get('bpmmax', 180)

        all_lines = []

        for dataloader_id in self.predictions.keys():
            predictions = []
            ground_truths = []

            dataloader_predictions   = self.predictions[dataloader_id]
            dataloader_ground_truths = self.ground_truths[dataloader_id]

            for subject in dataloader_predictions.keys():
                pred_subj = dataloader_predictions[subject]
                gt_subj   = dataloader_ground_truths[subject]

                for record in pred_subj.keys():
                    pred_rec = pred_subj[record]
                    gt_rec   = gt_subj[record]

                    pred_ = []
                    gt_   = []
                    for i, idx in enumerate(sorted(pred_rec.keys())):
                        p_chunk = pred_rec[idx]
                        g_chunk = gt_rec[idx]
                        if i > 0:
                            p_chunk = p_chunk[-self.config['data']['chunk_interval']:]
                            g_chunk = g_chunk[-self.config['data']['chunk_interval']:]
                        pred_.append(p_chunk)
                        gt_.append(g_chunk)

                    p_full = np.concatenate(pred_, axis=0)
                    g_full = np.concatenate(gt_,   axis=0)

                    predictions.append(p_full)
                    ground_truths.append(g_full)

                    # Tính HR
                    try:
                        measures_pred = calculate_hr_and_hrv(p_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
                        measures_gt   = calculate_hr_and_hrv(g_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
                        hr_pred = float(measures_pred[0])
                        hr_gt   = float(measures_gt[0])
                        err_abs = abs(hr_pred - hr_gt)
                    except Exception as e:
                        hr_pred = float('nan')
                        hr_gt   = float('nan')
                        err_abs = float('nan')

                    # Chuỗi BVP đưa về 1D và format 6 chữ số thập phân
                    p_1d = np.asarray(p_full).reshape(-1)
                    g_1d = np.asarray(g_full).reshape(-1)
                    min_len = min(len(p_1d), len(g_1d))
                    p_1d = p_1d[:min_len]
                    g_1d = g_1d[:min_len]

                    pred_str = ",".join(f"{v:.6f}" for v in p_1d.tolist())
                    gt_str   = ",".join(f"{v:.6f}" for v in g_1d.tolist())

                    block = [
                        "-" * 80,
                        f"{datetime.now().isoformat()} | dl={dataloader_id} | subject={subject} | record={record}",
                        f"HR_gt={hr_gt:.2f} | HR_pred={hr_pred:.2f} | |err|={err_abs:.2f} | len={min_len}",
                        f"BVP_pred=[{pred_str}]",
                        f"BVP_gt=[{gt_str}]",
                    ]
                    all_lines.extend(block)

            # Metrics tổng (như cũ)
            metrics = calculate_hr_and_hrv_metrics(
                predictions,
                ground_truths,
                diff=('diff' in self.config['data']['wave_type'][0])
            )
            for metric_name, metric_value in metrics.items():
                self.log(f'test/{dataloader_id}/{metric_name}', metric_value, prog_bar=('bpm' in metric_name))

        # Ghi tất cả vào CÙNG MỘT FILE LOG
        with open(hr_log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Test HR & BVP detail – {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n")
            for line in all_lines:
                f.write(line + "\n")

        self.test_predictions_final = self.predictions
        self.test_ground_truths_final = self.ground_truths
        self.predictions = {}
        self.ground_truths = {}
        gc.collect()

        return super().on_test_epoch_end()

   
    # def on_test_epoch_end(self):

    #     #if self.config.get("visualize_attention", False):
    #         # self.visualize_attention_test()
    #         # self.visualize_attention_comparison()
    #         # self.visualize_graph_attention_test()

    #     for dataloader_id in self.predictions.keys():
    #         predictions = []
    #         ground_truths = []
    #         dataloader_predictions = self.predictions[dataloader_id]
    #         dataloader_ground_truths = self.ground_truths[dataloader_id]
        
    #         for subject in dataloader_predictions.keys():
    #             pred_subj = dataloader_predictions[subject]
    #             gt_subj = dataloader_ground_truths[subject]
    #             for record in pred_subj.keys():
    #                 pred_rec = pred_subj[record]
    #                 gt_rec = gt_subj[record]
    #                 pred_ = []
    #                 gt_ = []
    #                 for i, idx in enumerate(sorted(pred_rec.keys())):
    #                     pred = pred_rec[idx]
    #                     gt = gt_rec[idx]
    #                     if i > 0:
    #                         pred = pred[-self.config['data']['chunk_interval']:]
    #                         gt = gt[-self.config['data']['chunk_interval']:]
    #                     pred_.append(pred)
    #                     gt_.append(gt)
    #                 pred_ = np.concatenate(pred_, axis=0)
    #                 gt_ = np.concatenate(gt_, axis=0)
    #                 predictions.append(pred_)
    #                 ground_truths.append(gt_)
            
    #         metrics = calculate_hr_and_hrv_metrics(predictions, ground_truths, diff='diff' in self.config['data']['wave_type'][0])
    #         for metric_name, metric_value in metrics.items():
    #             self.log(f'test/{dataloader_id}/{metric_name}', metric_value, prog_bar='bpm' in metric_name)

    #     self.test_predictions_final = self.predictions
    #     self.test_ground_truths_final = self.ground_truths
    #     self.predictions = {}
    #     self.ground_truths = {}
    #     gc.collect()

    #     return super().on_test_epoch_end()

    # def plot_test_waves(self, out_dir='wave_plots',
    #                     max_frames=1000,
    #                     fs=30, bpmmin=40, bpmmax=180,
    #                     do_processing=True,
    #                     do_smoothing=True):

    #     if self.test_predictions_final is None or self.test_ground_truths_final is None:
    #         print("Chưa có dữ liệu test_predictions_final. Hãy chắc chắn đã chạy test trước.")
    #         return

    #     os.makedirs(out_dir, exist_ok=True)
        
    #     hr_pred_list = []
    #     hr_gt_list = []
    #     # Duyệt qua từng dataloader
    #     for dataloader_id in self.test_predictions_final.keys():
    #         preds_dict = self.test_predictions_final[dataloader_id]
    #         gts_dict   = self.test_ground_truths_final[dataloader_id]

    #         # Duyệt qua từng subject
    #         for subject in preds_dict.keys():
    #             pred_subj = preds_dict[subject]
    #             gt_subj   = gts_dict[subject]

    #             # Duyệt qua từng record
    #             for record in pred_subj.keys():
    #                 p_dict = pred_subj[record]
    #                 g_dict = gt_subj[record]

    #                 # Ghép các chunk thành 1 sóng liên tục
    #                 p_full = []
    #                 g_full = []
    #                 sorted_indices = sorted(p_dict.keys())
    #                 for i, idx in enumerate(sorted_indices):
    #                     p_chunk = p_dict[idx]
    #                     g_chunk = g_dict[idx]
    #                     # Nếu có overlap giữa các chunk, cắt bớt phần dư
    #                     if i > 0:
    #                         p_chunk = p_chunk[-self.config['data']['chunk_interval']:]
    #                         g_chunk = g_chunk[-self.config['data']['chunk_interval']:]
    #                     p_full.append(p_chunk)
    #                     g_full.append(g_chunk)
                    
    #                 p_full = np.concatenate(p_full, axis=0)
    #                 g_full = np.concatenate(g_full, axis=0)

    #                 # Cắt tín hiệu chỉ vẽ tối đa max_frames
    #                 max_len = min(len(p_full), len(g_full), max_frames)
    #                 p_full = p_full[:max_len]
    #                 g_full = g_full[:max_len]

    #                 # Xử lý tín hiệu tương tự như trong tính toán HR/HRV
    #                 if do_processing:
    #                     # Bước 1: Preprocess (format, nan, detrend)
    #                     p_pre = preprocess(p_full, diff=False)
    #                     g_pre = preprocess(g_full, diff=False)
                        
    #                     # Bước 2: Lọc tín hiệu với bộ lọc bandpass
    #                     freq_min = bpmmin / 60.0
    #                     freq_max = bpmmax / 60.0
    #                     [b, a] = butter(3, [freq_min / fs * 2, freq_max / fs * 2], btype='bandpass')
    #                     p_processed = filtfilt(b, a, p_pre)
    #                     g_processed = filtfilt(b, a, g_pre)
    #                 else:
    #                     p_processed = p_full
    #                     g_processed = g_full

    #                 scale_factor = np.std(g_processed) / np.std(p_processed)
    #                 p_processed *= scale_factor

    #                 # (Tùy chọn) Làm mịn tín hiệu bằng Savitzky-Golay filter
    #                 # if do_smoothing:
    #                 #     window_length = 21 if len(p_processed) >= 21 else (len(p_processed) // 2) * 2 + 1
    #                 #     p_processed = savgol_filter(p_processed, window_length=window_length, polyorder=3)
    #                 #     g_processed = savgol_filter(g_processed, window_length=window_length, polyorder=3)
                    
    #                 # Vẽ biểu đồ
    #                 plt.figure(figsize=(8,4))
    #                 plt.plot(p_processed, label='Pred wave (processed)', color='red')
    #                 plt.plot(g_processed, label='GT wave (processed)', color='black')
    #                 plt.title(f'Subject={subject} | Record={record}')
    #                 plt.xlabel('Frame')
    #                 plt.ylabel('Amplitude')
    #                 plt.legend()
                    
    #                 # Lưu file PNG
    #                 filename = f"{subject}_{record}_wave.png"
    #                 plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    #                 plt.close()

    #                 print(f"Đã lưu file {filename} vào {out_dir}")
    #                 # (4) Tính BPM để lát nữa vẽ scatter
    #                 #    (measures[0] chính là BPM)
    #                 measures_pred = calculate_hr_and_hrv(p_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
    #                 measures_gt   = calculate_hr_and_hrv(g_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
    #                 hr_pred = measures_pred[0]  # BPM
    #                 hr_gt   = measures_gt[0]    # BPM

    #                 hr_pred_list.append(hr_pred)
    #                 hr_gt_list.append(hr_gt)

    def plot_test_waves(self, out_dir='wave_plots',
                        max_frames=1000,
                        fs=30, bpmmin=40, bpmmax=180,
                        do_processing=True,
                        do_smoothing=True,
                        psd_norm=True):
        """
        Vẽ 2 hình cho mỗi (subject, record):
        1) Time-domain: Pred vs GT (như hiện tại)
        2) Power Spectrum (Welch): Pred vs GT trên trục BPM, có vạch HR ước lượng
        """
        if self.test_predictions_final is None or self.test_ground_truths_final is None:
            print("Chưa có dữ liệu test_predictions_final. Hãy chắc chắn đã chạy test trước.")
            return

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import butter, filtfilt, welch

        os.makedirs(out_dir, exist_ok=True)

        # Màu & style
        COL_GT   = "#79a7ff"   # xanh dương
        COL_PRED = "#91D04F"   # xanh lá
        LW = 4.0
        ALPHA = 0.95

        def psd_bpm(x, fs, bpm_min, bpm_max, nperseg=None):
            import numpy as np
            from scipy.signal import welch

            # Ép về 1D + làm sạch NaN/Inf
            x = np.asarray(x, dtype=np.float64).ravel()
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Bảo vệ tín hiệu quá ngắn / hằng số
            if x.size < 8 or np.allclose(x, x[0], atol=1e-12):
                return np.array([]), np.array([]), float('nan')

            # Chọn nperseg hợp lệ
            if nperseg is None:
                nperseg = min(1024, x.size)
            nperseg = max(8, int(nperseg))  # tối thiểu vài điểm để ra phổ

            # Welch luôn trả f (1D) và Pxx (1D) khi x là 1D
            f, Pxx = welch(x, fs=float(fs), nperseg=nperseg)

            bpm = f * 60.0
            # Đảm bảo min < max
            lo, hi = (float(bpm_min), float(bpm_max)) if bpm_min < bpm_max else (float(bpm_max), float(bpm_min))
            mask = (bpm >= lo) & (bpm <= hi)

            bpm_axis = bpm[mask]
            psd_band = Pxx[mask]

            # Chuẩn hoá (nếu cần)
            if psd_norm and psd_band.size > 0:
                mx = psd_band.max()
                if mx > 0:
                    psd_band = psd_band / mx

            peak_bpm = float(bpm_axis[np.argmax(psd_band)]) if psd_band.size > 0 else float('nan')
            return bpm_axis, psd_band, peak_bpm

        hr_pred_list, hr_gt_list = [], []

        for dataloader_id in self.test_predictions_final.keys():
            preds_dict = self.test_predictions_final[dataloader_id]
            gts_dict   = self.test_ground_truths_final[dataloader_id]

            for subject in preds_dict.keys():
                pred_subj = preds_dict[subject]
                gt_subj   = gts_dict[subject]

                for record in pred_subj.keys():
                    p_dict = pred_subj[record]
                    g_dict = gt_subj[record]

                    # Ghép các chunk
                    p_full, g_full = [], []
                    sorted_indices = sorted(p_dict.keys())
                    for i, idx in enumerate(sorted_indices):
                        p_chunk = p_dict[idx]
                        g_chunk = g_dict[idx]
                        if i > 0:
                            p_chunk = p_chunk[-self.config['data']['chunk_interval']:]
                            g_chunk = g_chunk[-self.config['data']['chunk_interval']:]
                        p_full.append(p_chunk); g_full.append(g_chunk)

                    p_full = np.concatenate(p_full, axis=0)
                    g_full = np.concatenate(g_full, axis=0)

                    # Cắt tối đa
                    max_len = min(len(p_full), len(g_full), max_frames)
                    p_full = p_full[:max_len]; g_full = g_full[:max_len]

                    # Xử lý
                    if do_processing:
                        p_pre = preprocess(p_full, diff=False)
                        g_pre = preprocess(g_full, diff=False)
                        freq_min = bpmmin / 60.0
                        freq_max = bpmmax / 60.0
                        b, a = butter(3, [freq_min / fs * 2, freq_max / fs * 2], btype='bandpass')
                        p_processed = filtfilt(b, a, p_pre)
                        g_processed = filtfilt(b, a, g_pre)
                    else:
                        p_processed = p_full
                        g_processed = g_full

                    # Thu nhỏ biên độ để nhìn "dịu" hơn
                    AMP_SHRINK = 0.2
                    p_processed *= AMP_SHRINK
                    g_processed *= AMP_SHRINK

                    # Match scale (tránh chia 0)
                    stdp = np.std(p_processed)
                    if stdp > 1e-12:
                        p_processed = p_processed * (np.std(g_processed) / stdp)

                    # ================== FIGURE: Time-domain ==================
                    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                    g_line, = ax.plot(g_processed, lw=LW, color=COL_GT,   alpha=ALPHA,
                                    label="GT",   solid_capstyle="round", antialiased=True)
                    p_line, = ax.plot(p_processed, lw=LW, color=COL_PRED, alpha=ALPHA,
                                    label="Pred", solid_capstyle="round", antialiased=True)


                    ax.set_xticks([]); ax.set_yticks([])
                    ax.grid(False)

                    n = len(g_processed)
                    ax.set_xlim(-0.5, n - 0.5)
                    ymin = float(min(np.min(g_processed), np.min(p_processed)))
                    ymax = float(max(np.max(g_processed), np.max(p_processed)))
                    if ymax <= ymin:
                        ymax, ymin = ymin + 1.0, ymin - 1.0
                    pad = 0.07 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
                    ax.legend(handles=[p_line, g_line], labels=["Pred", "GT"],
                            loc="upper left", frameon=True)

                    plt.tight_layout()
                    filename = f"{subject}_{record}_wave.png"
                    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Đã lưu file {filename} vào {out_dir}")

                    # ================== FIGURE: Power Spectrum (Welch) ==================
                    bpm_axis_p, psd_p, peak_pred = psd_bpm(p_full, fs, bpmmin, bpmmax)
                    bpm_axis_g, psd_g, peak_gt   = psd_bpm(g_full, fs, bpmmin, bpmmax)

                    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
                    line_g, = ax2.plot(bpm_axis_g, psd_g, lw=LW, color=COL_GT,   alpha=ALPHA,
                                    label="GT PSD",   solid_capstyle="round", antialiased=True)
                    line_p, = ax2.plot(bpm_axis_p, psd_p, lw=LW, color=COL_PRED, alpha=ALPHA,
                                    label="Pred PSD", solid_capstyle="round", antialiased=True)
                    # Vạch dọc tại đỉnh (ước lượng HR)
                    if np.isfinite(peak_pred):
                        ax2.axvline(peak_pred, linestyle="--", linewidth=2, color=COL_PRED, alpha=0.8)
                    if np.isfinite(peak_gt):
                        ax2.axvline(peak_gt, linestyle="--", linewidth=2, color=COL_GT,   alpha=0.8)

                    # Trục & style
                    ax2.set_xlim(bpmmin, bpmmax)
                    ax2.set_xlabel("BPM")
                    ax2.set_ylabel("Power" if psd_norm else "Power")
                    ax2.grid(False)
                    # Ẩn tick y cho “sạch” giống time-domain (nếu muốn)
                    ax2.set_yticks([])
                    ax2.legend(loc="upper right", frameon=True)

                    plt.tight_layout()
                    filename_psd = f"{subject}_{record}_psd.png"
                    plt.savefig(os.path.join(out_dir, filename_psd), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Đã lưu file {filename_psd} vào {out_dir}")

                    # --------- HR cho thống kê khác (không hiển thị) ----------
                    measures_pred = calculate_hr_and_hrv(p_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
                    measures_gt   = calculate_hr_and_hrv(g_full, diff=False, fs=fs, bpmmin=bpmmin, bpmmax=bpmmax)
                    hr_pred_list.append(measures_pred[0])
                    hr_gt_list.append(measures_gt[0])



     # ===============================
        # CUỐI CÙNG: Vẽ scatter plot BPM
        # ===============================
        # Nếu len(hr_pred_list) > 0
        if len(hr_pred_list) > 0:
            hr_pred_array = np.array(hr_pred_list)
            hr_gt_array = np.array(hr_gt_list)

            # Lọc bỏ NaN (nếu có)
            mask = ~np.isnan(hr_pred_array) & ~np.isnan(hr_gt_array)
            hr_pred_array = hr_pred_array[mask]
            hr_gt_array = hr_gt_array[mask]

            plt.figure(figsize=(8, 4))

            # Sử dụng scatter với colormap
            scatter = plt.scatter(
                hr_gt_array, 
                hr_pred_array, 
                c=hr_gt_array,  # Màu dựa trên giá trị GT BPM
                cmap='viridis',  # Dùng colormap 'viridis'
                alpha=0.7, 
                edgecolors='k'
            )

            # Vẽ đường y = x
            min_val = min(hr_gt_array.min(), hr_pred_array.min())
            max_val = max(hr_gt_array.max(), hr_pred_array.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')

            plt.xlabel('GT PPG HR [bpm]')
            plt.ylabel('rPPG HR [bpm]')
            plt.title('Scatter Plot')

            # Thêm colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('GT PPG HR [bpm]')

            plt.grid(True)

            scatter_filename = 'scatter_hr.png'
            plt.savefig(os.path.join(out_dir, scatter_filename), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Đã lưu file scatter plot {scatter_filename} vào {out_dir}")
        else:
            print("Không có dữ liệu BPM để vẽ scatter plot.")

    # def visualize_attention_test(self):
    #     final_block = self.model.layers.spatial_blocks[-1]
    #     # final_block = self.model.layers.long_term_blocks[-1]
    #     hook_handle = final_block.register_forward_hook(Model.save_attention_hook)
        
    #     test_loader = self.test_dataloader()[0]
    #     batch = next(iter(test_loader))
    #     frames, waves, data = batch  
        
    #     # Lấy sample đầu tiên: (1, 180, C, H, W)
    #     sample = frames[0:1].to(self.device)
    #     _ = self(sample)  # Forward pass để hook lưu attention map
        
    #     # 3. Lấy attention map: expected shape (720, 8, 16, 16)
    #     attn_map = final_block.attn.last_attn
    #     if attn_map is None:
    #         print("Không lấy được attention map từ block cuối.")
    #         hook_handle.remove()
    #         return
        
    #     attn_avg = attn_map.mean(dim=1)
        
    #     # 5. Reshape thành (180, 4, 16, 16): 180 frame, mỗi frame có 4 window
    #     attn_avg = attn_avg.reshape(180, 4, 16, 16)
        
    #     output_dir = self.config.get('attention_output_dir', './attention_outputs_lsts')
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # 6. Xử lý từng frame
    #     for frame_idx in range(180):
    #         windows = attn_avg[frame_idx]  # shape: (4, 16, 16)
    #         window_heatmaps = []
    #         for win_idx in range(4):
    #             window = windows[win_idx]  # shape: (16, 16)
    #             # Trung bình theo chiều query (hàng) → vector (16,)
    #             patch_scores = window.mean(dim=0)  # shape: (16,)
    #             patch_scores = patch_scores.detach().cpu().numpy()
    #             # Chuẩn hóa về [0, 1]
    #             patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
    #             # Reshape thành 4x4 grid (giả sử 16 patch)
    #             window_heatmap = patch_scores.reshape(4, 4)
    #             window_heatmaps.append(window_heatmap)
            
    #         # 7. Ghép 4 window lại theo thứ tự: 
    #         #    0 - top-left, 1 - top-right, 2 - bottom-left, 3 - bottom-right
    #         top_row = np.concatenate([window_heatmaps[0], window_heatmaps[1]], axis=1)    # (4, 8)
    #         bottom_row = np.concatenate([window_heatmaps[2], window_heatmaps[3]], axis=1) # (4, 8)
    #         frame_heatmap = np.concatenate([top_row, bottom_row], axis=0)                 # (8, 8)
            
    #         # 8. Lấy ảnh gốc của frame từ sample tensor (shape: (1, 180, C, H, W))
    #         orig_frame = sample[0, frame_idx]  # shape: (C, H, W)
    #         orig_frame = orig_frame.permute(1, 2, 0).cpu().numpy()  # shape: (H, W, C) # đã ở khoảng [0,1]   
    #         resized_heatmap = cv2.resize(
    #             frame_heatmap, 
    #             (orig_frame.shape[1], orig_frame.shape[0]), 
    #             interpolation=cv2.INTER_CUBIC
    #         )

    #         resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())
    #         overlay, heatmap = show_mask_on_image(orig_frame, resized_heatmap)
    #         overlay_path = os.path.join(output_dir, f"attention_overlay_frame_{frame_idx:03d}.png")
    #         cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    #     print(f"Đã lưu overlay của 180 frame vào thư mục: {output_dir}")
    #     hook_handle.remove()

    def visualize_attention_test(self):
        # Lưu attention + frame gốc CHỈ cho block số 10, với TẤT CẢ các batch
        import os, cv2, torch
        import numpy as np

        blocks = self.model.layers.spatial_blocks
        test_loader = self.test_dataloader()[0]

        TARGET_BLOCK = 10  # <--- chỉ số block muốn lưu

        output_dir = self.config.get('attention_output_dir', './attention_outputs_lsts')
        os.makedirs(output_dir, exist_ok=True)

        if TARGET_BLOCK >= len(blocks):
            print(f"[Warn] TARGET_BLOCK={TARGET_BLOCK} >= số block ({len(blocks)}). Không có gì để lưu.")
            return

        was_training = self.training
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                frames, waves, data = batch
                # Lấy 1 sample để visualize: (1, T, C, H, W)
                sample = frames[0:1].to(self.device, non_blocking=True)
                T = sample.shape[1]

                # Forward 1 lần để các block cập nhật last_attn
                _ = self(sample)

                batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_output_dir, exist_ok=True)

                # --- CHỈ xử lý block số 10 ---
                block_idx = TARGET_BLOCK
                block = blocks[block_idx]

                attn_map = getattr(block.attn, "last_attn", None)
                if attn_map is None:
                    print(f"Không lấy được attention map từ block {block_idx} trong batch {batch_idx}.")
                    continue

                # [B*nW, H, N, N] -> trung bình head -> [B*nW, N, N]
                attn_avg = attn_map.mean(dim=1).detach()

                # Theo bố cục cũ: (T, 4, 16, 16). Nếu không khớp thì bỏ qua batch này.
                try:
                    attn_avg = attn_avg.reshape(T, 4, 16, 16)
                except Exception as e:
                    print(f"Không reshape được attention (block {block_idx}, batch {batch_idx}): {e}")
                    continue

                block_output_dir = os.path.join(batch_output_dir, f"block_{block_idx}")
                os.makedirs(block_output_dir, exist_ok=True)

                for frame_idx in range(T):
                    windows = attn_avg[frame_idx]  # (4, 16, 16)
                    window_heatmaps = []
                    for win_idx in range(4):
                        # Trung bình theo chiều query -> (16,)
                        patch_scores = windows[win_idx].mean(dim=0).detach().cpu().numpy()
                        pmin, pmax = patch_scores.min(), patch_scores.max()
                        if pmax - pmin < 1e-12:
                            patch_scores = np.zeros_like(patch_scores)
                        else:
                            patch_scores = (patch_scores - pmin) / (pmax - pmin)
                        window_heatmaps.append(patch_scores.reshape(4, 4))

                    # Ghép 4 window: 0 TL, 1 TR, 2 BL, 3 BR -> (8,8)
                    top = np.concatenate([window_heatmaps[0], window_heatmaps[1]], axis=1)
                    bot = np.concatenate([window_heatmaps[2], window_heatmaps[3]], axis=1)
                    frame_heatmap = np.concatenate([top, bot], axis=0)

                    # Ảnh gốc (H,W,C) [0,1] hoặc [0,255] (RGB)
                    orig_frame = sample[0, frame_idx].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                    if orig_frame.max() > 1.5:
                        orig_frame = orig_frame / 255.0
                    orig_rgb_u8 = (np.clip(orig_frame, 0, 1) * 255).round().astype(np.uint8)

                    # Lưu frame gốc
                    orig_path = os.path.join(block_output_dir, f"frame_{frame_idx:03d}_orig.png")
                    cv2.imwrite(orig_path, cv2.cvtColor(orig_rgb_u8, cv2.COLOR_RGB2BGR))

                    # Resize heatmap lên size ảnh gốc + chuẩn hoá
                    resized_heatmap = cv2.resize(
                        frame_heatmap,
                        (orig_rgb_u8.shape[1], orig_rgb_u8.shape[0]),
                        interpolation=cv2.INTER_CUBIC
                    )
                    rmin, rmax = resized_heatmap.min(), resized_heatmap.max()
                    if rmax - rmin > 1e-12:
                        resized_heatmap = (resized_heatmap - rmin) / (rmax - rmin)
                    else:
                        resized_heatmap = np.zeros_like(resized_heatmap)

                    # Overlay màu JET
                    overlay, _ = show_mask_on_image(
                        orig_rgb_u8, resized_heatmap,
                        alpha=0.55, cmap="jet", gamma=0.85, sharpen_amount=1.1, use_percentile=True
                    )
                    out_path = os.path.join(block_output_dir, f"attention_overlay_frame_{frame_idx:03d}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                print(f"Đã lưu overlay + frame gốc của {T} frame cho block {block_idx} trong batch {batch_idx} tại: {block_output_dir}")

        if was_training:
            self.train()


            
    # def visualize_attention_test(self):
    #     """
    #     Visualize attention cho nhiều subject:
    #     - Duyệt tất cả test loaders / batches / samples (có thể giới hạn qua config).
    #     - Suy ra động: T (frames), windows_per_frame, tokens_per_window (= Wh*Ww).
    #     - Không reshape cứng (180,4,16,16) nữa.
    #     """
    #     import math

    #     # Các thông số để tránh ghi quá nhiều file
    #     viz_cfg = self.config.get('visualize', {}).get('attn', {})
    #     output_dir = self.config.get('attention_output_dir', viz_cfg.get('out_dir', './attention_outputs_lsts'))
    #     max_loaders   = viz_cfg.get('max_loaders', None)      # None = tất cả
    #     max_batches   = viz_cfg.get('max_batches', None)      # None = tất cả
    #     max_samples   = viz_cfg.get('max_samples', None)      # None = tất cả
    #     max_frames    = viz_cfg.get('max_frames', None)       # None = tất cả
    #     frame_stride  = viz_cfg.get('frame_stride', 1)        # lấy mỗi k frame
    #     only_first_batch = viz_cfg.get('only_first_batch', False)  # tương thích kiểu cũ

    #     os.makedirs(output_dir, exist_ok=True)

    #     # Lấy vài thông số để ghép lưới window (suy từ model)
    #     # Giả định chuẩn: input_resolution chia hết cho patch_size và window_size cố định trong model
    #     # Nếu không, vẫn tính được từ attn shape (bên dưới).
    #     input_res   = getattr(self.model, 'input_resolution', 128)
    #     patch_size  = getattr(self.model, 'patch_size', (1,16,16))
    #     window_size = getattr(self.model, 'window_size', (1,4,4))
    #     # số patch theo H/W sau patch-embed
    #     H_tokens = math.ceil(input_res / patch_size[1])
    #     W_tokens = math.ceil(input_res / patch_size[2])
    #     # lưới window theo H/W sau chia window_size
    #     win_grid_h = math.ceil(H_tokens / window_size[1])
    #     win_grid_w = math.ceil(W_tokens / window_size[2])

    #     # Lấy các block swin
    #     blocks = self.model.layers.spatial_blocks

    #     was_training = self.training
    #     self.eval()

    #     test_loaders = self.test_dataloader()
    #     with torch.no_grad():
    #         for loader_idx, loader in enumerate(test_loaders):
    #             if max_loaders is not None and loader_idx >= max_loaders:
    #                 break

    #             for batch_idx, batch in enumerate(loader):
    #                 if only_first_batch and batch_idx > 0:
    #                     break
    #                 if max_batches is not None and batch_idx >= max_batches:
    #                     break

    #                 frames, waves, data = batch
    #                 B, T, C, H, W = frames.shape

    #                 for i in range(B):
    #                     if max_samples is not None and i >= max_samples:
    #                         break

    #                     # metadata để đặt tên thư mục
    #                     md = data[i]
    #                     subject = md.get('subject', f's{i}')
    #                     record  = md.get('record',  f'r{i}')

    #                     sample = frames[i:i+1].to(self.device, non_blocking=True)  # [1, T, C, H, W]
    #                     _ = self(sample)  # chạy forward 1 lần để cập nhật last_attn ở từng block

    #                     # lấy số frame thực tế và giới hạn theo config
    #                     T_eff = sample.shape[1]
    #                     if max_frames is not None:
    #                         T_eff = min(T_eff, max_frames)

    #                     sample_dir = os.path.join(
    #                         output_dir,
    #                         f"dl_{loader_idx}",
    #                         f"batch_{batch_idx}",
    #                         f"sample_{i}_sub-{subject}_rec-{record}"
    #                     )
    #                     os.makedirs(sample_dir, exist_ok=True)

    #                     # ảnh gốc để overlay (vì sample đã là [1,T,C,H,W])
    #                     sample_np = sample[0, :T_eff].detach().cpu().permute(0, 2, 3, 1).numpy()  # [T_eff, H, W, C]

    #                     for block_idx, block in enumerate(blocks):
    #                         attn_map = getattr(block.attn, "last_attn", None)
    #                         if attn_map is None:
    #                             print(f"[Warn] Không có attention ở block {block_idx} (loader={loader_idx}, batch={batch_idx}, sample={i}).")
    #                             continue

    #                         # attn_map: [B*nW, nHeads, N, N]  -> trung bình head -> [B*nW, N, N]
    #                         attn_avg = attn_map.mean(dim=1)  # [B*nW, N, N]
    #                         # Với B=1, tổng windows = attn_avg.shape[0]
    #                         total_windows = attn_avg.shape[0]

    #                         # tokens mỗi window (N = Wd*Wh*Ww). Với window_size=(1,4,4) => N = 16
    #                         tokens_per_win = attn_avg.shape[-1]
    #                         # suy ra số frame dùng trong attention (Wd=1 nên n_frame_attn ~ T_eff nếu không pad)
    #                         # an toàn: lấy min giữa T_eff và total_windows (tránh chia 0 nếu dữ liệu lạ)
    #                         frames_for_reshape = min(T_eff, total_windows)

    #                         # windows mỗi frame
    #                         windows_per_frame = total_windows // frames_for_reshape
    #                         if windows_per_frame == 0:
    #                             print(f"[Warn] windows_per_frame=0 tại block {block_idx}, bỏ qua.")
    #                             continue

    #                         # reshape: [total_windows, N, N] -> [frames_for_reshape, windows_per_frame, N, N]
    #                         attn_avg = attn_avg[:frames_for_reshape * windows_per_frame]
    #                         attn_avg = attn_avg.reshape(frames_for_reshape, windows_per_frame, tokens_per_win, tokens_per_win)

    #                         # kích thước lưới patch trong một window (giả định N = Wh*Ww và Wh, Ww nguyên)
    #                         pw = int(round(tokens_per_win ** 0.5))
    #                         ph = pw
    #                         if pw * ph != tokens_per_win:
    #                             # nếu không phải hình vuông, ta chỉ vẽ từng window riêng lẻ hoặc bỏ qua
    #                             print(f"[Warn] N={tokens_per_win} không là số chính phương, bỏ qua block {block_idx}.")
    #                             continue

    #                         # suy ra lưới window theo H/W: ưu tiên dùng win_grid_h/w từ cấu hình model,
    #                         # nếu không khớp thì fallback về sqrt (giả định vuông vức)
    #                         if win_grid_h * win_grid_w != windows_per_frame:
    #                             g = int(round(windows_per_frame ** 0.5))
    #                             win_h, win_w = g, max(1, windows_per_frame // max(1, g))
    #                         else:
    #                             win_h, win_w = win_grid_h, win_grid_w

    #                         block_dir = os.path.join(sample_dir, f"block_{block_idx}")
    #                         os.makedirs(block_dir, exist_ok=True)

    #                         for t in range(0, frames_for_reshape, max(1, frame_stride)):
    #                             windows_t = attn_avg[t]  # [windows_per_frame, N, N]
    #                             win_heatmaps = []

    #                             for widx in range(windows_per_frame):
    #                                 # trung bình theo chiều query -> vector (N,)
    #                                 patch_scores = windows_t[widx].mean(dim=0)  # [N]
    #                                 patch_scores = patch_scores.detach().cpu().numpy()
    #                                 patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
    #                                 win_heatmaps.append(patch_scores.reshape(ph, pw))  # [ph, pw]

    #                             # ghép lưới window: (win_h x win_w), mỗi ô là (ph x pw) -> ảnh (win_h*ph, win_w*pw)
    #                             rows = []
    #                             for rr in range(win_h):
    #                                 row = np.concatenate(win_heatmaps[rr*win_w:(rr+1)*win_w], axis=1)
    #                                 rows.append(row)
    #                             frame_heatmap = np.concatenate(rows, axis=0)  # [win_h*ph, win_w*pw]

    #                             # Overlay lên frame gốc
    #                             orig_frame = sample_np[t]  # [H, W, C] (0..1 nếu input chuẩn)
    #                             resized_heatmap = cv2.resize(frame_heatmap, (orig_frame.shape[1], orig_frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    #                             resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min() + 1e-8)

    #                             overlay, _ = show_mask_on_image(
    #                                 orig_frame, resized_heatmap,
    #                                 alpha=0.55, cmap="jet", gamma=0.85, sharpen_amount=1.1, use_percentile=True
    #                             )
    #                             out_path = os.path.join(block_dir, f"attn_overlay_t{t:03d}.png")
    #                             cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    #                     print(f"[OK] Saved attention overlays -> {sample_dir}")

    #                 # nếu bạn muốn vẫn dừng sớm theo kiểu cũ:
    #                 if only_first_batch:
    #                     break

    #     if was_training:
    #         self.train()


    # ------------------- TEST KẾT THÚC -------------------
    
    def train_dataloader(self):
        train_sets = []
        for args in self.config['data']['train_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            train_sets.append(dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config['split_idx'], training=True))
        train_set = ConcatDataset(train_sets)
        
        train_loader = DataLoader(
            train_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config['data']['num_workers']>0,
            collate_fn=collate_fn,
        )
        
        return train_loader

    def val_dataloader(self):
        val_loaders = []
        # Giả sử trong config có khai báo "val_sets" tương tự như "test_sets"
        for args in self.config['data'].get('val_sets', []):
            dataset_cls = get_dataset_cls(args['name'])
            val_set = dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config['split_idx'], training=False)
            val_loader = DataLoader(
                val_set,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=False,
                persistent_workers=False,
                collate_fn=collate_fn,
            )
            val_loaders.append(val_loader)
        return val_loaders

    def test_dataloader(self):
        test_loaders = []
        for args in self.config['data']['test_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            test_set = dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config['split_idx'], training=False)
            
            test_loader = DataLoader(
                test_set,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=False,
                persistent_workers=False,
                collate_fn=collate_fn,
            )
            test_loaders.append(test_loader)
        
        return test_loaders
    
    def configure_optimizers(self):
        optimizer = get_optimizer_cls(self.config['optimizer']['name'])(get_wd_params(self), **self.config['optimizer']['hparams'])
        if 'scheduler' in self.config['optimizer']:
            if self.config['optimizer']['scheduler']['name'] == 'step':
                scheduler = StepLR(optimizer, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    },
                }
            elif self.config['optimizer']['scheduler']['name'] == 'onecycle':
                scheduler = OneCycleLR(optimizer, max_lr=self.config['optimizer']['hparams']['lr'], total_steps=self.num_steps, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
        return optimizer

    @staticmethod
    def save_attention_hook(module, input, output):
        pass
    
    @property
    def num_steps(self):
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps


