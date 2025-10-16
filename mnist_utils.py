from importlib.resources import files
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator
import random
import re
import numpy as np
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import argparse

from stats import (ood_stats, ambiguity_stats, calib_stats, knn_classification_metrics, calib_stats_values)
from models import (get_models_dict, str_to_class)
from model_setups import (get_heavy_weight_decay_models, get_epochs_models_heavy, 
                          get_aug_models_heavy, get_seq_models, get_aug_epoch_models,
                          get_batch_size_models_heavy)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if ROOT_PATH=="/home/jloch/Desktop/diff/luzern/random_experiments/mnist":
    DATA_PATH = "/home/jloch/Desktop/diff/luzern/values_datasets/mnist"
else:
    DATA_PATH = "/work3/jloch/MNIST/data"

def get_tsne_probs(X, prob_map, x_vec, y_vec):
    """
    Bilinear-interpolate class probabilities from a (H,W,C) prob_map at t-SNE coords X (N,2).
    Returns (N,C).
    """
    X = np.asarray(X, dtype=np.float64)
    interp = RegularGridInterpolator(
        (np.asarray(y_vec, dtype=np.float64), np.asarray(x_vec, dtype=np.float64)),
        np.asarray(prob_map, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    # RegularGridInterpolator expects (y, x) pairs
    pts = np.column_stack([X[:, 1], X[:, 0]])
    return interp(pts)

def class_density_map(X, y, std_mult=0.05, max_sidelength=256, truncate=3, square=True, tqdm_disable=False,
                      std_mult_image=None):
    """
    X: (N,2) t-SNE coordinates
    y: (N,) integer labels in [0..9]
    Returns: prob_map (S,S,10), summed_density (S,S), x_vec (S,), y_vec (S,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    S = int(max_sidelength)
    num_classes = 10

    # --- bandwidth in data units ---
    sigma_world = float(std_mult) * float(X.std() if X.size else 1.0)
    if sigma_world <= 0:
        sigma_world = 1e-8  # degenerate case safety

    # --- bounds with kernel padding (half-size = truncate*sigma) ---
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    padx = truncate * sigma_world
    pady = truncate * sigma_world
    xmin_p, xmax_p = xmin - padx, xmax + padx
    ymin_p, ymax_p = ymin - pady, ymax + pady

    # --- optional square bounds by expanding the smaller span ---
    if square:
        span_x = xmax_p - xmin_p
        span_y = ymax_p - ymin_p
        if span_x == 0: span_x = 1e-8
        if span_y == 0: span_y = 1e-8
        if span_x > span_y:
            extra = (span_x - span_y) * 0.5
            ymin_p -= extra
            ymax_p += extra
        elif span_y > span_x:
            extra = (span_y - span_x) * 0.5
            xmin_p -= extra
            xmax_p += extra

    # --- grid vectors (data-space coordinates per pixel center) ---
    x_vec = np.linspace(xmin_p, xmax_p, S)
    y_vec = np.linspace(ymin_p, ymax_p, S)

    # pixel size in data units
    dx = (xmax_p - xmin_p) / (S - 1) if S > 1 else 1.0
    dy = (ymax_p - ymin_p) / (S - 1) if S > 1 else 1.0
    if dx <= 0: dx = 1e-8
    if dy <= 0: dy = 1e-8

    # Gaussian std in pixel units (isotropic in data → possibly anisotropic in pixels)
    sigma_x_pix = sigma_world / dx
    sigma_y_pix = sigma_world / dy
    sigma_x_pix = max(sigma_x_pix, 1e-8)
    sigma_y_pix = max(sigma_y_pix, 1e-8)

    rx = int(np.ceil(truncate * sigma_x_pix))
    ry = int(np.ceil(truncate * sigma_y_pix))

    density = np.zeros((S, S, num_classes), dtype=np.float64)

    # --- accumulate truncated Gaussian per point (subpixel-accurate center) ---
    # map data coords to fractional pixel coords
    fx = (X[:, 0] - xmin_p) / dx  # 0..S-1 (fractional)
    fy = (X[:, 1] - ymin_p) / dy

    loop_var = range(X.shape[0])
    if not tqdm_disable:
        loop_var = tqdm(loop_var, desc="Building density map", ncols=80, leave=False)

    for i in loop_var:
        c = int(y[i])
        if c < 0 or c >= num_classes:
            continue  # ignore out-of-range labels
        cx = fx[i]
        cy = fy[i]
        # window bounds (inclusive) with clipping
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(S - 1, int(np.ceil(cx + rx)))
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(S - 1, int(np.ceil(cy + ry)))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        # subpixel distances in pixel units
        dxp = (xs - cx) / sigma_x_pix
        dyp = (ys - cy) / sigma_y_pix
        # separable Gaussian
        gx = np.exp(-0.5 * (dxp ** 2))
        gy = np.exp(-0.5 * (dyp ** 2))
        kernel = np.outer(gy, gx)  # (len(ys), len(xs))

        density[y0:y1 + 1, x0:x1 + 1, c] += kernel

    summed_density = density.sum(axis=2)
    eps = 1e-12
    prob_map = density / np.clip(summed_density[..., None], eps, None)

    return prob_map, summed_density, x_vec, y_vec

def get_dataloaders(root=DATA_PATH,
    batch_size=256, num_workers=4, shuffle_train=True, ignore_digits=[],
    augment=False, interpolation_factor=0.0, ambiguous_vae_samples=False
):
    """
    Returns dataloaders for t-SNE 2D coords and MNIST labels as
    well as standard image dataloaders.
    """
    tsne_path = os.path.join(root, "mnist_tsne.pth")
    save_dict = torch.load(tsne_path, weights_only=False)

    tsne_train = np.asarray(save_dict["train_tsne"], float)
    tsne_test = np.asarray(save_dict["test_tsne"], float)
    probs_train = np.asarray(save_dict["probs_train"], float)
    probs_test = np.asarray(save_dict["probs_test"], float)

    
    tr_ds = MNISTWithTSNE(root=root, train=True, tsne_array=tsne_train, prob_array=probs_train, ignore_digits=ignore_digits, augment=augment,
                          interpolation_factor=interpolation_factor, ambiguous_vae_samples=ambiguous_vae_samples)
    te_ds = MNISTWithTSNE(root=root, train=False, tsne_array=tsne_test, prob_array=probs_test, ignore_digits=ignore_digits,
                          interpolation_factor=interpolation_factor, ambiguous_vae_samples=ambiguous_vae_samples)

    train_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, drop_last=False)
    val_dl   = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_dl, val_dl


def increase_contrast_clip(factor=0.1):
    def f(x):
        # x is a tensor in [0,1], shape (C,H,W)
        mean = 0.5
        x = mean + (x - mean) * (1 + factor)  # increase contrast
        return torch.clamp(x, 0.0, 1.0)
    return transforms.Lambda(f)

clip_factor = 0.4
interp = transforms.InterpolationMode.BILINEAR
train_transform = transforms.Compose([
    transforms.RandomAffine(15, shear=(-10,10), scale=(0.9, 1.1), translate=(0.1, 0.1), interpolation=interp),
    transforms.ToTensor(),
    increase_contrast_clip(factor=clip_factor),
    transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean/std
    ]) 
test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

def get_train_transform(augment):
    """ augment: float in [0,1], 0=no aug, 1=full aug"""
    m = float(augment)
    assert 0.0 < m <= 1.0, f"augment must be in (0,1] but got {m}"
    return transforms.Compose([
        transforms.RandomAffine(15*m, shear=(-10*m,10*m), scale=(1-0.1*m, 1+0.1*m), translate=(m*0.1, m*0.1), interpolation=interp),
        transforms.ToTensor(),
        increase_contrast_clip(factor=clip_factor*m),
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean/std
        ]) 

class MNISTWithTSNE(Dataset):
    def __init__(self, root, train, tsne_array, prob_array, download=False, dtype=torch.float32,
                ignore_digits=[], augment=False, normalize_tsne=True, interpolation_factor=0.0,
                ambiguous_vae_samples=False):
        if augment:
            transform = get_train_transform(augment)
        else:
            transform = test_transform
        if ambiguous_vae_samples:
            assert interpolation_factor == 0.0, "interpolation_factor must be 0 if ambiguous_vae_samples is True"
        self.base = datasets.MNIST(root=root, train=train, transform=transform,
                                   target_transform=None, download=download)
        assert len(tsne_array) == len(self.base.data), "t-SNE array length must match MNIST split"

        self.orig_tsne = torch.as_tensor(tsne_array.copy())

        if normalize_tsne:
            mu,std = tsne_array.mean(axis=0), tsne_array.std(axis=0)
            std = np.maximum(std, 1e-8)
            tsne_array = (tsne_array - mu[None,:]) / std[None,:]
        self.tsne = torch.as_tensor(tsne_array, dtype=dtype)
        self.probs = torch.zeros(len(self.base.data), 10, dtype=dtype) 
        self.probs[torch.arange(len(self.base.targets)), self.base.targets] = 1.0
        prob_array = torch.as_tensor(prob_array, dtype=dtype)

        if ambiguous_vae_samples:
            #idx where vae samples start
            vae_dict_path = os.path.join(root, "interp_images.npy")
            vae_dict = np.load(vae_dict_path, allow_pickle=True).item()["train" if train else "test"]
            vae_dict = {k: torch.as_tensor(v) for k, v in vae_dict.items()}
            n_vae = len(vae_dict["indices_A"])
            probs_vae = torch.zeros(n_vae, 10, dtype=dtype)
            #linear mapping that puts 0.25 -> 0.05 and 0.75 -> 0.95
            amb_map = lambda p: (0.9*(p-0.25)/0.5 + 0.05).clamp(0,1)
            probs_vae[torch.arange(n_vae), vae_dict["gt_A"]] = amb_map(vae_dict["ratio_A"])
            probs_vae[torch.arange(n_vae), vae_dict["gt_B"]] = amb_map(vae_dict["ratio_B"])
            self.probs = torch.cat([self.probs, probs_vae], dim=0)
            interp_images = vae_dict["interp_images"][:,0].mul_(255).clamp(0,255).to(torch.uint8)
            self.base.data = torch.cat([self.base.data, interp_images], dim=0)
            top_gt = torch.where(vae_dict["ratio_A"] > vae_dict["ratio_B"], vae_dict["gt_A"], vae_dict["gt_B"])
            self.base.targets = torch.cat([self.base.targets, top_gt], dim=0)

            tsne_A = self.tsne[vae_dict["indices_A"]]
            tsne_B = self.tsne[vae_dict["indices_B"]]
            tsne_interp = tsne_A * vae_dict["ratio_A"][:,None] + tsne_B * vae_dict["ratio_B"][:,None]
            self.tsne = torch.cat([self.tsne, tsne_interp], dim=0)

            orig_tsne_A = self.orig_tsne[vae_dict["indices_A"]]
            orig_tsne_B = self.orig_tsne[vae_dict["indices_B"]]
            orig_tsne_interp = orig_tsne_A * vae_dict["ratio_A"][:,None] + orig_tsne_B * vae_dict["ratio_B"][:,None]
            self.orig_tsne = torch.cat([self.orig_tsne, orig_tsne_interp], dim=0)

        if ignore_digits:
            mask = self.probs[:, ignore_digits].sum(dim=1) < 1e-8
            self.base.data = self.base.data[mask]
            self.base.targets = self.base.targets[mask]
            self.tsne = self.tsne[mask]
            self.probs = self.probs[mask]
            prob_array = prob_array[mask[:len(prob_array)]]
            
        if interpolation_factor > 0.0:
            int_f = interpolation_factor
            n = len(prob_array)
            if ignore_digits:
                #filter out ignored digits and renormalize
                mult_array = torch.as_tensor([[0.0 if i in ignore_digits else 1.0 for i in range(self.probs.shape[1])]], dtype=dtype)
                prob_array *= mult_array
                prob_array /= prob_array.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.probs[:n] = self.probs[:n]*(1-int_f) + int_f*torch.as_tensor(prob_array, dtype=dtype)
    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        tsne = self.tsne[idx]
        probs = self.probs[idx]
        return img, tsne, probs

def dict2str(d):
    return ", ".join([f"{k}: {v}" for k, v in d.items()])

def scaled_lr(lr_ref, bs_ref, bs):
    """
    Scale the learning rate according to batch size using a hybrid sqrt→linear rule.

    Args:
        lr_ref (float): reference learning rate.
        bs_ref (int): reference batch size.
        bs (int or float): new batch size.

    Returns:
        float: scaled learning rate.
    """
    bs = float(bs)
    log_bs = np.log2(bs)
    log_512 = np.log2(512)
    log_8192 = np.log2(8192)

    # Determine alpha (exponent for scaling)
    if bs <= 512:
        alpha = 0.5
    elif bs >= 8192:
        alpha = 1.0
    else:
        # linear interpolation in log2 space
        alpha = 0.5 + 0.5 * (log_bs - log_512) / (log_8192 - log_512)

    # Scaled learning rate
    lr = lr_ref * (bs / bs_ref) ** alpha
    return float(lr)

def train(
    model, train_dl, val_dl, epochs=20, lr=1e-4, weight_decay=0.0,
    device=None, val_every_steps=500, kl_scale=0.0, n_vali_samples=1000, fixed_vali_samples=False,
    tqdm_disable=False, cosine_anneal_epochs=20, soft_labels=False, ckpt_every_epochs=0,
    also_save_ema_ckpts=False, epoch_ema_memory=1, lr_decay=1.0,
    normalize_lr_wrt_batch_size=64,
):
    """
    Basic training loop with BCEWithLogitsLoss + Adam.
    Logs training metrics every step and validation every val_every_steps.
    """
    lr = scaled_lr(lr, normalize_lr_wrt_batch_size, train_dl.batch_size)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if also_save_ema_ckpts:
        ema_state_dict = {}
        for k in model.state_dict().keys():
            ema_state_dict[k] = model.state_dict()[k].detach().cpu().clone()
        memory = len(train_dl)*epoch_ema_memory
        ema_lambda = (memory - 1) / memory
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "steps": 0, "val_every_steps": val_every_steps,
        "ckpt_epochs": {},
        "ema_ckpt_epochs": {},
    }
    if cosine_anneal_epochs and cosine_anneal_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_anneal_epochs * len(train_dl))
    else:
        scheduler = argparse.Namespace(step=lambda: None)  # dummy scheduler
    # Precompute fixed validation indices if requested
    if fixed_vali_samples:
        all_idx = np.arange(len(val_dl.dataset))
        np.random.shuffle(all_idx)
        fixed_indices = all_idx[:n_vali_samples]
    else:
        fixed_indices = None
    if not tqdm_disable:
        total_steps = (len(train_dl) * epochs)
        tqdm_loop = tqdm(total=total_steps, desc="Training")
    train_loss, train_acc = float('nan'), float('nan')
    val_loss, val_acc = float('nan'), float('nan')
    for epoch in range(1, epochs + 1):
        model.train()
        for xb_image, xb_tsne, yb_probs in train_dl:
            if not tqdm_disable:
                
                tqdm_loop.update(1)
                tqdm_loop.set_postfix_str(dict2str({"epoch": epoch, 
                                       "Tr/Va loss": f"{train_loss:.02f}/{val_loss:.02f}",
                                       "Tr/Va acc": f"{train_acc:.03f}/{val_acc:.03f}"}))
                
            history["steps"] += 1
            xb_image, xb_tsne, yb_probs = xb_image.to(device), xb_tsne.to(device), yb_probs.to(device)
            if soft_labels:
                yb = yb_probs
            else:
                #sample from the probs
                yb = torch.multinomial(yb_probs, num_samples=1).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb_image, xb_tsne)
            if isinstance(out, tuple):
                logits, kl = out
            else:
                logits, kl = out, torch.zeros((), device=device)

            loss = criterion(logits, yb) + kl_scale * kl
            loss.backward()
            optimizer.step()
            if epoch > epochs - cosine_anneal_epochs:
                scheduler.step()

            pred = nn.Softmax(dim=1)(logits).argmax(dim=1)
            train_acc = (yb_probs[torch.arange(len(pred)),pred]).mean()

            train_loss = loss.item()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc.item())

            if also_save_ema_ckpts:
                # EMA update with decay 0.99
                for k, v in model.state_dict().items():
                    ema_state_dict[k] = ema_lambda * ema_state_dict[k] + (1-ema_lambda) * v.detach().cpu()
            
            # --- Perform validation every val_every_steps ---
            if val_every_steps is not None and (history["steps"] % val_every_steps == 0):
                # --- Validation ---
                model.eval()
                v_loss_accum, v_correct, v_count = 0.0, 0, 0
                with torch.no_grad():
                    if fixed_vali_samples:
                        subset = torch.utils.data.Subset(val_dl.dataset, fixed_indices)
                        tmp_loader = torch.utils.data.DataLoader(
                            subset, batch_size=val_dl.batch_size, shuffle=False
                        )
                    else:
                        all_idx = np.arange(len(val_dl.dataset))
                        np.random.shuffle(all_idx)
                        idx = all_idx[:n_vali_samples]
                        subset = torch.utils.data.Subset(val_dl.dataset, idx)
                        tmp_loader = torch.utils.data.DataLoader(
                            subset, batch_size=val_dl.batch_size, shuffle=False
                        )

                    for xb, xb_tsne, yb_probs in tmp_loader:
                        xb, xb_tsne, yb_probs = xb.to(device), xb_tsne.to(device), yb_probs.to(device)
                        out = model(xb, xb_tsne)
                        if soft_labels:
                            yb = yb_probs
                        else:
                            yb = torch.multinomial(yb_probs, num_samples=1).squeeze(1)
                        if isinstance(out, tuple):
                            logits, kl = out
                        else:
                            logits, kl = out, torch.zeros((), device=device)
                        v_loss = criterion(logits, yb) + kl_scale * kl
                        v_loss_accum += v_loss.item() * yb.size(0)
                        pred = nn.Softmax(dim=1)(logits).argmax(dim=1)
                        v_correct += (yb_probs[torch.arange(len(pred)),pred]).sum().item()
                        v_count += yb.size(0)

                val_loss = v_loss_accum / max(v_count, 1)
                val_acc  = v_correct / max(v_count, 1)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                model.train()  # back to train mode
        # --- End epoch ---
        # Decay lr per epoch if requested
        if lr_decay != 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_decay
        if ckpt_every_epochs:
            if isinstance(ckpt_every_epochs, int):
                save_ckpt_epoch_flag = ckpt_every_epochs > 0 and (epoch % ckpt_every_epochs == 0)
            elif isinstance(ckpt_every_epochs, (list, tuple)):
                save_ckpt_epoch_flag = epoch in ckpt_every_epochs
            else:
                raise ValueError(f"ckpt_every_epochs must be int or list/tuple of ints, found {type(ckpt_every_epochs)}")
            if save_ckpt_epoch_flag:
                state_dict_detached = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                history["ckpt_epochs"][epoch] = state_dict_detached
                if also_save_ema_ckpts:
                    ema_state_dict_clone = {k: v.clone() for k, v in ema_state_dict.items()}
                    history["ema_ckpt_epochs"][epoch] = ema_state_dict_clone
    if also_save_ema_ckpts and not save_ckpt_epoch_flag:
        history["ema_ckpt_epochs"] = {}
        history["ema_ckpt_epochs"][epoch] = ema_state_dict_clone
    return history

def get_interp_probs(yb,yb_probs,au_factor):
    """Given a an integer vector (bs,) and a probability matrix (bs,C) samples
    probabilities from yb_probs with probability au_factor, otherwise
    just uses the integer label. Independent per batch element.
    """
    if au_factor <= 0.0:
        return yb

    interp_prob_mask = (torch.rand(yb.shape, device=yb.device) < au_factor)
    
    interp_sampled = torch.multinomial(yb_probs, num_samples=1).squeeze(1)
    yb_new = torch.where(interp_prob_mask, interp_sampled, yb)
    return yb_new

def model_probs_on_all_points(model, dataloaders, device=None, logits_instead=False):
    """
    Get model class probabilities on all points in a dataloader.
    Returns (N,C) array of probabilities.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_probs = []
    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]
    for dataloader in dataloaders:
        all_probs.append([])
        with torch.no_grad():
            for xb_image, xb_tsne, yb_probs in dataloader:
                xb_image, xb_tsne, yb_probs = xb_image.to(device), xb_tsne.to(device), yb_probs.to(device)
                out = model(xb_image, xb_tsne)
                if isinstance(out, tuple):
                    logits, kl = out
                else:
                    logits, kl = out, torch.zeros((), device=device)
                if logits_instead:
                    all_probs[-1].append(logits.cpu())
                else:
                    probs = nn.Softmax(dim=1)(logits)
                    all_probs[-1].append(probs.cpu())
        all_probs[-1] = torch.cat(all_probs[-1], dim=0)
    return all_probs


def voronoi_raster(points, values, xlim, ylim, size, supersample=4):
    """
    Render a (smooth) Voronoi field on a fixed grid by supersampling.

    Args
    ----
    points : (N,2) array          site coordinates [[x,y], ...]
    values : (N,) or (N,C) array  value at each site (scalar or vector)
    xlim   : (xmin, xmax)
    ylim   : (ymin, ymax)
    size   : (H, W)               output resolution
    supersample : int             supersampling factor (e.g., 4 or 8)

    Returns
    -------
    img : (H,W) or (H,W,C) array  rasterized field
    x_vec : (W,) array            x coords of pixel centers
    y_vec : (H,) array            y coords of pixel centers (bottom→top)
    """
    assert points.ndim == 2 and points.shape[1] == 2, f"points must be (N,2) but got {points.shape}"
    assert values.ndim in (1, 2) and values.shape[0] == points.shape[0], f"expected values to be (N,) or (N,C) but got {values.shape} for {points.shape[0]} points"
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values)
    H, W   = map(int, size)
    s      = int(supersample)

    # High-res grid (subpixel centers)
    Hh, Wh = H * s, W * s
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)

    dx = (xmax - xmin) / Wh
    dy = (ymax - ymin) / Hh
    xs_hi = xmin + (np.arange(Wh) + 0.5) * dx
    ys_hi = ymin + (np.arange(Hh) + 0.5) * dy

    Xh, Yh = np.meshgrid(xs_hi, ys_hi)          # (Hh,Wh)
    coords = np.column_stack([Xh.ravel(), Yh.ravel()])

    # Nearest site via KD-tree
    tree = cKDTree(points)
    _, nn_idx = tree.query(coords, k=1)         # (Hh*Wh,)

    # Map to values
    if values.ndim == 1:
        field_hi = values[nn_idx].reshape(Hh, Wh)
        # downsample by block average to approximate pixel-area fractions
        img = field_hi.reshape(H, s, W, s).mean(axis=(1, 3))
    else:
        C = values.shape[1]
        field_hi = values[nn_idx].reshape(Hh, Wh, C)
        img = field_hi.reshape(H, s, W, s, C).mean(axis=(1, 3))

    # Pixel-center coordinate vectors for the low-res image
    x_vec = xmin + (np.arange(W) + 0.5) * (xmax - xmin) / W
    y_vec = ymin + (np.arange(H) + 0.5) * (ymax - ymin) / H

    return img, x_vec, y_vec

def loss_plot(history):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train loss')
    if history['val_loss']:
        val_x = np.arange(1,len(history['val_loss'])+1) * history['val_every_steps']
        fmt = ".-" if len(history['val_loss']) < 20 else "-"
        plt.plot(val_x, history['val_loss'], fmt, label='val loss')
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train acc')
    if history['val_acc']:
        val_x = np.arange(1,len(history['val_acc'])+1) * history['val_every_steps']
        fmt = ".-" if len(history['val_acc']) < 20 else "-"
        plt.plot(val_x, history['val_acc'], fmt, label='val acc')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    #if 95% of points >0.8:
    if np.sum(np.array(history['train_acc']) > 0.8) > 0.95 * len(history['train_acc']):
        plt.ylim(0.8,None)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def idx_to_mask(idx, total_length):
    mask = torch.zeros(total_length, dtype=torch.bool)
    if isinstance(idx, slice):
        mask[idx] = True
    elif isinstance(idx, (list, tuple)):
        mask[list(idx)] = True
    else:
        mask[idx] = True
    return mask

def uncertainty_stats_from_ckpts(data, resolution=256, tqdm_disable=False, include_test=True, include_train=True, T=1.0,
                                 add_stats=False, add_knn_sep=True):
    m_cfg = data["model_setup"]
    model_list = [ckpt for ckpt in data["checkpoints"]]
    is_EU=(data["uncertainty_setup"].get("ignore_digits", []) == [2, 3, 5]),
    is_AU2=data["uncertainty_setup"].get("ambiguous_vae_samples", False)

    assert include_train or include_test, "At least one of include_train or include_test must be True"
    tab10_colors = plt.get_cmap('tab10').colors
    save_dict = torch.load(os.path.join(DATA_PATH, "mnist_tsne.pth"), weights_only=False)
    (xvec0, xvec1), (yvec0, yvec1) = save_dict["x_vec"][[0,-1]], save_dict["y_vec"][[0,-1]]
    r = int(resolution)
    if not isinstance(model_list, (list, tuple)):
        model_list = [model_list]

    out_dict = {}

    voronoi_entropies = []
    voronoi_rgbs = []
    voronoi_probs = []
    train_dl, val_dl = get_dataloaders(augment=False, shuffle_train=False, ambiguous_vae_samples=True)

    if include_train and include_test:
        tsne = torch.cat([train_dl.dataset.orig_tsne, val_dl.dataset.orig_tsne], dim=0)
        gts = torch.cat([save_dict["train_labels"], save_dict["test_labels"]], dim=0)
        probs_gts = torch.cat([train_dl.dataset.probs, val_dl.dataset.probs], dim=0)
        vae_idx_train = torch.arange(len(save_dict["train_labels"]), len(train_dl.dataset.probs)).tolist()
        vae_idx_val = torch.arange(len(save_dict["test_labels"]), len(val_dl.dataset.probs)).tolist()
        vae_idx = vae_idx_train + (len(vae_idx_train)+vae_idx_val).tolist()
        vae_mask = idx_to_mask(vae_idx, len(probs_gts))
    elif include_test:
        tsne = val_dl.dataset.orig_tsne
        gts = save_dict["test_labels"]
        probs_gts = val_dl.dataset.probs
        vae_mask = idx_to_mask(slice(len(gts),None), len(val_dl.dataset.probs))
    elif include_train:
        tsne = train_dl.dataset.orig_tsne
        gts = save_dict["train_labels"]
        probs_gts = train_dl.dataset.probs
        vae_mask = idx_to_mask(slice(len(gts),None), len(train_dl.dataset.probs))

    if not tqdm_disable:
        bar = tqdm(model_list, desc="Processing models")
    all_model_probs = []
    all_model_logits = []
    for model_ckpt in model_list:
        model = model_from_cfg(m_cfg)
        model.load_state_dict(model_ckpt)
        model.eval()
        if include_train and include_test:
            model_logits = torch.cat(model_probs_on_all_points(model, [train_dl, val_dl], logits_instead=1), dim=0)
        elif include_test:
            model_logits = model_probs_on_all_points(model, val_dl, logits_instead=1)[0]
        elif include_train:
            model_logits = model_probs_on_all_points(model, train_dl, logits_instead=1)[0]

        model_probs = (model_logits/T).softmax(dim=1)
        
        voronoi,x_vec,y_vec = voronoi_raster(tsne, model_probs.numpy(), (xvec0, xvec1), (yvec0, yvec1), (r,r), supersample=4)

        voronoi_rgb = np.zeros((voronoi.shape[0], voronoi.shape[1], 3), dtype=np.float64)
        for c in range(10):
            voronoi_rgb += voronoi[:, :, c:c+1] * np.array(tab10_colors[c])[None, None, :]

        voronoi_entropy = np.sum(-voronoi * np.log(voronoi + 1e-12), axis=2).reshape(voronoi.shape[0], voronoi.shape[1])

        voronoi_entropies.append(voronoi_entropy)
        voronoi_rgbs.append(voronoi_rgb)
        voronoi_probs.append(voronoi)
        all_model_logits.append(model_logits)
        all_model_probs.append(model_probs)
        if not tqdm_disable:
            bar.update(1)
    if not tqdm_disable:
        bar.close()
    all_model_probs = torch.stack(all_model_probs, axis=0)
    all_model_logits = torch.stack(all_model_logits, axis=0)


    if is_EU:
        #mask = ~torch.isin(out_dict["eval"]["gts"], torch.tensor([2,3,5]))
        # based on probs instead. IID is where GT probs for 2,3,5 sum to near zero
        iid_mask = probs_gts[:,[2,3,5]].sum(1)< 1e-8
    else:
        iid_mask = None

    out_dict["extent"] = (xvec0, xvec1, yvec0, yvec1)
    out_dict["voronoi_entropies"] = torch.as_tensor(np.stack(voronoi_entropies, axis=0))
    out_dict["voronoi_rgbs"] = torch.as_tensor(np.stack(voronoi_rgbs, axis=0))
    out_dict["voronoi_probs"] = torch.as_tensor(np.stack(voronoi_probs, axis=0)).permute(0,3,1,2)  # (N,C,H,W)
    out_dict["mean_rgb"] = out_dict["voronoi_rgbs"].mean(0).clamp(0.0, 1.0)
    out_dict["unc"] = calculate_uncertainty(torch.as_tensor(out_dict["voronoi_probs"], dtype=torch.float32))
    out_dict["eval"] = {"probs": all_model_probs, "tsne": tsne, "preds": all_model_probs.argmax(axis=2), 
                        "gts": gts, "probs_gts": probs_gts, 
                        "unc": calculate_uncertainty(torch.as_tensor(all_model_probs, dtype=torch.float32).permute(0,2,1)),
                        "logits": all_model_logits, "is_AU2": is_AU2, "is_EU": is_EU,
                        "iid_mask": iid_mask, "vae_mask": vae_mask,
                        }
    if add_knn_sep:
        out_dict["knn_sep"] = knn_classification_metrics(out_dict)
        out_dict["knn_sep"]["mask"] = ["No Unc", "EU", "AU", "AU+EU"]
    if add_stats:
        out_dict["ood_stats"] = ood_stats(out_dict, do_plot=False)#, is_ood=mask)
        out_dict["calib_stats"] = calib_stats_values(out_dict, is_AU2=is_AU2)
        out_dict["amb_stats"] = ambiguity_stats(out_dict)#, mask=mask, vae_mask=vae_mask)
    return out_dict

def model_from_cfg(m_cfg, as_func=False):
    model_name = m_cfg["model"]
    model_dict_old = get_models_dict(tsne=m_cfg.get("tsne", False))
    if model_name in model_dict_old:
        model = lambda: model_dict_old[model_name]()
    else:
        assert model_name in str_to_class, f"Unknown model {model_name}, expected one of {list(str_to_class.keys())}"
        model = lambda: str_to_class[model_name](**m_cfg.get("model_args", {}))
    if as_func:
        return model
    else:
        return model()

def train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10, 
                    save_dir=os.path.join(ROOT_PATH, "saves"),
                    skip_existing=True, save_intermediate=False):
    """
    Train ensembles of models across uncertainty and model setups.

    Loop order (outer→inner): uncertainty_setups → model_setups → model repeat index.

    For every (uncertainty_setup, model_setup) pair:
        * Train n_models_per_setup instances (with different random seeds) sequentially.
        * After finishing all n models, save a .pth file in save_dir containing:
            - "checkpoints": list[ dict ] of state_dicts (CPU) length n_models_per_setup
            - "histories": list of training history dicts (as returned by train())
            - "final_acc": list of tuples (train_acc, test_acc) evaluated on FULL sets (no augmentation)
            - metadata: "uncertainty_setup", "model_setup", plus the configuration dicts
    Filename pattern:  f"{uncertainty_key}__{model_key}.pth"

    Notes / Assumptions:
        * Data augmentation only applied to training dataloader if model_setup["data_aug"] is True.
        * Evaluation (train + test accuracy) always uses un-augmented transforms (setting augment=False).
        * If a model_setup provides "tsne": True we pass tsne flag when constructing model.
        * au_factor currently only stored (no AU synthesis logic present in this file); kept for future extension.
        * Seeds: we vary torch.manual_seed, numpy, and random for reproducibility per model.
    """
    print(f"Total setups: {len(model_setups)}")
    os.makedirs(save_dir, exist_ok=True)

    def _progress_filename(base_fname, progress_count):
        if not save_intermediate or progress_count >= n_models_per_setup:
            return base_fname
        base, ext = os.path.splitext(base_fname)
        return f"{base}_({progress_count}of{n_models_per_setup}){ext}"

    def _write_snapshot(base_fname, save_obj, progress_count, previous_path=None):
        target_fname = _progress_filename(base_fname, progress_count)
        target_path = os.path.join(save_dir, target_fname)
        tmp_path = target_path + ".tmp"
        torch.save(save_obj, tmp_path)
        os.replace(tmp_path, target_path)
        if previous_path and previous_path != target_path and os.path.exists(previous_path):
            try:
                os.remove(previous_path)
            except FileNotFoundError:
                pass
        return target_path

    def _match_standard_progress(prefix, fname):
        pattern = rf"{re.escape(prefix)}(?:_\((\d+)(?:of)(\d+)\))?\.pth"
        match = re.fullmatch(pattern, fname)
        if not match:
            return None
        if match.group(1):
            current = int(match.group(1))
            total = int(match.group(2))
            if total != n_models_per_setup:
                return None
            return current
        return n_models_per_setup

    def _match_epoch_progress(prefix, fname):
        pattern = rf"{re.escape(prefix)}_e(\d+)(_ema)?(?:_\((\d+)(?:of)(\d+)\))?\.pth"
        match = re.fullmatch(pattern, fname)
        if not match:
            return None
        epoch = int(match.group(1))
        variant = "ema" if match.group(2) else "normal"
        if match.group(3):
            current = int(match.group(3))
            total = int(match.group(4))
            if total != n_models_per_setup:
                return None
            return epoch, variant, current
        return epoch, variant, n_models_per_setup


    # Outer loop: uncertainty setups
    for u_key, u_cfg in uncertainty_setups.items():
        # Inner loop: model setups
        for m_key, m_cfg in model_setups.items():
            if isinstance(m_cfg.get("epochs", 20),list):
                epoch_list = m_cfg["epochs"]
                m_cfg["epochs"] = max(epoch_list)
                epoch_mode = True
                ckpt_every_epochs = epoch_list
            else:
                epoch_mode = False
                ckpt_every_epochs = 0
            #model_name = m_cfg.get("model")
            if skip_existing and not epoch_mode:
                fname = f"{u_key}__{m_key}.pth"
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath):
                    print(f"Skipping existing file {fpath}")
                    continue
            train_dl_aug, val_dl_aug = get_dataloaders(ignore_digits=u_cfg.get("ignore_digits", []), 
                                                        augment=m_cfg.get("data_aug", False),
                                                        ambiguous_vae_samples=u_cfg.get("ambiguous_vae_samples", 0),
                                                        batch_size=m_cfg.get("batch_size", 256),)
            prefix = f"{u_key}__{m_key}"
            base_fname = f"{prefix}.pth"
            checkpoints = []
            histories = []
            final_acc = []
            prev_progress_path = None
            epoch_prev_paths = {}
            existing_models = 0

            if skip_existing:
                if epoch_mode:
                    required_keys = [("normal", epoch) for epoch in epoch_list]
                    if m_cfg.get("also_save_ema_ckpts", False):
                        required_keys += [("ema", epoch) for epoch in epoch_list]

                    entries = {}
                    for fname in os.listdir(save_dir):
                        parsed = _match_epoch_progress(prefix, fname)
                        if not parsed:
                            continue
                        epoch_val, variant, progress = parsed
                        key = (variant, epoch_val)
                        entries.setdefault(key, {})[progress] = fname

                    required_present = all(key in entries for key in required_keys)
                    final_complete = required_present and all(
                        n_models_per_setup in entries[key] for key in required_keys
                    )
                    if final_complete:
                        print(f"Skipping existing files for {prefix} (all epochs complete)")
                        continue

                    if save_intermediate and required_present:
                        partial_sets = []
                        for key in required_keys:
                            partial = {p for p in entries[key].keys() if 0 < p < n_models_per_setup}
                            if not partial:
                                partial_sets = []
                                break
                            partial_sets.append(partial)
                        common_progress = set.intersection(*partial_sets) if partial_sets else set()
                        best_progress = max(common_progress) if common_progress else 0
                        if best_progress > 0:
                            loaded_epoch_data = {}
                            for key in required_keys:
                                fname = entries[key][best_progress]
                                path = os.path.join(save_dir, fname)
                                data = torch.load(path, weights_only=False)
                                loaded_epoch_data[key] = (path, data)
                                epoch_prev_paths[key] = path
                            first_data = loaded_epoch_data[required_keys[0]][1]
                            histories = list(first_data.get("histories", []))
                            checkpoints = []
                            final_acc = []
                            for idx_existing in range(best_progress):
                                ckpt_entry = {}
                                acc_entry = {}
                                for variant, epoch_val in required_keys:
                                    _, data = loaded_epoch_data[(variant, epoch_val)]
                                    ckpt_entry.setdefault(variant, {})[epoch_val] = data["checkpoints"][idx_existing]
                                    acc_entry.setdefault(variant, {})[epoch_val] = data["final_acc"][idx_existing]
                                checkpoints.append(ckpt_entry)
                                final_acc.append(acc_entry)
                            existing_models = len(checkpoints)
                            print(f"Resuming {prefix} (epoch mode) from {existing_models}/{n_models_per_setup} models")
                else:
                    final_path = os.path.join(save_dir, base_fname)
                    if os.path.isfile(final_path):
                        print(f"Skipping existing file {final_path}")
                        continue
                    if save_intermediate:
                        best_progress = 0
                        best_fname = None
                        for fname in os.listdir(save_dir):
                            progress = _match_standard_progress(prefix, fname)
                            if progress is None or progress >= n_models_per_setup:
                                continue
                            if progress > best_progress:
                                best_progress = progress
                                best_fname = fname
                        if best_progress > 0 and best_fname:
                            resume_path = os.path.join(save_dir, best_fname)
                            resume_data = torch.load(resume_path, weights_only=False)
                            checkpoints = list(resume_data["checkpoints"])
                            histories = list(resume_data["histories"])
                            final_acc = list(resume_data["final_acc"])
                            existing_models = len(checkpoints)
                            prev_progress_path = resume_path
                            print(f"Resuming {prefix} from {existing_models}/{n_models_per_setup} models using {best_fname}")

            for model_idx in range(existing_models, n_models_per_setup):
                seed = 1234 + model_idx  # deterministic but distinct
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                model = model_from_cfg(m_cfg)
                #model = get_models_dict(tsne=m_cfg.get("tsne", False), channel_mult=m_cfg.get("channel_mult", 1))[model_name]()

                history = train(
                    model,
                    train_dl=train_dl_aug,
                    val_dl=val_dl_aug,
                    epochs               = m_cfg.get("epochs", 20),
                    lr                   = m_cfg.get("lr", 1e-4),
                    weight_decay         = m_cfg.get("weight_decay", 0.0),
                    val_every_steps      = m_cfg.get("val_every_steps", 500),
                    kl_scale             = m_cfg.get("kl_scale", 0.0),
                    n_vali_samples       = m_cfg.get("n_vali_samples", 1000),
                    fixed_vali_samples   = m_cfg.get("fixed_vali_samples", False),
                    tqdm_disable         = m_cfg.get("tqdm_disable", False),
                    cosine_anneal_epochs = m_cfg.get("cosine_anneal_epochs", m_cfg.get("epochs", 20)),
                    soft_labels          = u_cfg.get("soft_labels", False),
                    lr_decay    = m_cfg.get("lr_decay", 1.0),
                    ckpt_every_epochs    = ckpt_every_epochs,
                    also_save_ema_ckpts  = m_cfg.get("also_save_ema_ckpts", False)
                )

                # Final evaluation on FULL (un-augmented) train + test
                train_dl_eval, test_dl_eval = get_dataloaders(ignore_digits=u_cfg.get("ignore_digits", []), 
                                                              ambiguous_vae_samples=u_cfg.get("ambiguous_vae_samples", 0),
                                                        augment=False, shuffle_train=False)
                if epoch_mode:
                    # remove checkpoints from histories to keep file size down
                    # and put them in the checkpoints list instead, now a 2d list. 
                    # similarly for final_acc, evaluate over all the checkpoints and store in a 2d final_acc variable instead of 1d
                    if m_cfg.get("also_save_ema_ckpts", False):
                        checkpoints.append({"ema": history.pop("ema_ckpt_epochs"),
                                            "normal": history.pop("ckpt_epochs")})
                    else:
                        checkpoints.append({"normal": history.pop("ckpt_epochs")})
                    # final acc over normal checkpoints
                    final_acc.append({})
                    for ema_or_normal, ckpt_dict in checkpoints[-1].items():
                        final_acc[-1][ema_or_normal] = {}
                        for epoch, state_dict in ckpt_dict.items():
                            model.load_state_dict(state_dict)
                            train_probs, test_probs = model_probs_on_all_points(
                                model, [train_dl_eval, test_dl_eval],
                            )
                            train_acc = (train_dl_eval.dataset.probs[torch.arange(len(train_dl_eval.dataset)),train_probs.argmax(axis=1)]).mean()
                            test_acc = (test_dl_eval.dataset.probs[torch.arange(len(test_dl_eval.dataset)),test_probs.argmax(axis=1)]).mean()
                            final_acc[-1][ema_or_normal][epoch] = {"train_acc": train_acc.item(), "test_acc": test_acc.item()}
                            print(f"Model {model_idx+1}/{n_models_per_setup} ({u_key}, {m_key}, {ema_or_normal}, epoch {epoch}): Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
                    if save_intermediate and checkpoints:
                        progress = len(checkpoints)
                        for ema_or_normal in checkpoints[0].keys():
                            ema_str = "_ema" if ema_or_normal == "ema" else ""
                            for epoch in checkpoints[0][ema_or_normal].keys():
                                save_obj = {
                                    "checkpoints": [ckpt[ema_or_normal][epoch] for ckpt in checkpoints],
                                    "histories": histories,
                                    "final_acc": [fa[ema_or_normal][epoch] for fa in final_acc],
                                    "uncertainty_setup": u_cfg,
                                    "model_setup": {**m_cfg, "epochs": epoch},
                                }
                                base_fname = f"{u_key}__{m_key}_e{epoch}{ema_str}.pth"
                                key = (ema_or_normal, epoch)
                                prev_path = epoch_prev_paths.get(key)
                                new_path = _write_snapshot(base_fname, save_obj, progress, prev_path)
                                epoch_prev_paths[key] = new_path
                                print(f"Saved ensemble results to {new_path}")
                else:
                    train_probs, test_probs = model_probs_on_all_points(model, [train_dl_eval, test_dl_eval])
                    train_acc = (train_dl_eval.dataset.probs[torch.arange(len(train_dl_eval.dataset)),train_probs.argmax(axis=1)]).mean()
                    test_acc = (test_dl_eval.dataset.probs[torch.arange(len(test_dl_eval.dataset)),test_probs.argmax(axis=1)]).mean()
                    print(f"Model {model_idx+1}/{n_models_per_setup} ({u_key}, {m_key}): Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

                    # Store CPU checkpoint
                    checkpoints.append({k: v.cpu() for k, v in model.state_dict().items()})
                    histories.append(history)
                    final_acc.append({"train_acc": train_acc, "test_acc": test_acc})

                    if save_intermediate:
                        save_obj = {
                            "checkpoints": checkpoints,
                            "histories": histories,
                            "final_acc": final_acc,
                            "uncertainty_setup": u_cfg,
                            "model_setup": m_cfg,
                        }
                        progress = len(checkpoints)
                        prev_progress_path = _write_snapshot(f"{u_key}__{m_key}.pth", save_obj, progress, prev_progress_path)
                        print(f"Saved ensemble results to {prev_progress_path}")
            if epoch_mode:
                # save a checkpoint by appending e.g. _e10_ema or _e10 to the filename
                # keep histories dictionaries identical for each save
                # make sure to replace the epoch key in model setup in addition to a "ema" boolean key
                for ema_or_normal in checkpoints[0].keys():
                    for epoch in checkpoints[0][ema_or_normal].keys():
                        save_obj = {
                            "checkpoints": [ckpt[ema_or_normal][epoch] for ckpt in checkpoints],
                            "histories": histories,
                            "final_acc": [fa[ema_or_normal][epoch] for fa in final_acc],  # list[(train_acc, test_acc)]
                            "uncertainty_setup": u_cfg,
                            "model_setup": {**m_cfg, "epochs": epoch},
                        }
                        ema_str = "_ema" if ema_or_normal == "ema" else ""
                        fname = f"{u_key}__{m_key}_e{epoch}{ema_str}.pth"
                        fpath = os.path.join(save_dir, fname)
                        torch.save(save_obj, fpath)
                        print(f"Saved ensemble results to {fpath}")
            else:
                save_obj = {
                    "checkpoints": checkpoints,
                    "histories": histories,
                    "final_acc": final_acc,  # list[(train_acc, test_acc)]
                    "uncertainty_setup": u_cfg,
                    "model_setup": m_cfg,
                }
                fname = f"{u_key}__{m_key}.pth"
                fpath = os.path.join(save_dir, fname)
                torch.save(save_obj, fpath)
                print(f"Saved ensemble results to {fpath}")
    return None

def calculate_uncertainty(softmax_preds: torch.Tensor):
    """ Assumes shape structure:
     (N, C, H, W) where N is number of stochastic forward passes,
     C is number of classes, H is height, and W is width.
    """
    mean_softmax = torch.mean(softmax_preds, dim=0)
    pred_entropy = torch.zeros(*softmax_preds.shape[2:], device=mean_softmax.device)
    for y in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[y] * torch.log(mean_softmax[y])
        nan_pos = torch.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1
    expected_entropy = torch.zeros(
        softmax_preds.shape[0], *softmax_preds.shape[2:], device=softmax_preds.device
    )
    for pred in range(softmax_preds.shape[0]):
        entropy = torch.zeros(*softmax_preds.shape[2:], device=softmax_preds.device)
        for y in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, y] * torch.log(softmax_preds[pred, y])
            nan_pos = torch.isnan(entropy_class)
            entropy[~nan_pos] += entropy_class[~nan_pos]
        entropy *= -1
        expected_entropy[pred] = entropy
    expected_entropy = torch.mean(expected_entropy, dim=0)
    mutual_information = pred_entropy - expected_entropy
    negative_mask = mutual_information < 0
    mutual_information[negative_mask] = mutual_information[~negative_mask].min()
    return {"TU": pred_entropy,
            "AU": expected_entropy,
            "EU": mutual_information}

def plot_voronoi(out_dict, colorbar=False, title="", cmap="viridis", vmax=1.0, plot_digits=True):
    titles = [title, "TU", "AU", "EU"]
    images = [out_dict["mean_rgb"].cpu().numpy(),
              out_dict["unc"]["TU"].cpu().numpy(),
              out_dict["unc"]["AU"].cpu().numpy(),
              out_dict["unc"]["EU"].cpu().numpy()]
    extent = out_dict["extent"]
    points = out_dict["eval"]["tsne"].numpy()
    if plot_digits:
        centers = []
        for i in range(10):
            #weight = out_dict["eval"]["probs_gts"][:,i].numpy()
            weight = (out_dict["eval"]["probs_gts"][:,i].numpy()==1).astype(float)
            print(weight.sum())
            pts = (points*weight[:,None])/weight.sum()
            center = pts.sum(axis=0)
            centers.append((i, center))

    plt.figure(figsize=(16,5))
    for i in range(4):
        plt.subplot(1,4,i+1)
        if i == 0:
            plt.imshow(images[i], origin='lower', extent=extent, vmin=0, vmax=1)
        else:
            im = plt.imshow(images[i], origin='lower', extent=extent, cmap=cmap, vmax=vmax, vmin=0)
            if colorbar:
                plt.colorbar(im, fraction=0.046, pad=0.04)
        if plot_digits:
            for digit, center in centers:
                if center is not None:
                    plt.text(center[0], center[1], str(digit), fontsize=25, fontweight='light',
                        color=f'C{digit}', ha='center', va='center',
                        path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])
        plt.title(titles[i], fontsize=16, fontweight='bold' if i>0 else 'normal')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


seq_type_to_varying_param = {
    "aug_H": "data_aug",
    "aug_H_EU": "data_aug",
    "batch_size_H": "batch_size",
    "batch_size_H_aug": "batch_size",
    "channels": "channels",
    "channels_aug": "channels",
    "depth": "depth",
    "depth_aug": "depth",
    "epochs_H": "epochs",
    "epochs_H_aug": "epochs",
    "epochs_lrd_H": "epochs",
    "epochs_lrd_H_ema": "epochs",
    "weight_decay_H": "weight_decay",
    "weight_decay_H_aug": "weight_decay"}


"""seq_types = {"channels_aug": "[MODEL]_c[CHANNELS]_aug1",
             "channels": "[MODEL]_c[CHANNELS]_aug0",
             "depth_aug": "[MODEL]_d[DEPTH]_aug1",
             "depth": "[MODEL]_d[DEPTH]_aug0",
             "aug": "[MODEL]_aug[AUG]",
             "epochs": "[MODEL]_e[EPOCHS]",
             "aug_H": "[MODEL]_aug[AUG]_H",
             "wd_H": "[MODEL]_wd[WEIGHT_DECAY]_H_aug0",
             "wd_H_aug": "[MODEL]_wd[WEIGHT_DECAY]_H_aug1",
             "epochs_H": "[MODEL]_e[EPOCHS]_H_aug0",
             "epochs_H_aug": "[MODEL]_e[EPOCHS]_H_aug1",}

seq_types_match = {"channels_aug": lambda name: name.split("_")[1].startswith("c") and name.endswith("_aug1"),
                    "channels": lambda name: name.split("_")[1].startswith("c") and name.endswith("_aug0"),
                    "depth_aug": lambda name: name.split("_")[1].startswith("d") and name.endswith("_aug1"),
                    "depth": lambda name: name.split("_")[1].startswith("d") and name.endswith("_aug0"),
                    "aug": lambda name: name.split("_")[1].startswith("aug") and not name.endswith("_H"),
                    "epochs": lambda name: name.split("_")[1].startswith("e") and "_H" not in name,
                    "aug_H": lambda name: name.split("_")[1].startswith("aug") and name.endswith("_H"),
                    "wd_H": lambda name: name.split("_")[1].startswith("wd") and "_H" in name and name.endswith("_aug0"),
                    "wd_H_aug": lambda name: name.split("_")[1].startswith("wd") and "_H" in name and name.endswith("_aug1"),
                    "epochs_H": lambda name: name.split("_")[1].startswith("e") and name.endswith("_H_aug0"),
                    "epochs_H_aug": lambda name: name.split("_")[1].startswith("e") and name.endswith("_H_aug1"),
                    }

model_types = ["CNN", "MLP"]
identifiers = {"d": ("depth",int), "c": ("channels",int), "aug": ("aug",float), "e": ("epochs",int), "wd": ("weight_decay",float)}
identifiers_inv = {v[0]: (k,v[1]) for k,v in identifiers.items()}
flags = {"H": "heavy"}
flags_inv = {"heavy": "H"}

def seq_type_from_name(name):
    #Returns the sequence type from the model name,
    #in addition to all the key-value pairs found in the name.
    #
    split_name = name.split("_")
    params = {}
    params["model_type"] = split_name[0]
    assert params["model_type"] in model_types, f"Unknown model type {params['model_type']} in {name}"
    for i,value in enumerate(split_name[1:]):

        #find identifier in value
        match = 0
        for idf, (idf_full, idf_type) in identifiers.items():
            if value.startswith(idf):
                params[idf_full] = idf_type(value[len(idf):])
                match = 1
                if i==0:
                    params["varying_param"] = idf_full
                break
        if match:
            continue
        #find flag in value
        for flag, flag_str in flags.items():
            if value == flag:
                params[flag_str] = True
                match = 1
                break
        if match:
            continue
        raise ValueError(f"Unknown identifier in {value} of {name}. Should start with one of {list(identifiers.keys())} or be one of {list(flags.keys())}")
    #match to a seq_type. 
    for seq_type, match_fn in seq_types_match.items():
        if match_fn(name):
            params["seq_type"] = seq_type
            break
    assert "seq_type" in params, f"Could not match sequence type for {name}"
    return params"""

AU2_EU_setup = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}

if __name__=="__main__":

    """
    # models 
    # 1. overfitted CNN (big, [CNN1])
    # 2. well-fitted CNN (big, [CNN1], data aug)
    # 3. underfitted CNN (small, [cnn1])
    # 4. overfitted MLP (big, [MLP1])
    # 5. well-fitted MLP (big, [MLP1], data aug)
    # 6. underfitted MLP (small [mlp1])
    # 7. well-fitted tsne MLP ([MLP1])

    # data setups
    # 1. Normal mnist
    # 2. no 2,3,5 (EU)
    # 3. no 2,3,5 (EU), augmented with AU (interpolation factor 1.0)
    # 4. augmented with AU (interpolation factor 1.0)
    # 5. no 2,3,5 (EU), augmented with partial AU (interpolation factor 0.3)
    # 6. augmented with partial AU (interpolation factor 0.3)

    # 10 models per setup
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--setup", type=int, default=0)
    argparser.add_argument("--augs", type=str, default="")
    argparser.add_argument("--epochs", type=str, default="")
    argparser.add_argument("--lr_decay", type=str, default="")
    args = argparser.parse_args()
    
    if args.setup==0:
        model_setups = {
            "overfitted_CNN": {"model": "CNN1"},
            "well_fitted_CNN": {"model": "CNN1", "data_aug": True},
            "underfitted_CNN": {"model": "cnn1"},
            "overfitted_MLP": {"model": "MLP1"},
            "well_fitted_MLP": {"model": "MLP1", "data_aug": True},
            "underfitted_MLP": {"model": "mlp1"},
            "well_fitted_tsne_MLP": {"model": "MLP1", "tsne": True},
        }

        uncertainty_setups = {
            #"NoUnc": {"ignore_digits": [], "au_factor": 0.0},
            #"AU": {"ignore_digits": [], "au_factor": 1.0},
            "EU": {"ignore_digits": [2, 3, 5], "au_factor": 0.0},
            "AU_EU": {"ignore_digits": [2, 3, 5], "au_factor": 1.0},
            "AU.3_EU": {"ignore_digits": [2, 3, 5], "au_factor": 0.3},
            #"AU.3": {"ignore_digits": [], "au_factor": 0.3},
        }
        
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==1:
        model_setups = {
            "uberfitted_CNN": {"model": "CNN3", "epochs": 50},
            "uberfitted_MLP": {"model": "MLP3", "epochs": 50},
        }

        uncertainty_setups = {
            "NoUnc": {"ignore_digits": [], "au_factor": 0.0},
            "AU": {"ignore_digits": [], "au_factor": 1.0},
            "EU": {"ignore_digits": [2, 3, 5], "au_factor": 0.0},
            "AU_EU": {"ignore_digits": [2, 3, 5], "au_factor": 1.0},
            "AU.3_EU": {"ignore_digits": [2, 3, 5], "au_factor": 0.3},
            "AU.3": {"ignore_digits": [], "au_factor": 0.3},
        }

        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==2:
        from pathlib import Path
        p = os.path.join(ROOT_PATH, "saves")
        print("Calculating uncertainty stats for ensembles in:")
        print(p)
        print("and adding to saved .pth files.")
        pathlist = list(Path(p).rglob("*.pth"))
        print(f"Found {len(pathlist)} files.")
        filelist = []
        print("Filtering for .pth files without unc_stats...")
        for f in pathlist:
            try:
                data = torch.load(str(f), weights_only=False)
                data["unc_stats"]#["knn_sep"]
            except:
                filelist.append(str(f))
        print(f"{len(filelist)} files need processing.")
        bar = tqdm(filelist, desc="Processing models")
        for f in filelist:
            bar.set_description(f"Processing {f}")
            data = torch.load(f, weights_only=False)
            unc_dict = uncertainty_stats_from_ckpts(data,
                                                    include_train=False, 
                                                    add_stats=True, 
                                                    tqdm_disable=True
                                                    )
            data["unc_stats"] = unc_dict
            torch.save(data, f)
            bar.update(1)
        bar.close()
    elif args.setup==3:
        print("Training ensembles for soft label setups and ambiguous VAE samples.")
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
            "AU2": {"ignore_digits": [], "ambiguous_vae_samples": True},
            "AU2_EU_soft": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True, "soft_labels": True},
            "AU2_soft": {"ignore_digits": [], "ambiguous_vae_samples": True, "soft_labels": True},
            "AU_EU_soft": {"ignore_digits": [2, 3, 5], "au_factor": 1.0, "soft_labels": True},
        }
        model_setups = {
            "overfitted_CNN": {"model": "CNN1"},
            "well_fitted_CNN": {"model": "CNN1", "data_aug": True},
            "underfitted_CNN": {"model": "cnn1"},
            "overfitted_MLP": {"model": "MLP1"},
            "well_fitted_MLP": {"model": "MLP1", "data_aug": True},
            "underfitted_MLP": {"model": "mlp1"},
            "well_fitted_tsne_MLP": {"model": "MLP1", "tsne": True},
        }
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==4:
        print("Training ensembles for soft label setups and ambiguous VAE samples.")
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True}, # < --- DONE, can be commented out
            "AU2": {"ignore_digits": [], "ambiguous_vae_samples": True}, #not done yet
            "AU2_EU_soft": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True, "soft_labels": True},
            "AU2_soft": {"ignore_digits": [], "ambiguous_vae_samples": True, "soft_labels": True},
            "AU_EU_soft": {"ignore_digits": [2, 3, 5], "au_factor": 1.0, "soft_labels": True},
        }
        model_setups = {
            "uberfitted_CNN": {"model": "CNN3", "epochs": 50},
            "uberfitted_MLP": {"model": "MLP3", "epochs": 50},
        }
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==5:
        print("Training a sequence of models where only magnitude of augmentations and network width is varied. AU2_EU is the unc setup")

        model_setups = get_seq_models()
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==6:
        print("Training a sequence of HEAVY models where only magnitude of augmentations is varied. AU2_EU is the unc setup")

        model_setups = get_aug_models_heavy()
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==7:
        print("Training a sequence of HEAVY models where only magnitude of augmentations is varied. AU2_EU is the unc setup")
        assert len(args.augs)>0, "Please provide --augs argument as a comma-separated list of floats, e.g. --augs 0,0.1,0.2"
        augs = [float(a) for a in args.augs.split(",")]
        model_setups = get_aug_models_heavy(augs=augs)
        uncertainty_setups = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==8:
        print("Training a sequence of HEAVY models where only number of epochs and network width is varied. AU2_EU is the unc setup")
        assert len(args.epochs)>0, "Please provide --epochs argument as a comma-separated list of integers, e.g. --epochs 10,20,30"
        epochs = [int(e) for e in args.epochs.split(",")]
        model_setups = get_epochs_models_heavy(epochs=epochs)
        uncertainty_setups = AU2_EU_setup
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==9:
        print("Training a sequence of HEAVY models where only magnitude of augmentations is varied. EU is the unc setup")
        if len(args.augs)>0:
            augs = [float(a) for a in args.augs.split(",")]
            model_setups = get_aug_models_heavy(augs=augs)
        else:
            model_setups = get_aug_models_heavy()
        uncertainty_setups = {"EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": False},}
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==10:
        print("Training a sequence of HEAVY models where only weight decay and network width is varied. AU2_EU is the unc setup")
        model_setups = get_heavy_weight_decay_models()
        uncertainty_setups = AU2_EU_setup
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==11:
        print("Small test to see if everything works with epoch sweeps and epoch_mode")
        epoch_list = [1,2,3]
        model_setups = {
            "CNN_test_aug0":   {"model": "CNN1", "epochs": epoch_list, "data_aug": 0, "also_save_ema_ckpts": True},
            "CNN_test_aug0.5": {"model": "CNN1", "epochs": epoch_list, "data_aug": 0.5, "also_save_ema_ckpts": True},
            "CNN_test_aug1":   {"model": "CNN1", "epochs": epoch_list, "data_aug": 1.0, "also_save_ema_ckpts": True},
        }
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==12:
        print("Batch size")
        model_setups = get_batch_size_models_heavy()
        uncertainty_setups = AU2_EU_setup
        train_ensembles(model_setups, uncertainty_setups)
    elif args.setup==13:
        print("1000 epochs")
        assert len(args.augs)>0, "Please provide --augs argument as a comma-separated list of floats, e.g. --augs 0,0.1,0.2"
        augs = [float(a) for a in args.augs.split(",")]
        model_setups = get_aug_epoch_models(augs=augs)
        uncertainty_setups = AU2_EU_setup
        train_ensembles(model_setups, uncertainty_setups, skip_existing=True, save_intermediate=True)
    elif args.setup==14:
        print("Small test to see if everything works with epoch sweeps and epoch_mode")
        epoch_list = [1,2,4]
        model_setups = {
            "CNN_test_aug0":   {"model": "CNN1", "epochs": epoch_list, "data_aug": float(args.augs), "also_save_ema_ckpts": True},
        }
        uncertainty_setups = AU2_EU_setup
        train_ensembles(model_setups, uncertainty_setups, skip_existing=True, save_intermediate=True)
    else:
        print("Unknown setup:", args.setup)