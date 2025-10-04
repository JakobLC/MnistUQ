from importlib.resources import files
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator
import random
import numpy as np
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from models import (get_models_dict,
                    CNN, MLP, CNNDeluxe, MLPDeluxe, str_to_class)
import argparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score


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
    batch_size=64, num_workers=0, shuffle_train=True, ignore_digits=[],
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

def train(
    model, train_dl, val_dl, epochs=20, lr=1e-4, weight_decay=0.0,
    device=None, val_every_steps=500, kl_scale=0.0, n_vali_samples=1000, fixed_vali_samples=False,
    tqdm_disable=False, cosine_anneal_epochs=20, au_factor=0.0, soft_labels=False
):
    """
    Basic training loop with BCEWithLogitsLoss + Adam.
    Logs training metrics every step and validation every val_every_steps.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "steps": 0, "val_every_steps": val_every_steps
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
    assert points.ndim == 2 and points.shape[1] == 2
    assert values.ndim in (1, 2) and values.shape[0] == points.shape[0]
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
                    skip_existing=True):
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

    # Outer loop: uncertainty setups
    for u_key, u_cfg in uncertainty_setups.items():
        # Inner loop: model setups
        for m_key, m_cfg in model_setups.items():
            #model_name = m_cfg.get("model")
            if skip_existing:
                fname = f"{u_key}__{m_key}.pth"
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath):
                    print(f"Skipping existing file {fpath}")
                    continue
            train_dl_aug, val_dl_aug = get_dataloaders(ignore_digits=u_cfg.get("ignore_digits", []), 
                                                        augment=m_cfg.get("data_aug", False),
                                                        ambiguous_vae_samples=u_cfg.get("ambiguous_vae_samples", 0))

            checkpoints = []
            histories = []
            final_acc = []

            for model_idx in range(n_models_per_setup):
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
                    au_factor = u_cfg.get("au_factor", 0.0),
                    soft_labels = u_cfg.get("soft_labels", False)
                )

                # Final evaluation on FULL (un-augmented) train + test
                train_dl_eval, test_dl_eval = get_dataloaders(ignore_digits=u_cfg.get("ignore_digits", []), 
                                                              ambiguous_vae_samples=u_cfg.get("ambiguous_vae_samples", 0),
                                                        augment=False, shuffle_train=False)
                train_probs, test_probs = model_probs_on_all_points(
                    model, [train_dl_eval, test_dl_eval],
                )
                #(train_probs.argmax(axis=1) == train_dl.dataset.base.targets).float().mean()
                #train_acc = (train_probs.argmax(dim=1) == torch.as_tensor(train_dl_eval.dataset.base.targets)).float().mean().item()
                #test_acc  = (test_probs.argmax(dim=1)  == torch.as_tensor(test_dl_eval.dataset.base.targets)).float().mean().item()
                train_acc = (train_dl_eval.dataset.probs[torch.arange(len(train_dl_eval.dataset)),train_probs.argmax(axis=1)]).mean()
                test_acc = (test_dl_eval.dataset.probs[torch.arange(len(test_dl_eval.dataset)),test_probs.argmax(axis=1)]).mean()
                print(f"Model {model_idx+1}/{n_models_per_setup} ({u_key}, {m_key}): Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

                # Store CPU checkpoint
                checkpoints.append({k: v.cpu() for k, v in model.state_dict().items()})
                histories.append(history)
                final_acc.append({"train_acc": train_acc, "test_acc": test_acc})

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
    return {"pred_entropy": pred_entropy,
            "aleatoric_uncertainty": expected_entropy,
            "epistemic_uncertainty": mutual_information}

def model_func_from_ensemble_dict(ensemble_dict):
    raise ValueError("Deprecated, use model_from_cfg instead.")
    model_setup = ensemble_dict["model_setup"]
    model_name = model_setup.get("model")
    tsne_flag = model_setup.get("tsne", False)
    def model_func():
        return get_models_dict(tsne=tsne_flag, channel_mult=model_setup.get("channel_mult", 1))[model_name]()
    return model_func

def idx_to_mask(idx, total_length):
    mask = torch.zeros(total_length, dtype=torch.bool)
    if isinstance(idx, slice):
        mask[idx] = True
    elif isinstance(idx, (list, tuple)):
        mask[list(idx)] = True
    else:
        mask[idx] = True
    return mask

def uncertainty_stats_from_ckpts(model_list, m_cfg, resolution=256, tqdm_disable=False, include_test=True, include_train=True, T=1.0,
                                 add_stats=False, is_EU=False, is_AU2=False):
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
        tsne = torch.cat([train_dl.dataset.tsne, val_dl.dataset.tsne], dim=0)
        gts = torch.cat([save_dict["train_labels"], save_dict["test_labels"]], dim=0)
        probs_gts = torch.cat([train_dl.dataset.probs, val_dl.dataset.probs], dim=0)
        vae_idx_train = torch.arange(len(save_dict["train_labels"]), len(train_dl.dataset.probs)).tolist()
        vae_idx_val = torch.arange(len(save_dict["test_labels"]), len(val_dl.dataset.probs)).tolist()
        vae_idx = vae_idx_train + (len(vae_idx_train)+vae_idx_val).tolist()
        vae_mask = idx_to_mask(vae_idx, len(probs_gts))
    elif include_test:
        tsne = val_dl.dataset.tsne
        gts = save_dict["test_labels"]
        probs_gts = val_dl.dataset.probs
        vae_mask = idx_to_mask(slice(len(gts),None), len(val_dl.dataset.probs))
    elif include_train:
        tsne = train_dl.dataset.tsne
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
    out_dict["extent"] = (xvec0, xvec1, yvec0, yvec1)
    out_dict["voronoi_entropies"] = torch.as_tensor(np.stack(voronoi_entropies, axis=0))
    out_dict["voronoi_rgbs"] = torch.as_tensor(np.stack(voronoi_rgbs, axis=0))
    out_dict["voronoi_probs"] = torch.as_tensor(np.stack(voronoi_probs, axis=0)).permute(0,3,1,2)  # (N,C,H,W)
    out_dict["mean_rgb"] = out_dict["voronoi_rgbs"].mean(0).clamp(0.0, 1.0)
    out_dict["unc"] = calculate_uncertainty(torch.as_tensor(out_dict["voronoi_probs"], dtype=torch.float32))
    out_dict["eval"] = {"probs": all_model_probs, "tsne": tsne, "preds": all_model_probs.argmax(axis=2), 
                        "gts": gts, "probs_gts": probs_gts, "vae_mask": vae_mask,
                        "unc": calculate_uncertainty(torch.as_tensor(all_model_probs, dtype=torch.float32).permute(0,2,1)),
                        "logits": all_model_logits
                        }
    if add_stats:
        if is_EU:
            #mask = ~torch.isin(out_dict["eval"]["gts"], torch.tensor([2,3,5]))
            # based on probs instead
            mask = out_dict["eval"]["probs_gts"][:,[2,3,5]].sum(1)< 1e-8
        else:
            mask = None
        out_dict["ood_stats"] = ood_stats(out_dict, do_plot=False, is_ood=mask)
        out_dict["calib_stats"] = calib_stats(out_dict, mask=mask, vae_mask=vae_mask, is_AU2=is_AU2)
        out_dict["amb_stats"] = ambiguity_stats(out_dict, mask=mask, vae_mask=vae_mask)
    return out_dict


def auroc(unc_score, is_ood, do_plot=False):
    """Computes AUROC for a set of 0/1 labels (is_ood) given uncertainty scores (higher = more uncertain).
    """
    fpr, tpr, thresholds = roc_curve(is_ood, unc_score)
    auc = roc_auc_score(is_ood, unc_score)
    if do_plot:
        plt.plot(fpr, tpr, linewidth=0.7)
        plt.plot([0,1],[0,1],"k--")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
    return auc, fpr, tpr, thresholds
    

def plot_voronoi(out_dict, colorbar=False, title="", cmap="viridis", vmax=1.0, plot_digits=True):
    titles = [title, "TU", "AU", "EU"]
    images = [out_dict["mean_rgb"].cpu().numpy(),
              out_dict["unc"]["pred_entropy"].cpu().numpy(),
              out_dict["unc"]["aleatoric_uncertainty"].cpu().numpy(),
              out_dict["unc"]["epistemic_uncertainty"].cpu().numpy()]
    extent = out_dict["extent"]
    if plot_digits:
        centers = []
        for i in range(10):
            mask = (out_dict["eval"]["gts"] == i)
            if torch.sum(mask) > 0:
                pts = out_dict["eval"]["points"][mask]
                center = pts.mean(axis=0)
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

def ece(probs, labels, n_bins=20, equal_bin_weight=False, return_bins=False):
    """Computes Expected Calibration Error (ECE) for a set of predicted probabilities and true labels.
    probs: (N,C) array of predicted class probabilities
    labels: (N,) array of true class labels (integers in [0,C-1])
    n_bins: number of bins to use
    equal_bin_weight: if True, each bin contributes equally to the final ECE; otherwise weighted by bin size.
    """
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = []
    bin_sizes = []
    for b in range(n_bins):
        bin_mask = (bin_indices == b)
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_confidence = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(accuracies[bin_mask])
            ece.append(np.abs(bin_accuracy - bin_confidence))
            bin_sizes.append(bin_size)
        else:
            ece.append(0.0)
            bin_sizes.append(0)
    if return_bins:
        return ece, bin_sizes
    else:
        if equal_bin_weight:
            return sum(ece)/len(ece)
        else:
            return sum([e * s for e, s in zip(ece, bin_sizes)]) / (sum(bin_sizes)+1e-12)

def calib_stats(out_dict,mask=None,vae_mask=None, do_plot=False, is_AU2=False):
    """Computes ECE with no temperature scaling, ECE with optimal temperature scaling, 
    for both NLL and ECE optimal temperatures.
    """
    if (mask is None) and (vae_mask is None):
        joint_mask = slice(None)
    elif (mask is None):
        joint_mask = torch.logical_not(vae_mask)
    elif (vae_mask is None):
        joint_mask = mask
    else:
        joint_mask = torch.logical_and(mask, torch.logical_not(vae_mask))
    gts = out_dict["eval"]["probs_gts"].numpy()[joint_mask].argmax(1)
    Ts = []
    probs_optece = 0
    probs_optnll = 0
    probs = 0
    for i in range(len(out_dict["eval"]["logits"])):
        logits = out_dict["eval"]["logits"][i].numpy()
        T_nll,T_ece = optimize_temp_scaling(logits[joint_mask], gts, do_plot=do_plot)
        Ts.append((T_nll, T_ece))
        probs += out_dict["eval"]["probs"][i].numpy()
        probs_optnll += nn.Softmax(dim=1)(torch.as_tensor(logits / T_nll, dtype=torch.float32)).numpy()
        probs_optece += nn.Softmax(dim=1)(torch.as_tensor(logits / T_ece, dtype=torch.float32)).numpy()
    
    if do_plot:
        #remove legend
        plt.subplot(1,2,1)
        plt.gca().get_legend().remove()
        plt.tight_layout()
        plt.show()
    probs /= len(out_dict["eval"]["logits"])
    probs_optnll /= len(Ts)
    probs_optece /= len(Ts)
    Ts = np.array(Ts)
    T_optnll = float(np.median(Ts[:,0]))
    T_optece = float(np.median(Ts[:,1]))

    if is_AU2:
        #use vae samples
        if mask is None:
            eval_mask = slice(None)
        else:
            eval_mask = mask
    else:
        eval_mask = joint_mask

    gts2 = out_dict["eval"]["probs_gts"].numpy()[eval_mask].argmax(1)
    probs_gts2 = out_dict["eval"]["probs_gts"].numpy()[eval_mask]
    probs_optnll2 = probs_optnll[eval_mask]
    probs_optece2 = probs_optece[eval_mask]
    probs2 = probs[eval_mask]
    n2 = len(gts2)

    ece_ = ece(probs2, gts2, n_bins=20)
    ece_optnll = ece(probs_optnll2, gts2, n_bins=20)
    ece_optece = ece(probs_optece2, gts2, n_bins=20)

    nll = -np.mean(np.log(probs2[np.arange(n2), gts2] + 1e-12))
    nll_optnll = -np.mean(np.log(probs_optnll2[np.arange(n2), gts2] + 1e-12))
    nll_optece = -np.mean(np.log(probs_optece2[np.arange(n2), gts2] + 1e-12))

    ece_optece_eq = ece(probs_optece2, gts2, n_bins=20, equal_bin_weight=True)
    ece_eq = ece(probs2, gts2, n_bins=20, equal_bin_weight=True)
    ece_optnll_eq = ece(probs_optnll2, gts2, n_bins=20, equal_bin_weight=True)

    ens_test_acc = probs_gts2[torch.arange(n2),probs2.argmax(1)].mean()
    ens_test_acc_optece = probs_gts2[torch.arange(n2),probs_optece2.argmax(1)].mean()
    ens_test_acc_optnll = probs_gts2[torch.arange(n2),probs_optnll2.argmax(1)].mean()

    stats = {
        "ece": ece_,
        "ece_optnll": ece_optnll,
        "ece_optece": ece_optece,
        "nll": nll,
        "nll_optnll": nll_optnll,
        "nll_optece": nll_optece,
        "T_opt_nll": T_optnll,
        "T_opt_ece": T_optece,
        "ece_optece_eq": ece_optece_eq,
        "ece_eq": ece_eq,
        "ece_optnll_eq": ece_optnll_eq,
        "ens_test_acc": ens_test_acc,
        "ens_test_acc_optece": ens_test_acc_optece,
        "ens_test_acc_optnll": ens_test_acc_optnll,
    }
    stats = {k: float(v) for k, v in stats.items()}
    return stats

def optimize_temp_scaling(logits, labels, bounds=(0.1, 10.0), n_eval=100, do_plot=False):
    """Optimize temperature scaling parameter on a validation set.
    logits: (N,C) array of model logits (pre-softmax)
    labels: (N,) array of true class labels (integers in [0,C-1])
    Returns optimal temperature (float).
    """
    
    def nll_loss(temp):
        scaled_logits = logits / temp
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        nll = -np.mean(np.log(scaled_probs[np.arange(len(labels)), labels] + 1e-12))
        return nll

    temps = np.linspace(bounds[0], bounds[1], n_eval)
    losses = [nll_loss(t) for t in temps]
    eces = [ece(torch.as_tensor(logits / t, dtype=torch.float32).softmax(dim=1).numpy(), labels) for t in temps]
    T_opt_nll = temps[np.argmin(losses)]
    T_opt_ece = temps[np.argmin(eces)]
    if do_plot:
        #plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(temps, losses, "-")
        plt.xlabel("Temperature")
        plt.ylabel("NLL Loss")
        plt.grid(True)
        ymin, ymax = plt.ylim()
        plt.vlines(T_opt_nll, ymin, ymax, colors="blue", linestyles="--", label=f"T_opt_nll: {T_opt_nll:.3f}")
        plt.vlines(T_opt_ece, ymin, ymax, colors="orange", linestyles="--", label=f"T_opt_ece: {T_opt_ece:.3f}")
        plt.ylim(ymin, ymax)
        plt.xlim(bounds[0], bounds[1])
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(temps, eces, "-", color="orange")
        plt.xlabel("Temperature")
        plt.ylabel("ECE")
        plt.grid(True)
        ymin, ymax = plt.ylim()
        plt.vlines(T_opt_nll, ymin, ymax, colors="blue", linestyles="--", label=f"T_opt_nll: {T_opt_nll:.3f}")
        plt.vlines(T_opt_ece, ymin, ymax, colors="orange", linestyles="--", label=f"T_opt_ece: {T_opt_ece:.3f}")
        plt.ylim(ymin, ymax)
        plt.xlim(bounds[0], bounds[1])
        #plt.tight_layout()
    return T_opt_nll, T_opt_ece

def ambiguity_stats(out_dict, do_plot=False, mask=None, vae_mask=None):
    """Computes spearmanr for a range of uncertainty measures given probabilistic ground truths.
    """
    if mask is None:
        mask = slice(None)
    if vae_mask is None:
        raise ValueError("vae_mask must be provided to ambiguity_stats")
    probs_gts = out_dict["eval"]["probs_gts"].numpy()[mask]
    entropy_gt = -np.sum(probs_gts * np.log(probs_gts + 1e-12), axis=1)
    is_high_entropy = np.quantile(entropy_gt, 0.9) < entropy_gt
    amb_stats = {}
    for k, unc in out_dict["eval"]["unc"].items():
        unc2 = unc.numpy()[mask]
        spearman_all = spearmanr(unc2, entropy_gt).correlation
        spearman_high = spearmanr(unc2[is_high_entropy], entropy_gt[is_high_entropy]).correlation
        spearman_low = spearmanr(unc2[~is_high_entropy], entropy_gt[~is_high_entropy]).correlation
        amb_stats[k] = {"spearman_all": spearman_all,
                       "spearman_high": spearman_high,
                       "spearman_low": spearman_low,
                       "mean_entropies": {"gt": entropy_gt.mean(),
                                         "all": unc2.mean(),
                                         "high": unc2[is_high_entropy].mean(),
                                         "low": unc2[~is_high_entropy].mean()},
                                        }
        
    # compute that mean abs probability difference between gts and pred
    # best pred is mean of all models
    probs_pred = out_dict["eval"]["probs"].numpy()[:,mask].mean(axis=0)
    vae_mask_mask = vae_mask[mask]
    amb_stats["mean_prob_diff"] = {"all": np.abs(probs_pred - probs_gts).mean(axis=1).mean(axis=0),
                                   "vae": np.abs(probs_pred - probs_gts).mean(axis=1)[vae_mask_mask].mean(),
                                   "non_vae": np.abs(probs_pred - probs_gts).mean(axis=1)[~vae_mask_mask].mean(),}

    if do_plot:
        #make a bar plot of spearman_all for each uncertainty measure
        plt.figure(figsize=(8,4))
        keys = [k for k in amb_stats.keys() if k not in ["mean_entropies", "mean_prob_diff"]]
        group_keys = ["all", "high", "low"]
        x = np.arange(3)
        width = 1/(len(keys)+1)
        for i, k in enumerate(keys):
            vals = [amb_stats[k][f"spearman_{group}"] for group in group_keys]
            plt.bar(x + i*width, vals, width=width, label=k)
            #mean_entropies written with text on top of bar
            for j, v in enumerate(vals):
                value = amb_stats[k]['mean_entropies'][group_keys[j]]
                plt.text(x[j] + i*width, v + 0.02, f"{short_fmt(value)}", ha='center', va='bottom', fontsize=8)
        plt.xticks(x + width*(len(keys)-1)/2, group_keys)
        plt.ylim(-1,1)
        plt.ylabel("Spearman Correlation")
        plt.title("Spearman Correlation of Uncertainty vs. GT Entropy")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
    return amb_stats

def ood_stats(out_dict, do_plot=True, is_ood=None):
    """Computes AUROC for detecting OOD samples (2,3,5) given uncertainty measures.
    """
    if is_ood is None:
        is_ood = out_dict["eval"]["probs_gts"][:,[2,3,5]].sum(1) >= 1e-8
    stats_dict = {}
    n = len(out_dict["eval"]["unc"])

    for i,(k,v) in enumerate(out_dict["eval"]["unc"].items()):
        if do_plot: plt.subplot(1,n,i+1)
        auc, fpr, tpr, thresholds = auroc(v, is_ood, do_plot=do_plot)
        stats_dict[k] = {"auc": auc, "ood_entropy": v[is_ood].mean(), "iid_entropy": v[~is_ood].mean()}
        if do_plot: 
            plt.title(f"{k}, AUROC: {auc:.4f}")
    return stats_dict

def short_fmt(x):
    """Format a float in a short way. 3 significant digits,
    but if <0.01 or >999 use scientific notation. At most 3 decimal places after the comma as well.
    """
    if x < 0.01 or x > 999:
        return f"{x:.1e}"
    else:
        if x>1:
            return f"{x:.3g}"
        else:
            return f"{x:.3f}"

def get_seq_models(channels_cnn = [4,8,16,32,64], 
                   channels_mlp = [32,64,128,256,512],
                   augs=[0,0.25,0.5,0.75,1],
                   depths_cnn = [1,2,4,8,16],
                   depths_mlp = [2,4,8,16,32],
                   epochs=[10,20,30,40,50]):

    """SETUPS:
        - Sweeping channel_mult, aug=0
        - Sweeping channel_mult, aug=1
        - Sweeping depth, cm=1, augs=0
        - Sweeping depth, cm=1, augs=1
        - Sweeeping augs, cm=1, center depth
    """
    
    #"CNN1": lambda: CNNDeluxe(base_channels=cm*16,num_blocks=2,num_downsamples=3)
    #"MLP1": lambda: MLPDeluxe(tsne=tsne, width=cm*128, num_layers=4)

    base_cnn = {"model": "models.CNNDeluxe", "data_aug": 0, "epochs": 20, "model_args": {"base_channels": 16, "num_blocks": 2, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "data_aug": 0, "epochs": 20, "model_args": {"width": 128, "num_layers": 4}}

    model_cfgs = {}
    #model_cfgs["CNN_base"] = {}
    #model_cfgs["MLP_base"] = {}
    for c in channels_cnn:
        model_cfgs[f"CNN_c{c}_aug0"] = {"model_args": {"base_channels": c}}
        model_cfgs[f"CNN_c{c}_aug1"] = {"model_args": {"base_channels": c}, "data_aug": 1}
    for c in channels_mlp:
        model_cfgs[f"MLP_c{c}_aug0"] = {"model_args": {"width": c}}
        model_cfgs[f"MLP_c{c}_aug1"] = {"model_args": {"width": c}, "data_aug": 1}
    for d in depths_cnn:
        model_cfgs[f"CNN_d{d}_aug0"] = {"model_args": {"num_blocks": d}}
        model_cfgs[f"CNN_d{d}_aug1"] = {"model_args": {"num_blocks": d}, "data_aug": 1}
    for d in depths_mlp:
        model_cfgs[f"MLP_d{d}_aug0"] = {"model_args": {"num_layers": d}}
        model_cfgs[f"MLP_d{d}_aug1"] = {"model_args": {"num_layers": d}, "data_aug": 1}
    for a in augs:
        model_cfgs[f"CNN_aug{a}"] = {"data_aug": a}
        model_cfgs[f"MLP_aug{a}"] = {"data_aug": a}
    for e in epochs:
        model_cfgs[f"CNN_e{e}"] = {"epochs": e, "model_args": {"base_channels": 32, "num_blocks": 4, "num_downsamples": 4}}
        model_cfgs[f"MLP_e{e}"] = {"epochs": e, "model_args": {"width": 512, "num_layers": 8}}

    # merge with base configs
    for k in model_cfgs:
        if "CNN" in k:
            model_cfgs[k]["model_args"] = {**base_cnn["model_args"], **model_cfgs[k].get("model_args", {})}
            model_cfgs[k] = {**base_cnn, **model_cfgs[k]}
        elif "MLP" in k:
            model_cfgs[k]["model_args"] = {**base_mlp["model_args"], **model_cfgs[k].get("model_args", {})}
            model_cfgs[k] = {**base_mlp, **model_cfgs[k]}
        else:
            raise ValueError(f"Unknown model type in key {k}")
    return model_cfgs

def get_seq_models_heavy(augs=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
    """SETUPS:
        - Sweeping augs for heavy models
    """
    base_cnn = {"model": "models.CNNDeluxe", "epochs": 50, "model_args": {"base_channels": 32, "num_blocks": 3, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "epochs": 50, "model_args": {"width": 256, "num_layers": 8}}
    model_cfgs = {}
    for a in augs:
        model_cfgs[f"CNN_aug{a}_H"] = {"data_aug": a}
        model_cfgs[f"MLP_aug{a}_H"] = {"data_aug": a}
    # merge with base configs
    for k in model_cfgs:
        if "CNN" in k:
            model_cfgs[k]["model_args"] = {**base_cnn["model_args"], **model_cfgs[k].get("model_args", {})}
            model_cfgs[k] = {**base_cnn, **model_cfgs[k]}
        elif "MLP" in k:
            model_cfgs[k]["model_args"] = {**base_mlp["model_args"], **model_cfgs[k].get("model_args", {})}
            model_cfgs[k] = {**base_mlp, **model_cfgs[k]}
        else:
            raise ValueError(f"Unknown model type in key {k}")
    return model_cfgs

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
        p = os.path.join(ROOT_PATH, "saves")
        print("Calculating uncertainty stats for ensembles in:")
        print(p)
        print("and adding to saved .pth files.")
        filelist = os.listdir(p)
        bar = tqdm(filelist, desc="Processing models")
        for filename in filelist:
            bar.set_description(f"Processing {filename}")
            data = torch.load(os.path.join(p, filename), weights_only=False)
            if "unc_stats" in data:
                bar.update(1)
                continue
            unc_dict = uncertainty_stats_from_ckpts(data["checkpoints"], 
                                                    m_cfg=data["model_setup"],
                                                    include_train=False, 
                                                    add_stats=True, 
                                                    tqdm_disable=True,
                                                    is_EU = (data["uncertainty_setup"].get("ignore_digits", []) == [2,3,5]),
                                                    is_AU2 = "AU2" in filename
                                                    )
            data["unc_stats"] = unc_dict
            torch.save(data, os.path.join(p, filename))
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
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
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
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==5:
        print("Training a sequence of models where only magnitude of augmentations and network width is varied. AU2_EU is the unc setup")

        model_setups = get_seq_models()
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==6:
        print("Training a sequence of HEAVY models where only magnitude of augmentations is varied. AU2_EU is the unc setup")

        model_setups = get_seq_models_heavy()
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==7:
        model_setups = get_seq_models_heavy(augs=[0.1,0.2,0.3])
        uncertainty_setups = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==8:
        model_setups = get_seq_models_heavy(augs=[0.4,0.5,0.6])
        uncertainty_setups = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==9:
        model_setups = get_seq_models_heavy(augs=[0.7,0.8,0.9])
        uncertainty_setups = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    elif args.setup==10:
        model_setups = get_seq_models_heavy(augs=[1])
        uncertainty_setups = {"AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},}
        train_ensembles(model_setups, uncertainty_setups, n_models_per_setup=10)
    else:
        print("Unknown setup:", args.setup)