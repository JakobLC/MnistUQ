import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from einops import rearrange
import os
import numpy as np
from mnist_utils import class_density_map
from tqdm.auto import tqdm

# SOURCE: https://github.com/MahanVeisi8/Latent-Diffusion-MNIST-DDPM-using-Autoencoder

class Config:
    DATASET_NAME = "MNIST"
    IMAGE_SIZE = 28
    LATENT_DIM = 64
    SAVE_DIR = "/home/jloch/Desktop/diff/luzern/random_experiments/mnist/diff_ae"
    MODEL_NAME = "Autoencoder_with_CAB.pth"
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encoder/Decoder Configurations
    ENCODER_CHANNELS = [1, 32, 64]         # Input -> Intermediate -> Latent
    DECODER_CHANNELS = [64, 32, 1]         # Latent -> Intermediate -> Output
    NUM_CABS_ENCODER = [1, 0]              # Number of CABs in each encoder layer
    NUM_CABS_DECODER = [1, 0]              # Number of CABs in each decoder layer
    REDUCTION_FACTOR = 8                  # Channel attention reduction factor
    DEBUG = False                          # Enable debugging prints

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_mnist_dataloader(train=True):
    dataset = datasets.MNIST(
        root="/home/jloch/Desktop/diff/luzern/values_datasets/mnist", train=train, download=False, transform=get_transforms()
    )
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=train)
    return dataloader

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, reduction=16, bias=False):
        super(CAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias),
        )
        self.ca = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return res + x  # Residual Connection

class Encoder(nn.Module):
    def __init__(self, latent_dim, channels, num_cabs_per_layer):
        super().__init__()
        layers = []
        for in_ch, out_ch, num_cabs in zip(channels[:-1], channels[1:], num_cabs_per_layer):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.extend([CAB(out_ch, reduction=Config.REDUCTION_FACTOR) for _ in range(num_cabs)])  # Add CABs
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 7 * 7, latent_dim),
        )

    def forward(self, x):
        if Config.DEBUG:
            print(f"Encoder Input Shape: {x.shape}")
        x = self.conv_layers(x)
        if Config.DEBUG:
            print(f"Encoder After Conv Layers: {x.shape}")
        x = self.flatten(x)
        if Config.DEBUG:
            print(f"Encoder Output (Flattened): {x.shape}")
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, channels, num_cabs_per_layer):
        super().__init__()
        self.channels = channels  # Save channels as a class attribute

        # Define the linear layer for latent-to-feature map transformation
        self.linear = nn.Linear(latent_dim, channels[0] * 7 * 7)


        layers = []
        for in_ch, out_ch, num_cabs in zip(channels[:-1], channels[1:], num_cabs_per_layer):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))  # Deconvolution layer
            layers.extend([CAB(out_ch, reduction=Config.REDUCTION_FACTOR) for _ in range(num_cabs)])  # Add CABs
            if out_ch != 1:  # Activation: ReLU for intermediate layers, Sigmoid for the final layer
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, z):
        if Config.DEBUG:
            print(f"Decoder Input Shape: {z.shape}")

        # Latent to feature map
        x = self.linear(z)
        if Config.DEBUG:
            print(f"After Linear Layer: {x.shape} (Expected: [batch_size, {self.channels[0] * 7 * 7}])")

        x = rearrange(x, "b (c h w) -> b c h w", c=self.channels[0], h=7, w=7)
        if Config.DEBUG:
            print(f"After Reshape: {x.shape} (Expected: [batch_size, {self.channels[0]}, 7, 7])")

        # Pass through the deconvolution layers
        x = self.deconv_layers(x)
        if Config.DEBUG:
            print(f"Decoder Output Shape: {x.shape} (Expected: [batch_size, 1, 28, 28])")

        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim, Config.ENCODER_CHANNELS, Config.NUM_CABS_ENCODER)
        self.decoder = Decoder(latent_dim, Config.DECODER_CHANNELS, Config.NUM_CABS_DECODER)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if Config.DEBUG:
            print(f"Autoencoder Input Shape: {x.shape}")
        z = self.encoder(x)
        if Config.DEBUG:
            print(f"Latent Representation Shape: {z.shape}")
        out = self.decoder(z)
        if Config.DEBUG:
            print(f"Autoencoder Output Shape: {out.shape}")
        return out

def encode_all_mnist(save_filename="mnist_latent_vae.pth", 
                     save_location="/home/jloch/Desktop/diff/luzern/values_datasets/mnist",
                     batch_size=64,
                     tqdm_disable=False):
    """Goes through all 70k MNIST images and encodes them to latent space."""
    X_images = get_MNIST_images()
    autoencoder = Autoencoder(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    load_path = os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
    autoencoder.load_state_dict(torch.load(load_path, map_location=Config.DEVICE, weights_only=True))
    autoencoder.eval()

    dataloader = DataLoader(X_images, batch_size=batch_size, shuffle=False)
    all_latents = []
    with torch.no_grad():
        for x_batch in tqdm(dataloader, desc="Encoding MNIST", disable=tqdm_disable):
            x_batch = x_batch.to(Config.DEVICE)[:,None]
            z_batch = autoencoder.encode(x_batch)
            all_latents.append(z_batch.cpu())
    all_latents = torch.cat(all_latents, dim=0)
    save_path = os.path.join(save_location, save_filename)
    torch.save(all_latents, save_path)
    return all_latents.numpy()

def latent_interpolation(xA, xB, steps=11, geodesic=True):
    autoencoder = Autoencoder(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    load_path = os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
    autoencoder.load_state_dict(torch.load(load_path, map_location=Config.DEVICE, weights_only=True))
    autoencoder.eval()

    with torch.no_grad():
        zA = autoencoder.encode(xA.to(Config.DEVICE))
        zB = autoencoder.encode(xB.to(Config.DEVICE))
        norm_A = torch.norm(zA, p=2)
        norm_B = torch.norm(zB, p=2)
        interpolated_frames = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * zA + alpha * zB
            if geodesic:
                z_interp = (z_interp / torch.norm(z_interp, p=2)) * ((1 - alpha) * norm_A + alpha * norm_B)
            x_interp = autoencoder.decode(z_interp).cpu().numpy()
            interpolated_frames.append(x_interp[0, 0])  # Assuming batch size of 1

    return interpolated_frames

def batched_interpolation(xA_batch, xB_batch, ratio_A=0.5, geodesic=True):
    autoencoder = Autoencoder(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    load_path = os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
    autoencoder.load_state_dict(torch.load(load_path, map_location=Config.DEVICE, weights_only=True))
    autoencoder.eval()

    with torch.no_grad():
        zA = autoencoder.encode(xA_batch.to(Config.DEVICE))
        zB = autoencoder.encode(xB_batch.to(Config.DEVICE))
        ratio_A = ratio_A.to(Config.DEVICE).reshape(-1,1)
        norm_A = torch.norm(zA, p=2, dim=1, keepdim=True)
        norm_B = torch.norm(zB, p=2, dim=1, keepdim=True)
        z_interp = ratio_A * zA + (1 - ratio_A) * zB
        if geodesic:
            z_interp = (z_interp / torch.norm(z_interp, p=2, dim=1, keepdim=True)) * (ratio_A * norm_A + (1 - ratio_A) * norm_B)
        x_interp = autoencoder.decode(z_interp).cpu().numpy()
    return x_interp  # Shape: [batch_size, 1, 28, 28]

def sample_digit_based(digits, gts, replace=True):
    digit_sample, num = np.unique(digits.flatten(), return_counts=True)
    out_idx = -np.ones(digits.shape)
    for digit,num_sample in zip(digit_sample, num):
        mask = (gts == digit)
        idx = np.random.choice(np.where(mask)[0], size=num_sample, replace=replace)
        out_idx[digits == digit] = idx
    assert (out_idx != -1).all(), "Some digits were not found in gts!"
    return out_idx
neighbourhood = {(0,3): 0.6,
                 (0,5): 1,
                 (0,6): 0.6,
                 (1,2): 1,
                 (1,7): 1,
                 (1,8): 0.3,
                 (2,3): 1,
                 (2,7): 0.3,
                 (2,8): 1,
                 (3,5): 0.6,
                 (3,8): 1,
                 (4,7): 0.6,
                 (4,9): 1,
                 (5,6): 1,
                 (5,8): 1,
                 (5,9): 1,
                 (6,9): 0.6,
                 (7,8): 1,
                 (7,9): 1,
                 (8,9): 1}

def get_MNIST_images():
    # ----- config -----
    batch_size = 1024
    save_path = "/home/jloch/Desktop/diff/luzern/values_datasets/mnist"
    # ----- load MNIST (no saving needed beyond the cache in save_path) -----
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=save_path, train=True, download=False, transform=transform)
    test_ds  = datasets.MNIST(root=save_path, train=False, download=False, transform=transform)

    # concatenate train and test sets
    full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    full_ds.targets = np.concatenate([train_ds.targets.numpy(), test_ds.targets.numpy()])

    loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----- stack images and labels -----
    Xs = []
    for x, _ in loader:
        Xs.append(x[:,0].numpy())
    X_images = np.concatenate(Xs, axis=0)
    return X_images


def get_masked_entropy_map(masked=True):
    save_path = "/home/jloch/Desktop/diff/luzern/values_datasets/mnist"
    save_dict = torch.load(os.path.join(save_path, "mnist_tsne.pth"),weights_only=False)
    prob_map, summed_density, x_vec, y_vec = class_density_map(
        X= torch.cat([save_dict["train_tsne"], save_dict["test_tsne"]], dim=0).numpy(), 
        y=torch.cat([save_dict["train_labels"], save_dict["test_labels"]], dim=0).numpy(),
        std_mult=0.15, max_sidelength=256, truncate=3, square=True, tqdm_disable=True)
    
    entropy_map = np.sum(-prob_map * np.log(prob_map + 1e-12), axis=2)
    if masked:
        white_mask = 1/(1+np.exp(38*summed_density[...,None]/summed_density[...,None].max()-4))[:,:,0]
        entropy_map = (1-white_mask)*entropy_map
    return entropy_map, x_vec, y_vec


def sample_from_entropy_map(
    entropy_map, tsne, x_vec, y_vec, gts,
    num_samples=10, gaussian_std=0.1, truncate=3,
    attempt_multiplier=5, tries_per_center=1, rng=None,
    tqdm_disable=False, sample_points_instead=False
):
    """
    Sample index pairs of MNIST digits near high-entropy t-SNE locations.
    Each attempt picks ONE center from the entropy map and tries (by default once)
    to draw two nearby indices with DIFFERENT labels using a Gaussian weighting.
    If it fails, it simply moves on to a NEW center. No fallbacks/widening.

    Args:
        entropy_map: (H, W) nonnegative map; used as categorical sampling probs over centers.
        tsne: (N, 2) t-SNE coordinates per sample.
        x_vec: (W,) x-coordinates for entropy_map columns.
        y_vec: (H,) y-coordinates for entropy_map rows.
        gts: (N,) integer labels in [0..9].
        num_samples: number of pairs to return (rows in output).
        gaussian_std: std of isotropic Gaussian weighting in t-SNE space.
        truncate: only consider points within radius = truncate * gaussian_std.
        attempt_multiplier: total attempts = attempt_multiplier * num_samples.
        tries_per_center: how many tries per chosen center (default 1).
        sample_points_insteads: if True, ignores entropy map and samples the
            gaussian location as a random point in t-SNE space.
        rng: np.random.Generator or None (uses np.random.default_rng()).

    Returns:
        np.ndarray of shape (num_samples, 2) with indices. If not enough pairs are found
        within the allowed attempts, remaining rows are filled with -1.
    """
    rng = rng or np.random.default_rng()

    H, W = entropy_map.shape
    N = tsne.shape[0]
    gaussian_std *= tsne.std(axis=0).mean()  # scale to t-SNE spread
    R = float(truncate) * float(gaussian_std)
    R2 = R * R
    sigma2 = float(gaussian_std) ** 2

    # 1) Build probs over centers
    probs = entropy_map.astype(float).ravel()
    probs = np.clip(probs, 0, None)
    total = probs.sum()    
    if total <= 0 or not np.isfinite(total):
        probs = np.ones_like(probs) / probs.size
    else:
        probs /= total
    

    # Precompute a fast view of t-SNE coords
    X = tsne[:, 0]
    Y = tsne[:, 1]

    out = np.full((num_samples, 2), -1, dtype=int)
    found = 0
    max_attempts = int(attempt_multiplier * num_samples)

    if not tqdm_disable:
        tqdm_loop = tqdm(total=num_samples, desc="Sampling Indices")

    for _ in range(max_attempts):
        if found >= num_samples:
            break

        # Pick ONE center location from the entropy map
        if sample_points_instead:
            ri = rng.integers(0, N)
            cxv, cyv = float(X[ri]), float(Y[ri])
        else:
            flat_idx = rng.choice(probs.size, p=probs)
            cy, cx = np.unravel_index(flat_idx, (H, W))
            cxv, cyv = float(x_vec[cx]), float(y_vec[cy])

        # 2) Compute candidates within true circular truncate (no widening)
        dx = X - cxv
        dy = Y - cyv
        d2 = dx * dx + dy * dy
        cand_mask = d2 <= R2
        if not np.any(cand_mask):
            continue  # move to a NEW center

        cand_idx = np.flatnonzero(cand_mask)
        if cand_idx.size < 2:
            continue  # need at least two candidates

        w = np.exp(-0.5 * (d2[cand_idx] / sigma2)).astype(float)

        if not np.isfinite(w).any() or w.sum() <= 0:
            continue  # degenerate weights -> new center

        w = w / w.sum()

        for _try in range(int(tries_per_center)):
            i_local = rng.choice(cand_idx.size, p=w)
            j_local = rng.choice(cand_idx.size, p=w)
            i = cand_idx[i_local]
            j = cand_idx[j_local]
            if i != j and gts[i] != gts[j]:
                out[found] = (i, j)
                found += 1
                if not tqdm_disable:
                    tqdm_loop.update(1)
                break  # stop tries for this center; move to next attempt

    return out

def sample_from_vae_latents(dataset, gts,
    num_samples=10, num_neighbours=100,
    distribution="uniform", attempt_multiplier=10,
    tqdm_disable=False, assert_complete=True
    ):
    """
    Calculates distances in VAE latent space. Samples a random point
    and samples a random neighbour from the k nearest neighbours. if
    the labels differ, the sample is kept otherwise rejected.
    """
    latents_path = "/home/jloch/Desktop/diff/luzern/values_datasets/mnist/mnist_latent_vae.pth"
    latents = torch.load(latents_path,weights_only=False).numpy()
    if dataset=="train":
        latents = latents[:60000]
        rp = np.random.permutation(60000)
        groups = [rp[i*10000:(i+1)*10000] for i in range(6)]
    elif dataset=="test":
        latents = latents[60000:]
    else:
        raise ValueError("dataset must be 'train' or 'test'")
    assert len(latents)==len(gts), "Length of latents and gts must match!"
    N = len(gts)
    out = np.full((num_samples, 2), -1, dtype=int)
    found = 0
    max_attempts = int(attempt_multiplier * num_samples)
    if not tqdm_disable:
        tqdm_loop = tqdm(total=num_samples, desc="Sampling Indices from VAE Latents")
    for _ in range(max_attempts):
        if found >= num_samples:
            break
        ri = np.random.randint(0, N)
        if dataset=="test":
            subset = np.arange(10000)
        elif dataset=="train":
            #10k random subset
            group_idx = np.random.randint(len(groups))
            subset = groups[group_idx]
        dists = np.linalg.norm(latents[subset] - latents[ri:ri+1], axis=1)
        idx_sort = np.argsort(dists)
        idx_sort = subset[idx_sort]  # map back to full indices
        neighbours = idx_sort[1:num_neighbours+1]  # exclude self
        if distribution=="uniform":
            chosen = np.random.choice(neighbours, size=1, replace=False)[0]
        elif distribution=="distance":
            weights = 1/(dists[neighbours]+1e-12)
            weights /= weights.sum()
            chosen = np.random.choice(neighbours, size=1, replace=False, p=weights)[0]
        if gts[ri] != gts[chosen]:
            out[found] = (ri, chosen)
            found += 1
            if not tqdm_disable:
                tqdm_loop.update(1)
    if assert_complete:
        assert (out != -1).all(), "Not enough pairs found!"
    return out

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latent Space Interpolation with Autoencoder")
    parser.add_argument("--num_images_train", type=int, default=30000, help="Number of images to process")
    parser.add_argument("--num_images_test", type=int, default=5000, help="Number of test images to process")
    parser.add_argument("--method", type=str, choices=["random", "random_neighbours", "smart","vae_neighbours"], default="vae_neighbours", help="Indice selection method")
    parser.add_argument("--disable_rescale", action="store_true", help="Disable rescaling of interpolation images")
    parser.add_argument("--smart_std", type=float, default=0.15, help="Std for smart sampling method")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for interpolation")
    parser.add_argument("--save_filename", type=str, default="interp_images.npy", help="Filename for saving interpolated images")
    parser.add_argument("--save_location", type=str, default="/home/jloch/Desktop/diff/luzern/values_datasets/mnist", help="Location to save the interpolated images")
    parser.add_argument("--tqdm_disable", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--ratio_mode", type=str, choices=["uniform", "center","pyramid"], default="pyramid", help="Mode for choosing ratio_A in interpolation")
    parser.add_argument("--sample_points_instead", action="store_true", help="Sample points randomly in t-SNE space instead of using entropy map")
    args = parser.parse_args()
    save_path = "/home/jloch/Desktop/diff/luzern/values_datasets/mnist"
    save_dict = torch.load(os.path.join(save_path, "mnist_tsne.pth"),weights_only=False)

    dict_for_save = {}


    for dataset in ["train", "test"]:
        if dataset == "train":
            gts = save_dict["train_labels"].numpy()
            num_samples = args.num_images_train
            tsne = save_dict["train_tsne"].numpy()         
            X_images = get_MNIST_images()[:60000]
        else:
            gts = save_dict["test_labels"].numpy()
            num_samples = args.num_images_test
            tsne = save_dict["test_tsne"].numpy()
            X_images = get_MNIST_images()[60000:]
        N = len(gts)
        print(f"Using method: {args.method} with {num_samples} image pairs")
        if args.method=="random_neighbours":
            pairs = list(neighbourhood.keys())
            weights = np.array([neighbourhood[pair] for pair in pairs])
            weights = weights / weights.sum()
            chosen_pairs = np.random.choice(len(pairs), size=num_samples, p=weights, replace=True)
            chosen_pairs = np.array([list(pairs[i]) for i in chosen_pairs])
            interp_idx = sample_digit_based(chosen_pairs, gts)
        elif args.method=="random":
            interp_idx = np.random.choice(N, size=(num_samples,2), replace=True)
        elif args.method=="smart":
            entropy_map, x_vec, y_vec = get_masked_entropy_map(masked=True)
            
            interp_idx = sample_from_entropy_map(
                entropy_map, tsne, x_vec, y_vec, gts,
                num_samples=num_samples, gaussian_std=args.smart_std, truncate=3,
                attempt_multiplier=5, tries_per_center=1, rng=None, tqdm_disable=args.tqdm_disable,
                sample_points_instead=args.sample_points_instead
            )
        elif args.method=="vae_neighbours":
            interp_idx = sample_from_vae_latents(
                dataset, gts,
                num_samples=num_samples, num_neighbours=100,
                distribution="uniform", attempt_multiplier=10,
                tqdm_disable=args.tqdm_disable, assert_complete=True
            )
        if args.disable_rescale:
            f = lambda vals: vals
        else:
            f = lambda vals: np.clip((vals-0.3)*1.2+0.3,0,1)


        processed = 0
        num_batches = int(np.ceil(num_samples / args.batch_size))
        interp_images = []
        all_ratio_A = []
        for _ in tqdm(range(num_batches), desc="Interpolating Batches", disable=args.tqdm_disable):
            batch_idx = interp_idx[processed:processed+args.batch_size]
 
            xA_batch = torch.as_tensor(X_images[batch_idx[:,0].astype(int)])[:,None]
            xB_batch = torch.as_tensor(X_images[batch_idx[:,1].astype(int)])[:,None]
            processed += len(batch_idx)
            if args.ratio_mode=="uniform":
                ratio_A = 0.25 + 0.5 * torch.rand(len(batch_idx))
            elif args.ratio_mode=="center":
                ratio_A = 0.5*torch.ones(len(batch_idx))
            elif args.ratio_mode=="pyramid":
                ratio_A = 0.25 + 0.25 * torch.rand(len(batch_idx),2).sum(axis=1)
            all_ratio_A.append(ratio_A)
            interp_images.append(batched_interpolation(xA_batch,xB_batch, ratio_A=ratio_A))
        gt_A = gts[interp_idx[:,0]]
        gt_B = gts[interp_idx[:,1]]
        all_ratio_A = torch.cat(all_ratio_A, axis=0).numpy()

        interp_images = f(np.concatenate(interp_images, axis=0))
        
        save_file = os.path.join(args.save_location, args.save_filename)
        dict_for_save[dataset] = {
            "interp_images": interp_images,
            "gt_A": gt_A,
            "gt_B": gt_B,
            "indices_A": interp_idx[:,0],
            "indices_B": interp_idx[:,1],
            "ratio_A": all_ratio_A,
            "ratio_B": 1-all_ratio_A
            }
    if args.save_filename.endswith(".npy") or args.save_filename.endswith(".npz"):
        np.save(save_file, dict_for_save)
    else:
        assert args.save_filename.endswith(".pt") or args.save_filename.endswith(".pth"), "Save filename must end with .npy, .npz, .pt, or .pth"
        #convert to torch tensors
        dict_for_save = {dataset: {k: torch.as_tensor(v) for k,v in dict_for_save[dataset].items()} for dataset in dict_for_save}
        torch.save(dict_for_save, save_file)