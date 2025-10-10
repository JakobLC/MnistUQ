
import numpy as np
from tqdm.auto import tqdm
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import _sigmoid_calibration
from sklearn.neighbors import NearestNeighbors

def knn_classification_metrics(out_dict, k=10, num_classes=None, metric="euclidean"):
    """
    Leave-one-out KNN on the full dataset.

    Parameters
    ----------
    X : (N, D) array-like
    y : (N,) int array-like, labels in [0..C-1]
    k : int, number of neighbors (excluding self)
    num_classes : int or None (auto -> max(y)+1)
    metric : str, currently only 'euclidean' is supported

    Returns
    -------
    results : dict with keys
        'per_class_acc_majority' : (C,) float array
        'per_class_acc_soft'     : (C,) float array
        'weighted_acc_majority'  : float
        'weighted_acc_soft'      : float
        'support'                : (C,) int array (class counts)
    """
    iid_mask = out_dict["eval"]["iid_mask"]
    vae_mask = out_dict["eval"]["vae_mask"]
    AU = out_dict["eval"]["unc"]["AU"].numpy()
    EU = out_dict["eval"]["unc"]["EU"].numpy()
    masks = [np.where(~vae_mask & iid_mask)[0],
                np.where(~vae_mask & ~iid_mask)[0],
                np.where(vae_mask & iid_mask)[0],
                np.where(vae_mask & ~iid_mask)[0]]

    x = [np.log10(AU[mask]) for mask in masks]
    y = [np.log10(EU[mask]) for mask in masks]
    X = np.stack([np.concatenate(x,axis=0),np.concatenate(y,axis=0)],axis=1)
    y = np.array(sum([[i for _ in range(len(xi))] for i,xi in enumerate(x)],[]))
    N = X.shape[0]
    if N == 0:
        raise ValueError("Empty dataset.")
    if metric != "euclidean":
        raise ValueError("Only Euclidean distance supported in this implementation.")

    if num_classes is None:
        num_classes = int(y.max()) + 1

    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= N:
        raise ValueError("k must be less than the number of samples (leave-one-out requires k < N).")

    nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    # drop self (the first neighbor should be itself; if duplicates exist, still okay to drop col 0)
    idxs = idxs[:, 1:k+1]
    # --- Gather neighbor labels ---
    neigh_labels = y[idxs]  # (N, k)

    # --- Counts per class in each neighborhood ---
    # We'll use np.bincount row-wise.
    counts = np.zeros((N, num_classes), dtype=np.int32)
    for c in range(num_classes):
        counts[:, c] = (neigh_labels == c).sum(axis=1)

    # Majority-vote prediction (ties → lowest class id)
    maj_pred = counts.argmax(axis=1)

    # Correctness indicators
    correct_majority = (maj_pred == y).astype(np.float64)

    # Soft accuracy (probability of picking the correct class at random from the
    # neighbor class-frequency distribution): p = count(true_class)/k
    true_counts = counts[np.arange(N), y]
    soft_acc = true_counts / float(k)

    # --- Per-class accuracies ---
    support = np.bincount(y, minlength=num_classes)
    per_class_acc_majority = np.zeros(num_classes, dtype=np.float64)
    per_class_acc_soft = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        mask = (y == c)
        n_c = support[c]
        if n_c > 0:
            per_class_acc_majority[c] = correct_majority[mask].mean()
            per_class_acc_soft[c] = soft_acc[mask].mean()
        else:
            per_class_acc_majority[c] = np.nan  # no samples of this class
            per_class_acc_soft[c] = np.nan

    # --- Weighted overall accuracies (weights = class supports) ---
    total = support.sum()
    # Only include classes with support > 0
    nonzero = support > 0
    weighted_acc_majority = float((support[nonzero] * per_class_acc_majority[nonzero]).sum() / total)
    weighted_acc_soft = float((support[nonzero] * per_class_acc_soft[nonzero]).sum() / total)

    return {
        "per_class_acc_majority": per_class_acc_majority,
        "per_class_acc_soft": per_class_acc_soft,
        "weighted_acc_majority": weighted_acc_majority,
        "weighted_acc_soft": weighted_acc_soft,
        "support": support,
    }

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



def calib_stats(out_dict,iid_mask=None,vae_mask=None, do_plot=False, is_AU2=None):
    """Computes ECE with no temperature scaling, ECE with optimal temperature scaling, 
    for both NLL and ECE optimal temperatures.
    """
    if is_AU2 is None:
        assert "is_AU2" in out_dict["eval"], "Must provide is_AU2 flag if not present in out_dict"
        is_AU2 = out_dict["eval"]["is_AU2"]
    vae_mask = out_dict["eval"].get("vae_mask", vae_mask)
    iid_mask = out_dict["eval"].get("iid_mask", iid_mask)
    if (iid_mask is None) and (vae_mask is None):
        joint_mask = slice(None)
    elif (iid_mask is None):
        joint_mask = torch.logical_not(vae_mask)
    elif (vae_mask is None):
        joint_mask = iid_mask
    else:
        joint_mask = torch.logical_and(iid_mask, torch.logical_not(vae_mask))
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
        if iid_mask is None:
            eval_mask = slice(None)
        else:
            eval_mask = iid_mask
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
        assert "iid_mask" in out_dict["eval"], "If mask is not provided, out_dict must contain 'iid_mask'"
        mask = out_dict["eval"]["iid_mask"]
    if vae_mask is None:
        assert "vae_mask" in out_dict["eval"], "If vae_mask is not provided, out_dict must contain 'vae_mask'"
        vae_mask = out_dict["eval"]["vae_mask"]
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
                       "ncc_all": np.corrcoef(unc2, entropy_gt)[0,1],
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

def _fit_platt_scaling_binary(scores, labels):
    pos = np.count_nonzero(labels > 0.5)
    neg = np.count_nonzero(labels <= 0.5)
    if pos == 0 or neg == 0:
        p = np.clip(labels.mean(), 1e-4, 1 - 1e-4)
        return 0.0, float(np.log((1 - p) / p))
    return _sigmoid_calibration(scores, labels)


def _fit_platt_scaling_weighted(scores, fractional_correct):
    eps = 1e-12
    pos_mask = fractional_correct > eps
    neg_mask = (1.0 - fractional_correct) > eps

    if not np.any(pos_mask) or not np.any(neg_mask):
        p = np.clip(fractional_correct.mean(), 1e-4, 1 - 1e-4)
        return 0.0, float(np.log((1 - p) / p))

    y_true = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
    expanded_scores = np.concatenate([scores[pos_mask], scores[neg_mask]])
    weights = np.concatenate([fractional_correct[pos_mask], (1.0 - fractional_correct[neg_mask])])

    return _sigmoid_calibration(expanded_scores, y_true, sample_weight=weights)


def _fit_platt_scaling(uncertainties, correct):
    uncertainties = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(uncertainties) & np.isfinite(correct)
    if not np.any(finite_mask):
        return 0.0, 0.0
    uncertainties = uncertainties[finite_mask]
    correct = correct[finite_mask]

    scores = -uncertainties
    eps = 1e-12
    if np.all((correct <= eps) | (correct >= 1 - eps)):
        labels = (correct >= 0.5).astype(float)
        return _fit_platt_scaling_binary(scores, labels)
    else:
        return _fit_platt_scaling_weighted(scores, correct)


def _apply_platt_scaling(uncertainties, a, b):
    scores = -np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    logits = scores * a + b
    logits = np.clip(logits, -60.0, 60.0)
    confidences = 1.0 / (1.0 + np.exp(logits))
    return confidences


def _calibration_bins(confidences, correct, n_bins=20):
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(confidences, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    bin_total = np.bincount(binids, minlength=n_bins)
    sum_conf = np.bincount(binids, weights=confidences, minlength=n_bins)
    sum_correct = np.bincount(binids, weights=correct, minlength=n_bins)
    nonzero = bin_total > 0
    bin_conf = np.zeros_like(sum_conf)
    bin_acc = np.zeros_like(sum_correct)
    bin_conf[nonzero] = sum_conf[nonzero] / bin_total[nonzero]
    bin_acc[nonzero] = sum_correct[nonzero] / bin_total[nonzero]
    discrepancies = np.abs(bin_acc - bin_conf)
    num_nonzero = int(np.count_nonzero(nonzero))
    if num_nonzero == 0:
        ace = 0.0
    else:
        ace = float(discrepancies[nonzero].mean())
    total = bin_total.sum()
    if total == 0:
        ece_val = 0.0
    else:
        weights = bin_total / total
        ece_val = float(np.dot(discrepancies, weights))
    return ace, ece_val, {
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_counts": bin_total,
        "num_nonzero": num_nonzero,
    }


def calib_stats_values(
    out_dict,
    iid_mask=None,
    vae_mask=None,
    is_AU2=None,
    n_bins=20,
    use_monte_carlo=False,
    mc_samples=20,
    random_state=None,
    do_plot=False,
):
    if is_AU2 is None:
        assert "is_AU2" in out_dict["eval"], "Must provide is_AU2 flag if not present in out_dict"
        is_AU2 = out_dict["eval"]["is_AU2"]
    vae_mask = out_dict["eval"].get("vae_mask", vae_mask)
    iid_mask = out_dict["eval"].get("iid_mask", iid_mask)
    mc_samples = int(mc_samples)
    if mc_samples <= 0:
        mc_samples = 1
    if (iid_mask is None) and (vae_mask is None):
        joint_mask = slice(None)
    elif iid_mask is None:
        joint_mask = torch.logical_not(vae_mask)
    elif vae_mask is None:
        joint_mask = iid_mask
    else:
        joint_mask = torch.logical_and(iid_mask, torch.logical_not(vae_mask))

    if is_AU2:
        eval_mask = iid_mask if iid_mask is not None else slice(None)
    else:
        eval_mask = joint_mask

    probs_gts = out_dict["eval"]["probs_gts"].to(torch.float32)
    ensemble_probs = out_dict["eval"]["probs"].to(torch.float32).mean(dim=0)

    probs_gts_eval = probs_gts[eval_mask]
    ensemble_probs_eval = ensemble_probs[eval_mask]
    pred_labels = ensemble_probs_eval.argmax(dim=1)

    idx = torch.arange(probs_gts_eval.shape[0], device=probs_gts_eval.device)
    fractional_correct = probs_gts_eval[idx, pred_labels]

    if use_monte_carlo:
        generator = torch.Generator(device=probs_gts_eval.device)
        if random_state is not None:
            generator.manual_seed(int(random_state))
        sampled_labels = torch.multinomial(
            probs_gts_eval,
            num_samples=int(mc_samples),
            replacement=True,
            generator=generator,
        )
        correct_mc = (sampled_labels == pred_labels.unsqueeze(1)).to(torch.float32)
        correct_values = correct_mc.reshape(-1).cpu().numpy()
        effective_samples = correct_values.size
    else:
        correct_values = fractional_correct.cpu().numpy()
        effective_samples = correct_values.size

    stats = {}
    if do_plot:
        plt.figure(figsize=(12, 4))

    for k, (metric_name, unc_tensor) in enumerate(out_dict["eval"]["unc"].items()):
        unc_eval = unc_tensor[eval_mask]
        unc_eval_np = np.asarray(unc_eval, dtype=np.float64).reshape(-1)
        if use_monte_carlo:
            unc_fit = np.repeat(unc_eval_np, int(mc_samples))
        else:
            unc_fit = unc_eval_np

        a, b = _fit_platt_scaling(unc_fit, correct_values)
        confidences = _apply_platt_scaling(unc_fit, a, b)
        ace_val, ece_val, bin_dict = _calibration_bins(confidences, correct_values, n_bins=n_bins)

        stats[metric_name] = {
            "ace": float(ace_val),
            "ece": float(ece_val),
            "a": a,
            "b": b
        }
        if do_plot:
            plt.subplot(1, 3, k + 1)
            delta = 1/40
            y_steps = np.repeat(bin_dict["bin_acc"], 3)
            x_steps = sum([[i1,i2,float("nan")] for (i1,i2) in zip(bin_dict["bin_conf"]-delta,bin_dict["bin_conf"]+delta)],[])
            y_steps_ideal = sum([[i,i,float("nan")] for i in bin_dict["bin_conf"]],[])


            ratio_use = 1
            randperm = np.random.permutation(len(confidences))
            idx = randperm[:int(ratio_use*len(confidences))]

            plt.plot(confidences[idx], correct_values[idx], 'o', alpha=0.1)
            plt.plot(x_steps, y_steps, 'r-', linewidth=3)
            plt.plot(x_steps, y_steps_ideal, 'k-', linewidth=2)
            plt.xlabel("Mapped uncertainty")
            plt.ylabel("GT prob on predicted class")
            plt.title(f"{metric_name}\nECE={ece_val:.4f}, ACE={ace_val:.4f}")

    return stats

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
        
def ood_stats(out_dict, do_plot=True, is_ood=None):
    """Computes AUROC for detecting OOD samples (2,3,5) given uncertainty measures.
    """
    if is_ood is None:
        assert "iid_mask" in out_dict["eval"], "If is_ood is not provided, out_dict must contain 'iid_mask'"
        is_ood = ~out_dict["eval"]["iid_mask"].numpy()
    stats_dict = {}
    n = len(out_dict["eval"]["unc"])

    if do_plot: plt.figure(figsize=(5*n,4))

    for i,(k,v) in enumerate(out_dict["eval"]["unc"].items()):
        if do_plot: plt.subplot(1,n,i+1)
        auc, fpr, tpr, thresholds = auroc(v, is_ood, do_plot=do_plot)
        stats_dict[k] = {"auc": auc, "ood_entropy": v[is_ood].mean(), "iid_entropy": v[~is_ood].mean()}
        if do_plot: 
            plt.title(f"{k}, AUROC: {auc:.4f}")
    return stats_dict