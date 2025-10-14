import numpy as np

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

def get_aug_models_heavy(augs=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
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

def get_epochs_models_heavy(epochs=[10,20,30,40,50,60,70,80,90,100]):
    """SETUPS:
        - Sweeping epochs for heavy models
    """
    base_cnn = {"model": "models.CNNDeluxe", "epochs": 50, "model_args": {"base_channels": 32, "num_blocks": 3, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "epochs": 50, "model_args": {"width": 256, "num_layers": 8}}
    model_cfgs = {}
    for e in epochs:
        model_cfgs[f"CNN_e{e}_H_aug0"] = {"epochs": e}
        model_cfgs[f"MLP_e{e}_H_aug0"] = {"epochs": e}
        model_cfgs[f"CNN_e{e}_H_aug1"] = {"epochs": e, "data_aug": 1}
        model_cfgs[f"MLP_e{e}_H_aug1"] = {"epochs": e, "data_aug": 1}
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

def get_heavy_weight_decay_models(epochs=50, weight_decays=[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    """SETUPS:
        - Sweeping weight decay for heavy models
    """
    base_cnn = {"model": "models.CNNDeluxe", "epochs": epochs, "model_args": {"base_channels": 32, "num_blocks": 3, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "epochs": epochs, "model_args": {"width": 256, "num_layers": 8}}
    model_cfgs = {}
    for wd in weight_decays:
        model_cfgs[f"CNN_wd{wd}_H_aug0"] = {"weight_decay": wd}
        model_cfgs[f"MLP_wd{wd}_H_aug0"] = {"weight_decay": wd}
        model_cfgs[f"CNN_wd{wd}_H_aug1"] = {"weight_decay": wd, "data_aug": 1}
        model_cfgs[f"MLP_wd{wd}_H_aug1"] = {"weight_decay": wd, "data_aug": 1}
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

"""experiments:

1. epoch based sweeps. Heavy model with epochs [10,20, ..., 200], aug=[0,0.25,0.5,0.75,1]
    1. No lr decay ema/normal
    2. lr_decay=0.9 ema/normal
X 2. batch_size sweep [4,8,16,32,64,128,256,512,1024,2048] for heavy models aug=[0,1]
X 3. continuation of epoch_H_aug0 and epoch_H_aug1 for epochs in [120,140, ... 200]"""

def get_batch_size_models_heavy(batch_sizes=[32,64,128,256,512,1024,2048]):
    """SETUPS:
        - Sweeping batch sizes for heavy models
    """
    base_cnn = {"model": "models.CNNDeluxe", "epochs": 50, "model_args": {"base_channels": 32, "num_blocks": 3, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "epochs": 50, "model_args": {"width": 256, "num_layers": 8}}
    model_cfgs = {}
    for bs in batch_sizes:
        model_cfgs[f"CNN_bs{bs}_H_aug0"] = {"batch_size": bs}
        model_cfgs[f"MLP_bs{bs}_H_aug0"] = {"batch_size": bs}
        model_cfgs[f"CNN_bs{bs}_H_aug1"] = {"batch_size": bs, "data_aug": 1}
        model_cfgs[f"MLP_bs{bs}_H_aug1"] = {"batch_size": bs, "data_aug": 1}
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

def get_aug_epoch_models(augs=[0,1], 
                         epoch_list=np.logspace(1,3,30).astype(int).tolist(),
                         ):
    base_cnn = {"model": "models.CNNDeluxe", "epochs": epoch_list, "also_save_ema_ckpts": True, 
                "model_args": {"base_channels": 32, "num_blocks": 3, "num_downsamples": 3}}
    base_mlp = {"model": "models.MLPDeluxe", "epochs": epoch_list, "also_save_ema_ckpts": True,
                "model_args": {"width": 256, "num_layers": 8}}
    model_cfgs = {}
    for a in augs:
        model_cfgs[f"CNN_aug{a}_H"] = {"data_aug": a}
        model_cfgs[f"MLP_aug{a}_H"] = {"data_aug": a}
    # merge with base config
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

"""
        print("Small test to see if everything works with epoch sweeps and epoch_mode")
        epoch_list = [1,2,4]
        model_setups = {
            "CNN_test_aug0":   {"model": "CNN1", "epochs": epoch_list, "data_aug": 0, "also_save_ema_ckpts": True},
        uncertainty_setups = {
            "AU2_EU": {"ignore_digits": [2, 3, 5], "ambiguous_vae_samples": True},
        }
        train_ensembles(model_setups, uncertainty_setups, skip_existing=True, save_intermediate=True)


"""