import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    recurrent_cfg = cfg.get("recurrent", None)
    recurrent_enabled = (
        recurrent_cfg is not None and recurrent_cfg.get("enabled", False)
    )

    if recurrent_enabled:
        out_pred = self.model.predict(
            ctx_emb,
            ctx_act,
            return_all=True,
            predictor_depth=recurrent_cfg.get("max_depth", None),
        )
        preds_all = out_pred["preds"]
        final_target = (
            tgt_emb
            if recurrent_cfg.get("final_no_stopgrad", True)
            else tgt_emb.detach()
        )
        final_loss = (preds_all[-1] - final_target).pow(2).mean()

        if preds_all.size(0) > 1:
            inter_target = (
                tgt_emb.detach()
                if recurrent_cfg.get("intermediate_stopgrad", True)
                else tgt_emb
            )
            inter_loss = (preds_all[:-1] - inter_target.unsqueeze(0)).pow(2).mean()
        else:
            inter_loss = final_loss.new_zeros(())

        pred_loss = final_loss + recurrent_cfg.get("inter_loss_weight", 0.0) * inter_loss
        consistency_weight = recurrent_cfg.get("consistency_weight", 0.0)
        if consistency_weight and preds_all.size(0) > 1:
            consistency_loss = (preds_all[1:] - preds_all[:-1].detach()).pow(2).mean()
            pred_loss = pred_loss + consistency_weight * consistency_loss
        else:
            consistency_loss = final_loss.new_zeros(())

        output["pred_loss"] = pred_loss
        output["pred_loss_final"] = final_loss
        output["pred_loss_inter"] = inter_loss
        output["pred_loss_consistency"] = consistency_loss
        if recurrent_cfg.get("log_depth_stats", True):
            output["residual_mean"] = out_pred["residuals"].detach().mean()
    else:
        pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # LeWM loss
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if torch.is_tensor(v) and ("loss" in k or k == "residual_mean")
    }
    self.log_dict(metrics_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
