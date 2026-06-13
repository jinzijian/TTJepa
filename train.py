import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
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

        halt_loss_weight = recurrent_cfg.get("halt_loss_weight", 0.0)
        if halt_loss_weight and "continue_logits" in out_pred:
            continue_logits = out_pred["continue_logits"]
            with torch.no_grad():
                pred_err = (
                    preds_all.detach() - tgt_emb.detach().unsqueeze(0)
                ).pow(2).mean(dim=-1)
                continue_target = torch.zeros_like(pred_err)
                halt_label_mode = recurrent_cfg.get("halt_label_mode", "improvement")

                if halt_label_mode == "improvement":
                    min_improvement = recurrent_cfg.get("halt_min_improvement", 0.0)
                    if pred_err.size(0) > 1:
                        improvement = pred_err[:-1] - pred_err[1:]
                        continue_target[:-1] = (
                            improvement > float(min_improvement)
                        ).float()
                elif halt_label_mode == "relative_improvement":
                    min_rel_improvement = recurrent_cfg.get(
                        "halt_min_relative_improvement", 0.0
                    )
                    if pred_err.size(0) > 1:
                        improvement = pred_err[:-1] - pred_err[1:]
                        relative = improvement / pred_err[:-1].clamp_min(1e-8)
                        continue_target[:-1] = (
                            relative > float(min_rel_improvement)
                        ).float()
                elif halt_label_mode == "error_threshold":
                    error_threshold = recurrent_cfg.get("halt_error_threshold", None)
                    if error_threshold is None:
                        raise ValueError(
                            "recurrent.halt_error_threshold must be set when "
                            "halt_label_mode='error_threshold'"
                        )
                    continue_target = (pred_err > float(error_threshold)).float()
                elif halt_label_mode == "residual_threshold":
                    residual_threshold = recurrent_cfg.get(
                        "halt_residual_threshold", None
                    )
                    if residual_threshold is None:
                        raise ValueError(
                            "recurrent.halt_residual_threshold must be set when "
                            "halt_label_mode='residual_threshold'"
                        )
                    continue_target = (
                        out_pred["residuals"].detach() > float(residual_threshold)
                    ).float()
                else:
                    raise ValueError(
                        f"Unsupported recurrent.halt_label_mode={halt_label_mode}"
                    )
                continue_target[-1] = 0.0

            pos_weight = recurrent_cfg.get("halt_pos_weight", None)
            if pos_weight is not None:
                pos_weight = torch.as_tensor(
                    float(pos_weight),
                    device=continue_logits.device,
                    dtype=continue_logits.dtype,
                )
            halt_loss = F.binary_cross_entropy_with_logits(
                continue_logits, continue_target, pos_weight=pos_weight
            )
            pred_loss = pred_loss + halt_loss_weight * halt_loss
            continue_prob_mean = torch.sigmoid(continue_logits.detach()).mean()
            continue_target_rate = continue_target.detach().mean()
        else:
            halt_loss = final_loss.new_zeros(())
            continue_prob_mean = final_loss.new_zeros(())
            continue_target_rate = final_loss.new_zeros(())

        output["pred_loss"] = pred_loss
        output["pred_loss_final"] = final_loss
        output["pred_loss_inter"] = inter_loss
        output["pred_loss_consistency"] = consistency_loss
        output["pred_loss_halt"] = halt_loss
        output["continue_prob_mean"] = continue_prob_mean
        output["continue_target_rate"] = continue_target_rate
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
        if torch.is_tensor(v)
        and (
            "loss" in k
            or k
            in {
                "residual_mean",
                "continue_prob_mean",
                "continue_target_rate",
            }
        )
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
