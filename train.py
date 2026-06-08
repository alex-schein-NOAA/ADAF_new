import os
import time
# import wandb
import random
import datetime
import argparse
import numpy as np

# from str2bool import str2bool
# from icecream import ic
from shutil import copyfile
# from apex import optimizers
from collections import OrderedDict

import torch
# import torch.cuda.amp as amp
import torch.amp as amp
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from utils.dataloader_multifiles import get_data_loader
# from utils.logging_utils import log_to_file
from utils.YParams import YParams

#################################


class Trainer:
    def count_parameters(self):
        count_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                count_params += p.numel()
        return count_params

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.set_device()

        # init gpu
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) #os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        print("device: %s" % self.device)

        # model init
        if params.nettype == "EncDec":
            from models.encdec import EncDec as model
        elif params.nettype == "EncDec_two_encoder":
            from models.encdec import EncDec_two_encoder as model
        else:
            raise Exception("not implemented")
        self.model = model(params).to(self.device)
        # self.model = model(params).to(local_rank) # for torchrun

        # Load data
        print(f"rank {world_rank}, begin data loader init")
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params,
            params.train_data_path,
            dist.is_initialized(),
            train=True,
        )
        (
            self.valid_data_loader,
            self.valid_dataset,
            self.valid_sampler,
        ) = get_data_loader(
            params,
            params.valid_data_path,
            dist.is_initialized(),
            train=True,
        )

        # optimizer
        if params.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=params.lr)
        elif params.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=params.lr)
        # elif params.optimizer_type == "FusedAdam":
        #     self.optimizer = optimizers.FusedAdam(
        #         self.model.parameters(), lr=params.lr)
        else:
            raise Exception("not implemented")

        if params.enable_amp:
            self.gscaler = amp.GradScaler()

        # DDP
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=[params.local_rank],
                find_unused_parameters=True,
            )
        self.iters = 0
        self.startEpoch = 0
        self.plot = False
        self.plot_img_path = None

        # Dynamical Learning rate
        if params.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=params.lr_reduce_factor,
                patience=20,
                mode="min",
            )
        elif params.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.max_epochs,
                last_epoch=self.startEpoch - 1,
            )
        else:
            self.scheduler = None

        # Resume train
        if params.resuming:
            print(f"Loading checkpoint from {params.best_checkpoint_path}")
            self.restore_checkpoint(params.best_checkpoint_path)

        self.epoch = self.startEpoch

        if params.log_to_screen:
            print(
                f"Number of trainable model parameters: \
                {self.count_parameters()}"
            )


    def train(self):
        if self.params.log_to_screen:
            print("Starting Training Loop...")

        # best_valid_obs_loss = 1.0e6
        best_train_loss = 1.0e6

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                # different batch on each GPU
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)
            start = time.time()

            # train one epoch
            tr_time, data_time, step_time, train_logs = self.train_one_epoch()
            self.plot = False
            self.plot_img_path = None
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.params.log_to_screen:
                print(f"Epoch: {epoch + 1}")
                print(f"train data time={data_time}")
                print(f"train per step time={step_time}")
                print(f"train loss: {train_logs['loss_field']}")
                print(f"learning rate: {current_lr}")

            # valid one epoch
            if (epoch != 0) and (epoch % self.params.valid_frequency == 0):
                valid_time, valid_logs = self.validate_one_epoch()

                if self.params.log_to_screen:
                    print(f"Epoch: {epoch + 1}")
                    print(f"Valid time={valid_time}")
                    print(f"Valid loss={valid_logs['valid_loss_field']}")

                # LR scheduler
                if self.params.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(valid_logs["valid_loss_field"])

            # save model
            if (
                self.world_rank == 0
                and epoch % self.params.save_model_freq == 0
                and self.params.save_checkpoint
            ):
                self.save_checkpoint(self.params.checkpoint_path)

            if self.world_rank == 0 and self.params.save_checkpoint:
                if train_logs["loss_field"] <= best_train_loss:
                    print(
                        "Loss improved from {} to {}".format(
                            best_train_loss, train_logs["loss_field"]
                        )
                    )
                    best_train_loss = train_logs["loss_field"]

                    start = time.time()
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    print(f"save model time: {time.time() - start}")

    def loss_function(
        self,
        pre_field,
        tar_field,
        tar_obs,
        tar_field_obs,
        field_mask=None,
        obs_tar_mask=None,
        mask_out_of_range=True,
    ):
        """
        pre_field: model's output
        tar_field: label, after normalization
        """

        # Create masked versions for field loss
        pre_field_masked = pre_field.clone()
        tar_field_masked = tar_field.clone()
        tar_field_obs_masked = tar_field_obs.clone()
        
        if mask_out_of_range:
            pre_field_masked = torch.masked_fill(
                input=pre_field_masked, mask=~field_mask, value=0
            )  # fill input with 0 where field_mask is False.
            tar_field_masked = torch.masked_fill(
                input=tar_field_masked, mask=~field_mask, value=0
            )  # fill input with 0 where field_mask is False.
            tar_field_obs_masked = torch.masked_fill(
                input=tar_field_obs_masked, mask=~field_mask, value=0
            )  # fill input with 0 where field_mask is False.

        # type 1 loss
        loss_field = F.mse_loss(
            pre_field_masked, tar_field_masked)
        loss_field_channel_wise = F.mse_loss(
            pre_field_masked, tar_field_masked, reduction="none")
        loss_field_channel_wise = torch.mean(
            loss_field_channel_wise, dim=(0, 2, 3))

        # type 2 loss
        loss_field_obs = F.mse_loss(
            pre_field_masked, tar_field_obs_masked)

        # type 3 loss - use fresh masks for obs loss
        pre_field_obs_masked = torch.masked_fill(
            input=pre_field.clone(), mask=~obs_tar_mask, value=0
        )  # fill input with 0 where obs_tar_mask is False.
        tar_obs_masked = torch.masked_fill(
            input=tar_obs.clone(), mask=~obs_tar_mask, value=0)
        loss_obs = F.mse_loss(
            pre_field_obs_masked, tar_obs_masked)
        loss_obs_channel_wise = F.mse_loss(
            pre_field_obs_masked, tar_obs_masked, reduction="none")
        loss_obs_channel_wise = torch.mean(
            loss_obs_channel_wise, dim=(0, 2, 3))

        return {
            "loss_field": loss_field,
            "loss_field_channel_wise": loss_field_channel_wise,
            "loss_obs": loss_obs,
            "loss_obs_channel_wise": loss_obs_channel_wise,
            "loss_field_obs": loss_field_obs,
        }

    def train_one_epoch(self):
        print("Training...")
        self.epoch += 1
        if self.params.resuming:
            self.resumeEpoch += 1
        tr_time = 0
        data_time = 0
        steps_in_one_epoch = 0
        loss_field = 0.0
        loss_obs = 0.0
        loss_field_obs = 0.0
        loss_field_channel_wise = torch.zeros(
            len(self.params.target_vars), device=self.device, dtype=float
        )
        loss_obs_channel_wise = torch.zeros(
            len(self.params.target_vars), device=self.device, dtype=float
        )

        self.model.train()
        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            steps_in_one_epoch += 1
            data_start = time.time()

            if self.params.nettype == "EncDec_two_encoder":
                (
                    inp,
                    inp_sate,
                    target_field,
                    target_obs,
                    target_field_obs,
                    inp_hrrr,
                    _,
                    _,
                    field_mask,
                    obs_tar_mask,
                ) = data
            if self.params.nettype == "EncDec":
                (
                    inp,
                    target_field,
                    target_obs,
                    target_field_obs,
                    inp_hrrr,
                    _,
                    _,
                    field_mask,
                    obs_tar_mask,
                ) = data

            data_time += time.time() - data_start
            tr_start = time.time()

            self.optimizer.zero_grad()
            # with amp.autocast(self.params.enable_amp):
            with amp.autocast(device_type=self.device.type):
                inp = inp.to(self.device, dtype=torch.float)
                inp_hrrr = inp_hrrr.to(self.device, dtype=torch.float)
                target_field = target_field.to(self.device, dtype=torch.float)
                target_obs = target_obs.to(
                    self.device, dtype=torch.float)
                target_field_obs = target_field_obs.to(
                    self.device, dtype=torch.float)
                field_mask = torch.as_tensor(
                    field_mask, dtype=torch.bool, device=self.device
                )
                obs_tar_mask = torch.as_tensor(
                    obs_tar_mask, dtype=torch.bool, device=self.device
                )
                
                if self.params.nettype == "EncDec":
                    gen = self.model(inp)
                if self.params.nettype == "EncDec_two_encoder":
                    inp_sate = inp_sate.to(self.device, dtype=torch.float)
                    gen = self.model(inp, inp_sate)
                gen.to(self.device, dtype=torch.float)

                loss = self.loss_function(
                    pre_field=gen,
                    tar_field=target_field,
                    tar_obs=target_obs,
                    tar_field_obs=target_field_obs,
                    field_mask=field_mask,
                    obs_tar_mask=obs_tar_mask,
                )

                loss_field += loss["loss_field"].detach().item()
                loss_obs += loss["loss_obs"].detach().item()
                loss_field_obs += loss["loss_field_obs"].detach().item()
                loss_field_channel_wise += loss["loss_field_channel_wise"].detach()
                loss_obs_channel_wise += loss["loss_obs_channel_wise"].detach()

                if self.params.target == "obs":
                    # target: sparse observations
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_obs"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_obs"].backward()
                        self.optimizer.step()
                if self.params.target == "analysis":
                    # target: grided fields
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_field"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_field"].backward()
                        self.optimizer.step()
                if self.params.target == "analysis_obs":
                    # target: grided fields + sparse observations
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_field_obs"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_field_obs"].backward()
                        self.optimizer.step()

                if self.params.enable_amp:
                    self.gscaler.update()

                tr_time += time.time() - tr_start

        logs = {
            "loss_field": loss_field / steps_in_one_epoch,
            "loss_obs": loss_obs / steps_in_one_epoch,
            "loss_field_obs": loss_field_obs / steps_in_one_epoch,
        }
        for i_, var_ in enumerate(self.params.target_vars):
            tmp_var_1 = loss_obs_channel_wise[i_] / steps_in_one_epoch
            tmp_var_2 = loss_field_channel_wise[i_] / steps_in_one_epoch
            logs[f"loss_obs_{var_}"] = tmp_var_1
            logs[f"loss_field_{var_}"] = tmp_var_2

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                logs[key] = torch.tensor(logs[key], device=self.device)
                dist.all_reduce(logs[key])
                logs[key] = float(logs[key] / dist.get_world_size())

        # time of one step in epoch
        step_time = tr_time / steps_in_one_epoch

        return tr_time, data_time, step_time, logs

    def validate_one_epoch(self):
        print("validating...")
        self.model.eval()

        valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
        valid_loss_field = valid_buff[0].view(-1)
        valid_loss_obs = valid_buff[1].view(-1)
        valid_loss_field_obs = valid_buff[2].view(-1)
        valid_steps = valid_buff[3].view(-1)

        valid_start = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                self.plot = False
                self.plot_img_path = False

                if self.params.nettype == "EncDec_two_encoder":
                    (
                        inp,
                        inp_sate,
                        target_field,
                        target_obs,
                        target_field_obs,
                        inp_hrrr,
                        _,
                        _,
                        field_mask,
                        obs_tar_mask,
                    ) = data
                if self.params.nettype == "EncDec":
                    (
                        inp,
                        target_field,
                        target_obs,
                        target_field_obs,
                        inp_hrrr,
                        _,
                        _,
                        field_mask,
                        obs_tar_mask,
                    ) = data

                inp = inp.to(
                    self.device, dtype=torch.float)
                inp_hrrr = inp_hrrr.to(
                    self.device, dtype=torch.float)
                target_field = target_field.to(
                    self.device, dtype=torch.float)
                target_obs = target_obs.to(
                    self.device, dtype=torch.float)
                target_field_obs = target_field_obs.to(
                    self.device, dtype=torch.float)
                field_mask = field_mask.to(
                    self.device, dtype=torch.bool)
                obs_tar_mask = obs_tar_mask.to(
                    self.device, dtype=torch.bool)

                if self.params.nettype == "EncDec":
                    gen = self.model(inp)
                if self.params.nettype == "EncDec_two_encoder":
                    inp_sate = inp_sate.to(
                        self.device, dtype=torch.float)
                    gen = self.model(inp, inp_sate)
                gen.to(self.device, dtype=torch.float)

                loss = self.loss_function(
                    pre_field=gen,
                    tar_field=target_field,
                    tar_obs=target_obs,
                    tar_field_obs=target_field_obs,
                    field_mask=field_mask,
                    obs_tar_mask=obs_tar_mask,
                )

                valid_steps += 1.0
                valid_loss_field += loss["loss_field"]
                valid_loss_obs += loss["loss_obs"]
                valid_loss_field_obs += loss["loss_field_obs"]

        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        valid_buff[0:3] = valid_buff[0:3] / valid_buff[3]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        logs = {
            "valid_loss_field": valid_buff_cpu[0],
            "valid_loss_obs": valid_buff_cpu[1],
            "valid_loss_field_obs": valid_buff_cpu[2],
        }

        valid_time = time.time() - valid_start

        return valid_time, logs

    def load_model(self, model_path):
        if self.params.log_to_screen:
            print("Loading the model weights from {}".format(model_path))

        checkpoint = torch.load(
            model_path, map_location="cuda:{}".format(self.params.local_rank)
        )

        if dist.is_initialized():
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            new_model_state = OrderedDict()
            if "model_state" in checkpoint:
                model_key = "model_state"
            else:
                model_key = "state_dict"

            for key in checkpoint[model_key].keys():
                if "module." in key:
                    # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model.load_state_dict(new_model_state)
            self.model.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        if not model:
            model = self.model

        print("Saving model to {}".format(checkpoint_path))
        torch.save(
            {
                "iters": self.iters,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda:{}".format(self.params.local_rank)
        )
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except ValueError:
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
        self.resumeEpoch = 0
        if self.params.resuming:
            # restore checkpoint is used for finetuning as well as resuming.
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # uses config specified lr.
            for g in self.optimizer.param_groups:
                g["lr"] = self.params.lr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#####

class Params:
    def __init__(self):
        # --- Duplicate variables (retaining the OLD values) ---
        self.img_size_x = 1280 #960
        self.img_size_y = 500 #480 #512 

        # --- Unique variables from the new Dict ---
        self.upscale = 1
        self.in_chans = 21 #8
        self.out_chans = 5 #4
        self.window_size = 4
        self.patch_size = 5
        self.num_feat = 64
        self.drop_rate = 0.1
        self.drop_path_rate = 0.1
        self.attn_drop_rate = 0.1
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False
        self.resi_connection = "1conv"
        self.qkv_bias = True
        self.qk_scale = None
        self.img_range = 1.0
        self.depths = [3]
        self.embed_dim = 64
        self.num_heads = [4]
        self.mlp_ratio = 2

        # --- Unique variables from the intermediate class ---
        self.batch_size = 12 #4
        self.num_data_workers = 4
        self.n_in_channels = 17
        self.n_out_channels = 5
        self.inp_hrrr_vars = ['hrrr_sp', 'hrrr_q', 'hrrr_t', 'hrrr_u10', 'hrrr_v10']
        self.inp_obs_vars = ["sta_p", "sta_q", "sta_t", "sta_u10", "sta_v10"]
        self.field_tar_vars = ['rtma_sp', 'rtma_q', 'rtma_t', 'rtma_u10', 'rtma_v10']
        self.obs_time_window = 3
        self.hold_out_obs = True
        self.hold_out_obs_ratio = 0.1
        self.obs_mask_seed = 0
        self.learn_residual = True

        # --- Base variables from the original script ---
        self.target = "analysis_obs"
        self.lr_reduce_factor = 0.9
        self.max_epochs = 10 #2 #1200
        self.world_size = 1
        self.lr = 0.0002
        self.local_rank = -1
        self.nettype = "EncDec"
        self.device = "GPU"
        self.global_batch_size = 8
        self.enable_amp = False
        self.experiment_dir = f"/scratch3/BMC/wrfruc/aschein/ADAF_new/data/exp"
        self.checkpoint_path = f"/scratch3/BMC/wrfruc/aschein/ADAF_new/data/exp/training_checkpoints/ckpt.tar"
        self.best_checkpoint_path = f"/scratch3/BMC/wrfruc/aschein/ADAF_new/data/exp/training_checkpoints/best_ckpt.tar"
        self.resuming = False
        self.name = None
        self.entity = "your entity"
        self.project = "your project"
        self.group = None
        self.log_to_wandb = False
        self.log_to_screen = True

        # --- YParams / YAML configuration defaults ---
        self.train_data_path = f"/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/train_data/"
        self.valid_data_path = f"/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/valid_data/"
        self.optimizer_type = "Adam"
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_patience = 20 #added 2026-06-03
        self.valid_frequency = 1
        self.save_model_freq = 1
        self.save_checkpoint = True
        self.target_vars = ['rtma_sp', 'rtma_q', 'rtma_t', 'rtma_u10', 'rtma_v10']

    # Simple helpers so the script's bracket notation works
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    # Required for the `params.params.items()` line in the script
    @property
    def params(self):
        return self.__dict__

    # Required for `params.log()`
    def log(self):
        for key, value in sorted(self.__dict__.items()):
            print(f"{key}: {value}")


if __name__ == "__main__":
    
    params = Params()
    
    # Get SLURM info for DDP
    world_size = int(os.environ.get("WORLD_SIZE")) #This may not have been the right one to comment out - if job fails, come back and fix this
    world_rank = int(os.environ.get("SLURM_PROCID", 0))
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    # world_size = int(os.environ.get("SLURM_NTASKS", 2))
    # ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE"), 2)
    #num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    ## Unsure if this is good - seems like it's needed - basically in-line version of setup_distributed
    dist.init_process_group(backend="nccl")
    # rank = dist.get_rank() #does this conflict with, or is the same as, local_rank?
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    params.local_rank = local_rank  # <-- ADD THIS LINE

    rank = dist.get_rank()  # from world_rank in original train.py
    
    print(f"local_rank = {local_rank}")
    if rank==0:
        print(f"[Rank {rank}] world_size = {world_size} | world_rank = {world_rank} | num_nodes = {num_nodes}")# | ntasks_per_node = {ntasks_per_node}")

    trainer = Trainer(params, world_rank)
    trainer.train()
