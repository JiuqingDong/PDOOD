#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
from scipy import stats
from scipy.stats import entropy

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer
from collections import deque

from ..engine.evaluator import Evaluator, get_and_print_results
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.plot_util import plot_distribution, plot_distribution_1
from ..utils.train_utils import AverageMeter, gpu_mem_usage

from src.data import loader as data_loader

import numpy as np
import torch.nn.functional as F
from kNN_Eval import kNN_OOD


def setup_log():
    log = logging.get_logger("visual_prompt")
    log.debug(f"#####################")
    return log

logger = logging.get_logger("visual_prompt")

class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )
        self.log = setup_log()
        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            model.load_state_dict(torch.load(cfg.MODEL.WEIGHT_PATH))
            # # only use this for vtab in-domain experiments
            # checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            # self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            # logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

        if self.cfg.DATA.NAME == 'plant_village':
            self.ood_dataset_dic = {
                "apple"     : data_loader.construct_ood_loader(cfg, "ood_apple" ),
                "corn"      : data_loader.construct_ood_loader(cfg, "ood_corn"  ),
                "grape"     : data_loader.construct_ood_loader(cfg, "ood_grape" ),
                "potato"    : data_loader.construct_ood_loader(cfg, "ood_potato"),
                "tomato": data_loader.construct_ood_loader(cfg, "ood_tomato"    ),
                "others"    : data_loader.construct_ood_loader(cfg, "ood_others"),
            }
        elif self.cfg.DATA.NAME == 'paddy10':
            self.ood_dataset_dic = {
                "PADDY": data_loader.construct_ood_loader(cfg, "ood_paddy"),
            }
        elif self.cfg.DATA.NAME == 'cotton':
            self.ood_dataset_dic = {
                "COTTON": data_loader.construct_ood_loader(cfg, "ood_cotton"),
            }
        elif self.cfg.DATA.NAME == 'strawberry':
            self.ood_dataset_dic = {
                "STRAWBERRY": data_loader.construct_ood_loader(cfg, "ood_strawberry"),
            }
        elif self.cfg.DATA.NAME == 'mango':
            self.ood_dataset_dic = {
                "MANGO": data_loader.construct_ood_loader(cfg, "ood_mango"),
            }
        elif self.cfg.DATA.NAME == 'pvtc':
            self.ood_dataset_dic = {
                "PVTC": data_loader.construct_ood_loader(cfg, "ood_pvtc"),
            }
        elif self.cfg.DATA.NAME == 'pvtg':
            self.ood_dataset_dic = {
                "PVTG": data_loader.construct_ood_loader(cfg, "ood_pvtg"),
            }
        elif self.cfg.DATA.NAME == 'pvts':
            self.ood_dataset_dic = {
                "PVTS": data_loader.construct_ood_loader(cfg, "ood_pvts"),
            }
        else:
            print("===========ERROR===========")

        self.train_copy_loader = data_loader.construct_train_copy_loader(cfg)

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs, feature = self.model(inputs)  # (batchsize, num_cls)

            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs, feature

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            if epoch == 0:
                self.ood_evaluator(test_loader, epoch=epoch)

            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break

                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch * total_data * (
                                    total_epoch - epoch - 1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc >= best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
                if best_metric >= 0.9:
                    if test_loader is not None:
                        self.ood_evaluator(test_loader, epoch=epoch)

            else:
                patience += 1
                if epoch == total_epoch - 1:
                    self.ood_evaluator(test_loader, epoch=epoch)

            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=True):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs, _ = self.forward_one_batch(X, targets, False)

            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        # print("joint_logits", joint_logits.shape, joint_logits)     # （N_Images, Classes)
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")

        if save and self.cfg.MODEL.SAVE_CKPT:
            model_state = self.model.state_dict()
            out_path = os.path.join(self.cfg.OUTPUT_DIR, f"{test_name}_model.pth")
            torch.save(model_state, out_path)
            logger.info(f"Saved model for {test_name} at {out_path}")

    @torch.no_grad()
    def feature_save(self, data_loader, epoch, prefix=None):
        """save the features as npy"""
        # initialize features and target
        features = []
        for idx, input_data in enumerate(data_loader):
            X, targets = self.get_input(input_data)
            _, _, feature = self.forward_one_batch(X, targets, False)
            features.append(feature.cpu().detach())  # 将feature从GPU复制到CPU
        all_feature = torch.cat(features, dim=0)
        features_array = np.array(all_feature)

        # 保存为npy文件
        output_file = os.path.join(self.cfg.OUTPUT_DIR, f"epoch_{epoch}_{prefix}_feature.npy")
        np.save(output_file, features_array)
        print(f"Feature was saved as {output_file}")

    @torch.no_grad()
    def ood_evaluator(self, data_loader, epoch = None, T = 1):
        """ood_evaluator"""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        # add T as hyper Param
        # initialize features and targets
        InD_logits = []
        InD_score_entropy = []
        InD_score_var = []
        InD_score_msp = []
        InD_max_logits = []

        for idx, input_data in enumerate(data_loader):
            X, targets = self.get_input(input_data)
            _, outputs, _feature = self.forward_one_batch(X, targets, False)

            # print("outputs",outputs, outputs.shape)     # [Batch_size, IND_Classes]
            if _ == -1:
                return

            smax = to_np(F.softmax(outputs/T, dim=1))
            # Energy
            InD_logits.append(-to_np((T*torch.logsumexp(outputs / T, dim=1))))
            # entropy
            InD_score_entropy.append(entropy(smax, axis=1))
            # var
            InD_score_var.append(-np.var(smax, axis=1))
            # msp
            InD_score_msp.append(-np.max(smax, axis=1))
            # max_logits
            max_logit = to_np(outputs/T)
            InD_max_logits.append(-np.max(max_logit, axis=1))

        # total_testimages x num_classes
        InD_logits = concat(InD_logits)[:len(data_loader.dataset)].copy()
        InD_score_entropy = concat(InD_score_entropy)[:len(data_loader.dataset)].copy()
        InD_score_var = concat(InD_score_var)[:len(data_loader.dataset)].copy()
        InD_score_msp = concat(InD_score_msp)[:len(data_loader.dataset)].copy()
        InD_max_logits = concat(InD_max_logits)[:len(data_loader.dataset)].copy()

        self.feature_save(self.train_copy_loader, epoch, prefix='InD_train')
        self.feature_save(data_loader, epoch, prefix='InD_test')

        for name, ood_loader in self.ood_dataset_dic.items():
            # initialize features and target
            OOD_logits = []
            OOD_score_entropy = []
            OOD_score_var = []
            OOD_score_msp = []
            OOD_max_logits = []

            auroc_list, aupr_list, fpr_list = [], [], []
            self.feature_save(ood_loader, epoch, prefix='OOD_' + name)

            for idx, input_data in enumerate(ood_loader):
                X, targets = self.get_input(input_data)
                _, outputs, _feature = self.forward_one_batch(X, targets, False)

                if _ == -1:
                    return

                smax = to_np(F.softmax(outputs / T, dim=1))
                # energy
                OOD_logits.append(-to_np((T * torch.logsumexp(outputs / T, dim=1))))
                # entropy
                OOD_score_entropy.append(entropy(smax, axis=1))
                # var
                OOD_score_var.append(-np.var(smax, axis=1))
                # msp
                OOD_score_msp.append(-np.max(smax, axis=1))
                # max_logits
                max_logit = to_np(outputs/T)
                OOD_max_logits.append(-np.max(max_logit, axis=1))

            # total_testimages x num_classes
            OOD_logits = concat(OOD_logits)[:len(ood_loader.dataset)].copy()
            OOD_score_entropy = concat(OOD_score_entropy)[:len(ood_loader.dataset)].copy()
            OOD_score_var = concat(OOD_score_var)[:len(ood_loader.dataset)].copy()
            OOD_score_msp = concat(OOD_score_msp)[:len(ood_loader.dataset)].copy()
            OOD_max_logits = concat(OOD_max_logits)[:len(ood_loader.dataset)].copy()
            InD_knn_feature, OOD_knn_feature = kNN_OOD(root=self.cfg.OUTPUT_DIR, epoch=epoch, name=name)

            # Print statistic information
            # logger.info(f"InD_logits: {stats.describe(InD_logits)}")
            # logger.info(f"OOD_logits: {stats.describe(OOD_logits)}")

            # # Visualize Distribution
            plot_distribution_1(InD_logits, OOD_logits, epoch, f'{name}_energy', output_path = self.cfg.OUTPUT_DIR)
            plot_distribution_1(InD_score_entropy, OOD_score_entropy, epoch, f'{name}_entropy', output_path = self.cfg.OUTPUT_DIR)
            plot_distribution_1(InD_score_var, OOD_score_var, epoch, f'{name}_var', output_path = self.cfg.OUTPUT_DIR)
            plot_distribution_1(InD_score_msp, OOD_score_msp, epoch, f'{name}_msp', output_path = self.cfg.OUTPUT_DIR)
            plot_distribution_1(InD_max_logits, OOD_max_logits, epoch, f'{name}_max_logits', output_path = self.cfg.OUTPUT_DIR)
            plot_distribution_1(InD_knn_feature, OOD_knn_feature, epoch, f'{name}_kNN_feature', output_path=self.cfg.OUTPUT_DIR)

            # Final results: FPR@95, AUROC, AUPR
            logger.info(f"========================OOD DATASET {name}========================")
            logger.info(f"========================OOD_logits_energy========================")
            get_and_print_results(logger, InD_logits, OOD_logits, auroc_list, aupr_list, fpr_list)
            logger.info(f"========================OOD_score_entropy========================")
            get_and_print_results(logger, InD_score_entropy, OOD_score_entropy, auroc_list, aupr_list, fpr_list)
            logger.info(f"========================OOD_score_var========================")
            get_and_print_results(logger, InD_score_var, OOD_score_var, auroc_list, aupr_list, fpr_list)
            logger.info(f"========================OOD_score_msp========================")
            get_and_print_results(logger, InD_score_msp, OOD_score_msp, auroc_list, aupr_list, fpr_list)
            logger.info(f"========================OOD_max_logits========================")
            get_and_print_results(logger, InD_max_logits, OOD_max_logits, auroc_list, aupr_list, fpr_list)
            logger.info(f"==========================OOD_kNN_feature=========================")
            get_and_print_results(logger, InD_knn_feature, OOD_knn_feature, auroc_list, aupr_list, fpr_list)

