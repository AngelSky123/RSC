"""
CSI-PoseDG: 标准训练脚本 — 8:2 subject-based split

测试模型整体性能, 不使用RSC, 纯backbone训练.
训练/测试数据来自所有4个环境, 按被试8:2划分.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config_standard import get_config
from dataset_standard import build_dataloaders
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from utils import (
    set_seed, setup_logger, count_parameters,
    save_checkpoint, AverageMeter, Timer
)


def train_one_epoch(model, train_loader, optimizer, pose_loss_fn,
                    device, epoch, logger, args):
    model.train()
    meters = {
        'loss': AverageMeter(),
        'l_coord': AverageMeter(),
        'l_bone': AverageMeter(),
        'l_vel': AverageMeter(),
    }
    accum_steps = getattr(args, 'accumulate_grad', 1)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        # Standard forward (no RSC)
        outputs = model(csi)
        pred = outputs['p_final']

        # Pose loss only
        total_loss, loss_details = pose_loss_fn(pred, pose_3d)
        total_loss = total_loss / accum_steps
        total_loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(total_loss.item() * accum_steps, B)
        meters['l_coord'].update(loss_details['l_coord'], B)
        meters['l_bone'].update(loss_details['l_bone'], B)
        meters['l_vel'].update(loss_details['l_vel'], B)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {meters["loss"].avg:.4f} '
                f'Coord: {meters["l_coord"].avg:.4f} '
                f'Bone: {meters["l_bone"].avg:.4f} '
                f'Vel: {meters["l_vel"].avg:.4f}'
            )

        del outputs, total_loss, csi, pose_3d
        torch.cuda.empty_cache()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, test_loader, pose_loss_fn, device, evaluator, logger):
    model.eval()
    all_preds = []
    all_gts = []
    all_actions = []
    all_envs = []
    loss_meter = AverageMeter()

    for batch in test_loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        outputs = model(csi)
        pred = outputs['p_final']
        total_loss, _ = pose_loss_fn(pred, pose_3d)
        loss_meter.update(total_loss.item(), csi.shape[0])

        all_preds.append(pred.cpu())
        all_gts.append(pose_3d.cpu())
        all_actions.extend(batch['action'])
        all_envs.extend(batch['env'])

        del outputs, csi, pose_3d
        torch.cuda.empty_cache()

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Overall metrics
    metrics = evaluator.evaluate(all_preds, all_gts)

    # Prediction diversity
    pred_std = all_preds.mean(dim=1).std(dim=0).mean().item() * 1000
    gt_std = all_gts.mean(dim=1).std(dim=0).mean().item() * 1000

    logger.info(
        f'[Eval] Loss: {loss_meter.avg:.4f} | '
        f'MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'PA-MPJPE: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'PCK@50: {metrics["PCK@50 (%)"]:.1f}% | '
        f'PCK@20: {metrics["PCK@20 (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm (GT: {gt_std:.1f}mm)'
    )

    # Per-environment breakdown
    env_set = sorted(set(all_envs))
    if len(env_set) > 1:
        for env in env_set:
            mask = [i for i, e in enumerate(all_envs) if e == env]
            if len(mask) == 0:
                continue
            mask_t = torch.tensor(mask)
            env_preds = all_preds[mask_t]
            env_gts = all_gts[mask_t]
            env_metrics = evaluator.evaluate(env_preds, env_gts)
            logger.info(
                f'  [{env}] MPJPE: {env_metrics["MPJPE (mm)"]:.2f}mm | '
                f'PA-MPJPE: {env_metrics["PA-MPJPE (mm)"]:.2f}mm | '
                f'PCK@50: {env_metrics["PCK@50 (%)"]:.1f}% | '
                f'n={len(mask)}'
            )

    metrics['pred_std'] = pred_std
    metrics['gt_std'] = gt_std
    return metrics


def main():
    args = get_config()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(
        'CSI-Standard',
        log_file=os.path.join(args.save_dir, 'train.log')
    )

    logger.info(f'{"="*60}')
    logger.info(f'STANDARD 8:2 SPLIT MODE')
    logger.info(f'{"="*60}')
    logger.info(f'Configuration: {vars(args)}')
    logger.info(f'Device: {device}')

    # Data with subject-based split
    train_loader, test_loader = build_dataloaders(args)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    # Model (same architecture, no RSC during training)
    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    # Loss (pose only, no RSC/diversity losses)
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2)
    evaluator = PoseEvaluator(unit='meter')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    timer = Timer()
    timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, pose_loss_fn,
            device, epoch, logger, args
        )
        logger.info(
            f'[Train] Epoch {epoch} | Loss: {train_metrics["loss"]:.4f} '
            f'Coord: {train_metrics["l_coord"]:.4f} | '
            f'Time: {timer.elapsed_str()}'
        )

        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(
                model, test_loader, pose_loss_fn, device, evaluator, logger
            )
            current_mpjpe = eval_metrics['MPJPE (mm)']
            if current_mpjpe < best_mpjpe:
                best_mpjpe = current_mpjpe
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch, eval_metrics,
                    os.path.join(args.save_dir, 'best_model.pth')
                )
                logger.info(f'*** New best MPJPE: {best_mpjpe:.2f}mm ***')
            else:
                patience_counter += 1
                logger.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

            if patience_counter >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, {},
                os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth')
            )

    logger.info(f'\n{"="*60}')
    logger.info(f'Training complete! Best MPJPE: {best_mpjpe:.2f}mm')
    logger.info(f'Total time: {timer.elapsed_str()}')


if __name__ == '__main__':
    main()