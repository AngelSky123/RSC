"""
CSI-RSC-PoseDG: 训练脚本 v3 — with anti-collapse losses
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import get_config
from dataset import build_dataloaders
from models import CSIRSCPoseDG
from losses import TotalLoss, PoseLoss
from evaluate import PoseEvaluator
from utils import (
    set_seed, setup_logger, count_parameters,
    save_checkpoint, AverageMeter, Timer
)


def action_to_index(action_str):
    """Convert 'A01'~'A27' to integer 0~26."""
    return int(action_str[1:]) - 1


def train_one_epoch(model, train_loader, optimizer, loss_fn, pose_loss_fn,
                    device, epoch, logger, args):
    model.train()
    meters = {
        'loss': AverageMeter(),
        'l_pose_masked': AverageMeter(),
        'l_pose_clean': AverageMeter(),
        'l_cons': AverageMeter(),
        'l_div': AverageMeter(),
        'l_temp_div': AverageMeter(),
        'l_action': AverageMeter(),
        'l_coord': AverageMeter(),
    }
    accum_steps = getattr(args, 'accumulate_grad', 1)
    action_loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)
        action_labels = torch.tensor(
            [action_to_index(a) for a in batch['action']],
            dtype=torch.long, device=device
        )

        # RSC forward
        outputs = model.forward_rsc(
            csi, pose_3d,
            loss_fn=lambda pred, gt: pose_loss_fn(pred, gt)[0]
        )

        # Action classification loss
        action_loss = action_loss_fn(outputs['action_logits'], action_labels)

        # Total loss with anti-collapse terms
        total_loss, loss_dict = loss_fn(
            outputs, pose_3d, training=True, action_loss=action_loss
        )
        total_loss = total_loss / accum_steps
        total_loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        B = csi.shape[0]
        meters['loss'].update(loss_dict['l_total'], B)
        meters['l_pose_masked'].update(loss_dict.get('l_pose_masked', 0), B)
        meters['l_pose_clean'].update(loss_dict.get('l_pose_clean', 0), B)
        meters['l_cons'].update(loss_dict.get('l_cons', 0), B)
        meters['l_div'].update(loss_dict.get('l_div', 0), B)
        meters['l_temp_div'].update(loss_dict.get('l_temp_div', 0), B)
        meters['l_action'].update(loss_dict.get('l_action', 0), B)
        meters['l_coord'].update(loss_dict.get('l_coord_masked', 0), B)

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {meters["loss"].avg:.4f} '
                f'Pose(M): {meters["l_pose_masked"].avg:.4f} '
                f'Div: {meters["l_div"].avg:.4f} '
                f'TDiv: {meters["l_temp_div"].avg:.4f} '
                f'Act: {meters["l_action"].avg:.4f}'
            )

        del outputs, total_loss, csi, pose_3d
        torch.cuda.empty_cache()

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, test_loader, loss_fn, device, evaluator, logger):
    model.eval()
    all_preds = []
    all_gts = []
    loss_meter = AverageMeter()

    # Track prediction diversity
    all_mean_poses = []

    for batch in test_loader:
        csi = batch['csi'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        outputs = model(csi)
        total_loss, loss_dict = loss_fn(outputs, pose_3d, training=False)
        loss_meter.update(loss_dict['l_total'], csi.shape[0])

        pred = outputs['p_final']
        all_preds.append(pred.cpu())
        all_gts.append(pose_3d.cpu())
        all_mean_poses.append(pred.mean(dim=1).cpu())  # (B, 17, 3)

        del outputs, csi, pose_3d
        torch.cuda.empty_cache()

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_mean_poses = torch.cat(all_mean_poses, dim=0)  # (N, 17, 3)

    metrics = evaluator.evaluate(all_preds, all_gts)

    # Prediction diversity: std of mean poses across samples
    pred_std = all_mean_poses.std(dim=0).mean().item() * 1000
    gt_std = all_gts.mean(dim=1).std(dim=0).mean().item() * 1000

    logger.info(
        f'[Eval] Loss: {loss_meter.avg:.4f} | '
        f'MPJPE: {metrics["MPJPE (mm)"]:.2f}mm | '
        f'PA-MPJPE: {metrics["PA-MPJPE (mm)"]:.2f}mm | '
        f'PCK@50: {metrics["PCK@50 (%)"]:.1f}% | '
        f'PCK@20: {metrics["PCK@20 (%)"]:.1f}% | '
        f'PredStd: {pred_std:.1f}mm (GT: {gt_std:.1f}mm)'
    )

    metrics['pred_std'] = pred_std
    metrics['gt_std'] = gt_std
    return metrics


def main():
    args = get_config()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(
        'CSI-RSC-PoseDG',
        log_file=os.path.join(args.save_dir, 'train.log')
    )

    logger.info(f'Configuration: {vars(args)}')
    logger.info(f'Device: {device}')
    logger.info(f'Anti-collapse: gamma={args.gamma}, delta={args.delta}')

    data_exists = os.path.exists(args.data_root)
    train_loader, test_loader = build_dataloaders(args, synthetic=not data_exists)
    logger.info(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')

    model = CSIRSCPoseDG(args).to(device)
    logger.info(f'Model parameters: {count_parameters(model):,}')

    loss_fn = TotalLoss(
        lambda1=args.lambda1, lambda2=args.lambda2,
        alpha=args.alpha, beta=args.beta,
        gamma=args.gamma, delta=args.delta,
    )
    pose_loss_fn = PoseLoss(lambda1=args.lambda1, lambda2=args.lambda2)
    evaluator = PoseEvaluator(unit='meter')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    timer = Timer()
    timer.start()
    best_mpjpe = float('inf')
    patience_counter = 0
    patience = getattr(args, 'patience', 15)

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, pose_loss_fn,
            device, epoch, logger, args
        )
        logger.info(
            f'[Train] Epoch {epoch} | Loss: {train_metrics["loss"]:.4f} '
            f'Div: {train_metrics["l_div"]:.4f} '
            f'TDiv: {train_metrics["l_temp_div"]:.4f} '
            f'Act: {train_metrics["l_action"]:.4f} | '
            f'Time: {timer.elapsed_str()}'
        )

        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(
                model, test_loader, loss_fn, device, evaluator, logger
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
                logger.info(f'No improvement. Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
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