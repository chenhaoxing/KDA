import sys
import torch
from torch import optim
from torch.utils import data
from ptflops import get_model_complexity_info
from src.args import args_main
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.loss import AVGZSLLoss, L2Loss, SquaredL2Loss, ClsContrastiveLoss, KDA_Loss, CJMELoss
from src.metrics import DetailedLosses, MeanClassAccuracy, PercentOverlappingClasses, TargetDifficulty
from src.model import AVGZSLNet, DeviseModel, KDA, CJME
from src.sampler import SamplerFactory
from src.model_improvements import AVCA
from src.utils_improvements import get_model_params
from src.train import train
from src.utils import fix_seeds, setup_experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.nn import DataParallel
import os
from torch import nn


def main():
    args = args_main()
    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size
    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")

    if args.dataset_name == "AudioSetZSL":
        val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode="seen",
        )

        train_val_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode="seen",
        )

        val_all_dataset = AudioSetZSLDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode="all",
        )

    elif args.dataset_name == "VGGSound":
        val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )

        train_val_dataset = VGGSoundDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        val_dataset = UCFDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )

        train_val_dataset = UCFDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )

        train_val_dataset = ActivityNetDataset(
            args=args,
            dataset_split="train_val",
            zero_shot_mode=None,
        )

        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="test",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    
    contrastive_val_dataset = ContrastiveDataset(val_dataset)
    contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=1,
        alpha=1,
        kind='random'
    )

    train_val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=1,
        alpha=1,
        kind='random'
    )

    val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        batch_sampler=val_sampler,
        num_workers=64,
        pin_memory=True
    )
    
    train_val_loader = data.DataLoader(
        dataset=contrastive_train_val_dataset,
        batch_sampler=train_val_sampler,
        num_workers=64,
        pin_memory=True
    )
    

    val_all_loader = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        num_workers=64,
        pin_memory=True
    )

    if args.AVCA==True:
        model_params = get_model_params(args.lr, args.first_additional_triplet, args.second_additional_triplet, \
                                        args.reg_loss, args.additional_triplets_loss, args.embedding_dropout, \
                                        args.decoder_dropout, args.additional_dropout, args.embeddings_hidden_size, \
                                        args.decoder_hidden_size, args.depth_transformer, args.momentum)


    if args.ale==True or args.devise==True or args.sje==True:
        model= DeviseModel(args)
    elif args.kda==True:
        model=KDA(args)
    elif args.cjme==True:
        model=CJME(args)
    elif args.AVCA==True:
        model = AVCA(model_params, input_size_audio=args.input_size_audio, input_size_video=args.input_size_video)
    else:
        model = AVGZSLNet(args)

    model = model.to(args.device)
    
    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()
    if args.ale==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="weighted")
    elif args.devise==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="sum")
    elif args.sje==True:
        criterion = ClsContrastiveLoss(margin=0.1, max_violation=True, topk=1, reduction="sum")
    elif args.kda==True:
        criterion=KDA_Loss(lamda=5)
    elif args.cjme==True:
        criterion=CJMELoss(margin=args.margin, distance_fn=distance_fn)
    elif args.AVCA==True:
        criterion=None
    else:
        criterion = AVGZSLLoss(margin=args.margin, distance_fn=distance_fn)

    optimizer = optim.Adam(model.parameters(), betas=(0.5,0.999), eps=1e-08, lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=4, factor=0.1, verbose=True, min_lr=1e-08) if args.lr_scheduler else None

    metrics = [
        MeanClassAccuracy(model=model, dataset=val_all_dataset, device=args.device, distance_fn=distance_fn,
                          model_devise=args.ale or args.sje or args.devise,
                          new_model_attention=args.AVCA,
                          kda=args.kda,
                          args=args)
    ]

    logger.info(model)
    logger.info(criterion)
    logger.info(optimizer)
    logger.info(lr_scheduler)
    logger.info([metric.__class__.__name__ for metric in metrics])
    v_loader = val_loader

    best_loss, best_score = train(
        train_loader=train_val_loader if args.retrain_all else train_loader,
        val_loader=v_loader,
        distance_fn = distance_fn,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        new_model_attention=args.AVCA,
        val_stats=val_stats,
        log_dir=log_dir,
        model_devise=args.ale or args.sje or args.devise,
        kda=args.kda,
        cjme=args.cjme,
        args=args
    )

    logger.info(f"FINISHED. Run is stored at {log_dir}")


if __name__ == '__main__':
    main()
