import argparse
import logging
from pathlib import Path
import torch.optim as optim
import torch.utils.data
import yaml
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.maskhead import MaskHead
from utils.data_loader import CustomImageDataset
from utils.loss import compute_loss_with_masks

logger = logging.getLogger(__name__)

FITNESS_KEYS = [
    "best_fitness_p",
    "best_fitness_r",
    "best_fitness_ap50",
    "best_fitness_ap",
    "best_fitness_f"
]


def initialize_model(device, weights_file=None):
    """Initializes and returns a model.
    
    If a weights_file is provided, loads the weights from the file and also returns the config with the model.
    """
    if weights_file is not None:
        saved_config = torch.load(weights_file, map_location=device)
        model = MaskHead().to(device)
        state_dict = {k: v for k, v in saved_config['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights_file))  # report
        del state_dict
        return model, saved_config
    else:
        model = MaskHead().to(device)
        return model


def initialize_optimizer(
    hyperparameters,
    options,
    model,
    results_file,
    weights_file,
    is_pretrained=False,
    torch_config=None
):
    batch_size = options.batch_size
    nominal_batch_size = 64  # nominal batch size
    accumulate = max(round(nominal_batch_size / batch_size), 1)  # accumulate loss before optimizing
    hyperparameters['weight_decay'] *= batch_size * accumulate / nominal_batch_size  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.mask_head.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k or 'm.weight' in k or 'w.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if options.adam:
        optimizer = optim.Adam(pg0, lr=hyperparameters['lr0'], betas=(hyperparameters['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyperparameters['lr0'], momentum=hyperparameters['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyperparameters['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyperparameters['lrf']) + hyperparameters['lrf']  # cosine

    # Resume
    start_epoch = 0
    fitness_dict = {i: 0.0 for i in FITNESS_KEYS}
    if is_pretrained:
        # Optimizer
        if torch_config['optimizer'] is not None:
            optimizer.load_state_dict(torch_config['optimizer'])
            fitness_dict = {k: torch_config[k] for k in FITNESS_KEYS}

        # Results
        if torch_config.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(torch_config['training_results'])  # write results.txt

        # Epochs
        start_epoch = torch_config['epoch'] + 1
        if options.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights_file, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights_file, torch_config['epoch'], epochs))
            epochs += torch_config['epoch']  # finetune additional epochs

    return optimizer, start_epoch, accumulate, fitness_dict


def train(hyperparameters, options, device):
    logger.info(f'Hyperparameters: {hyperparameters}')
    logger.info(f'Options: {options}')

    save_dir = Path(options.save_dir)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / 'results.txt'

    epochs = options.epochs
    batch_size = options.batch_size
    weights_file = options.weights

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyperparameters, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(options), f, sort_keys=False)

    use_cuda = device.type != 'cpu'

    # TODO: write data loader
    train_data_loader = CustomImageDataset(
        options.train_ann_file,
        options.train_img_dir,
        batch_size
    )
    num_batches = len(train_data_loader)

    # initialize model
    is_pretrained = weights_file.endswith('.pt')
    if is_pretrained:
        model, torch_config = initialize_model(device, weights_file)
    else:
        model = initialize_model(device)
        torch_config = None

    # Optimizer
    optimizer, start_epoch, accumulate, fitness_dict = initialize_optimizer(
        hyperparameters,
        options,
        model,
        results_file,
        weights_file,
        is_pretrained,
        torch_config
    )

    torch.save(model, weights_dir / 'init.pt')

    for epoch in range(start_epoch, epochs):
        model.train()

        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='training %g' % epoch)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, masks) in pbar:
            # images is [bs, 3, 416, 416]
            # targets is [N, 7] where N = n1+n2+... and ni is the number of objects in i image
            # and the columns are [batch_index, x, y, w, h, class, conf=1]
            # masks is [N, 28, 28] where N = n1+n2+... and ni is the number of objects in i image

            # Forward
            mask_preds, _ = model(imgs, targets)  # [N, 28, 28]

            #TODO: write loss function
            loss = compute_loss_with_masks(mask_preds, masks, targets[:, 5])

            # Backward
            loss.backward()

            # Optimize
            if (i + 1) % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print
            pbar.set_description('Loss %.3g' % loss.item())

        # TODO: Run validation results after each epoch

        # TODO: Plot results after each epoch

        final_epoch = epoch + 1 == epochs
        if final_epoch:
            # TODO: get results (mAP, IoU etc) and update fitness dict
            pass

        # Save model and weights
        torch_config = {
            'epoch': epoch,
            'best_fitness': fitness_dict["best_fitness"],
            'best_fitness_p': fitness_dict["best_fitness_p"],
            'best_fitness_r': fitness_dict["best_fitness_r"],
            'best_fitness_ap50': fitness_dict["best_fitness_ap50"],
            'best_fitness_ap': fitness_dict["best_fitness_ap"],
            'best_fitness_f': fitness_dict["best_fitness_f"],
            'training_results': f.read(),
            'model': model.state_dict(),
            'optimizer': None if final_epoch else optimizer.state_dict(),
        }
        torch.save(torch_config, save_dir / 'epoch_{:03d}.pt'.format(epoch))

    torch.cuda.empty_cache()

# TODO: Write main function
