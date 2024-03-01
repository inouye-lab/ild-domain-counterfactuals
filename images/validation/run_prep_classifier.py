import pickle
import pandas as pd
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import yaml
import einops
import argparse
import sys
sys.path.append('../')
from dataset import MnistRotated, MnistColorRotated
from utils import Trainer

MODEL_PRESET_TO_NAME = {
    1: 'relax_can',
    2: 'can',
    3: 'dense',
    4: 'independent',
}
MODEL_NAME_TO_FIGURE_NAME = {
    'can': 'ILD-Can',
    'relax_can': 'ILD-Relax-Can',
    'dense': 'ILD-Dense',
    'indp': 'ILD-Independent',
}


def get_classifier(seed_list,
                   classifier_save_dir,
                   dataset):
    """Get the counterfactual metrics for each ILD type.
    exp_dir should be the experiment directory (e.g. 'causal3dident'). This is used to load the correct models.py file.
    ild_types should be a list of strings.
    ild_model_dirs should be a list of model directories for each ild_type.
    ild_checkpoints should be a list of model checkpoint idxs for each ild_type.
    """
    print(f'Getting all counterfactual metrics and saving to {classifier_save_dir}...\n\n')
    Path(classifier_save_dir).mkdir(parents=True, exist_ok=True)

    for seed in seed_list:
        domain_classifier_dir = Path(classifier_save_dir) / 'domain_classifier' / f'seed_{seed}'
        class_classifier_dir = Path(classifier_save_dir) / 'class_classifier' / f'seed_{seed}'
        if not Path(domain_classifier_dir).exists():
            # get seed from config of ild model
            Trainer(dataset, save_dir=domain_classifier_dir, batch_size=4096, seed=seed).train_model()
        if not Path(class_classifier_dir).exists():
            # get seed from config of ild model
            Trainer(dataset, save_dir=class_classifier_dir, batch_size=4096, seed=seed,
                    use_y_as_label=True).train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rmnist')
    args = parser.parse_args()

    seed_list = list(range(21))
    classifier_save_dir = Path(f'classifier/{args.dataset}')

    # load dataset
    if args.dataset in ['rmnist', 'rfmnist', 'rscmnist']:
        dataset = MnistRotated(['0', '15', '30', '45', '60'], '../data', mnist_type=args.dataset ,train=True)
    elif args.dataset in ['crmnist']:
        dataset = MnistColorRotated(['0','90'], '../data', train=True)
    else:
        raise NotImplementedError(f'Experiment dir: {args.dataset} not implemented.')

    get_classifier(seed_list, classifier_save_dir, dataset)
