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
from dataset import MnistRotated, MnistColorRotated

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


class CounterfactualDataset(torch.utils.data.Dataset):
    def __init__(self, counterfactual_images, original_domain_labels,
                 counterfactual_domain_labels, class_labels, counterfactual_model_info=None):
        self.counterfactual_images = counterfactual_images
        self.original_domain_labels = original_domain_labels
        self.counterfactual_domain_labels = counterfactual_domain_labels
        self.class_labels = class_labels
        self.counterfactual_model_info = counterfactual_model_info

    def __getitem__(self, index):
        return self.counterfactual_images[index], self.counterfactual_domain_labels[index], \
            self.original_domain_labels[index], self.class_labels[index]

    def __len__(self):
        return len(self.counterfactual_images)


class CounterFactualMetrics:

    def __init__(self,
                 dataset, causal_model_dir, causal_model_checkpoint_idx,
                 domain_classifier_dir, class_classifier_dir) -> None:
        self.model_dir = Path(causal_model_dir)
        self.model_checkpoint_idx = causal_model_checkpoint_idx
        # load model config
        config = vars(yaml.load((self.model_dir / 'config.yml').open('r'), Loader=yaml.UnsafeLoader))
        config['model_dir'] = self.model_dir
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = self._dict_to_args(config)
        self.f, self.g = self._load_f_and_g()
        # setup dataset args
        self.dataset = dataset
        try:
            self.n_domains = len(np.unique(self.dataset.train_domain))
            self.n_labels = len(np.unique(self.dataset.train_labels))
        except AttributeError:
            self.n_domains = len(np.unique(self.dataset.domain))
            self.n_labels = len(np.unique(self.dataset.label))
        # load classifiers
        self.domain_classifier = self._load_classifier(domain_classifier_dir, self.n_domains)
        self.class_classifier = self._load_classifier(class_classifier_dir, self.n_labels)

    def calculate_metrics(self, m=1):
        # normal counterfactual dataset
        counterfactual_dataset = self.make_counterfactual_dataset()
        cf_d_acc, cf_y_acc = self.classify_counterfactual_dataset(counterfactual_dataset)
        # cycle counterfactual dataset
        counterfactual_dataset = self.make_m_reversibility_dataset(m)
        reversibility_cf_d_acc, reversibility_cf_y_acc = self.classify_counterfactual_dataset(counterfactual_dataset)
        # null transformation counterfactual dataset
        counterfactual_dataset = self.make_m_composition_dataset(m)
        composition_cf_d_acc, composition_cf_y_acc = self.classify_counterfactual_dataset(counterfactual_dataset)

        return {'effectiveness': cf_d_acc, 'preservation': cf_y_acc,
                'reversibility_cf_d_acc': reversibility_cf_d_acc, 'reversibility': reversibility_cf_y_acc,
                'composition_cf_d_acc': composition_cf_d_acc, 'composition': composition_cf_y_acc}

    @torch.no_grad()
    def classify_counterfactual_dataset(self, counterfactual_dataset):
        counterfactual_loader = self._create_dataloader(counterfactual_dataset, batch_size=4096)
        with torch.no_grad():
            n_domain_correct = 0
            n_label_correct = 0
            n_total = 0
            for (x, d, og_d, y) in counterfactual_loader:
                x, y, d = x.to(self.args.device), y.to(self.args.device), d.to(self.args.device)
                domain_logits = self.domain_classifier(x)
                _, domain_preds = torch.max(domain_logits, 1)
                label_logits = self.class_classifier(x)
                _, label_preds = torch.max(label_logits, 1)

                n_domain_correct += torch.sum(domain_preds == d).item()
                n_label_correct += torch.sum(label_preds == y).item()
                n_total += len(label_preds)
        return (n_domain_correct / n_total) * 100, (n_label_correct / n_total) * 100

    @torch.no_grad()
    def make_counterfactual_dataset(self):
        rng = np.random.RandomState(self.args.seed)
        dataset_loader = self._create_dataloader(self.dataset)

        counterfactual_images = []
        original_domain_labels = []
        counterfactual_domain_labels = []
        class_labels = []
        for batch_idx, (x, y, d) in enumerate(dataset_loader):
            original_domain_labels.extend(d.tolist())
            class_labels.extend(y.tolist())
            new_domain_labels = rng.choice(self.n_domains, size=len(d))
            counterfactual_domain_labels.extend(new_domain_labels.tolist())

            eps_back = self.f.z_to_eps(self.g.x_to_z(x.to(self.args.device), d.to(self.args.device)),
                                       d.to(self.args.device), set_epsilon_to_mean=True)
            counter = self.g.z_to_x(self.f.eps_to_z(eps_back, torch.tensor(new_domain_labels).to(self.args.device)),
                                    torch.tensor(new_domain_labels).to(self.args.device))
            counterfactual_images.extend(list(counter.detach().cpu()))

        return CounterfactualDataset(counterfactual_images, original_domain_labels, counterfactual_domain_labels,
                                     class_labels, counterfactual_model_info={'model_dir': str(self.args.model_dir),
                                                                              'model_config': self.args.config})

    @torch.no_grad()
    def make_m_composition_dataset(self, m):
        """Create a dataset of m **null transformation** counterfactual images for each image in the dataset_loader.
        A null transformation is when d = d' (i.e. the domain label is unchanged)."""
        rng = np.random.RandomState(self.args.seed)
        dataset_loader = self._create_dataloader(self.dataset)

        counterfactual_images = []
        original_domain_labels = []
        counterfactual_domain_labels = []
        class_labels = []
        for batch_idx, (x, y, d) in enumerate(dataset_loader):
            original_domain_labels.extend(d.tolist())
            class_labels.extend(y.tolist())
            counterfactual_domain_labels.extend(d.tolist())

            x, d, new_domain_labels = x.to(self.args.device), d.to(self.args.device), d.to(self.args.device)
            for i in range(m):
                eps_back = self.f.z_to_eps(self.g.x_to_z(x, d), d, set_epsilon_to_mean=True)
                x = self.g.z_to_x(self.f.eps_to_z(eps_back, d), d)
            counterfactual_images.extend(list(x.detach().cpu()))

        return CounterfactualDataset(counterfactual_images, original_domain_labels, counterfactual_domain_labels,
                                     class_labels, counterfactual_model_info={'model_dir': str(self.args.model_dir),
                                                                              'model_config': self.args.config})

    @torch.no_grad()
    def make_m_reversibility_dataset(self, m):
        """Create a dataset of m **cyclic** counterfactual images for each image in the dataset_loader.
        A cyclic counterfactual transformation is: Counter( Counter(x, d->d'), d'->d ) (i.e. the domain label is changed
        and then changed back)."""
        rng = np.random.RandomState(self.args.seed)
        dataset_loader = self._create_dataloader(self.dataset)

        counterfactual_images = []
        original_domain_labels = []
        counterfactual_domain_labels = []
        class_labels = []
        for batch_idx, (x, y, d) in enumerate(dataset_loader):
            original_domain_labels.extend(d.tolist())
            class_labels.extend(y.tolist())
            new_domain_labels = rng.choice(self.n_domains, size=len(d))
            counterfactual_domain_labels.extend(d.tolist())

            x, d, new_domain_labels = x.to(self.args.device), d.to(self.args.device), torch.tensor(
                new_domain_labels).to(self.args.device)
            for i in range(m):
                # go to new domain
                eps_back = self.f.z_to_eps(self.g.x_to_z(x, d), d, set_epsilon_to_mean=True)
                counter = self.g.z_to_x(self.f.eps_to_z(eps_back, new_domain_labels), new_domain_labels)
                # go back to original domain
                eps_back = self.f.z_to_eps(self.g.x_to_z(x, new_domain_labels), new_domain_labels,
                                           set_epsilon_to_mean=True)
                counter = self.g.z_to_x(self.f.eps_to_z(eps_back, d), d)
            counterfactual_images.extend(list(counter.detach().cpu()))

        return CounterfactualDataset(counterfactual_images, original_domain_labels, counterfactual_domain_labels,
                                     class_labels, counterfactual_model_info={'model_dir': str(self.args.model_dir),
                                                                              'model_config': self.args.config})

    def _dict_to_args(self, config):
        """Takes a config dict and returns an args object with . access to all config keys"""

        class Args:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        args = Args(**config)
        args.config = config
        return args

    def _create_dataloader(self, dataset, batch_size=65, shuffle=False, num_workers=16):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _load_f_and_g(self):
        f, g= self._init_models()

        f.load_state_dict(torch.load(self.model_dir / f'F_{self.model_checkpoint_idx}.pt'))
        g.load_state_dict(torch.load(self.model_dir / f'G_{self.model_checkpoint_idx}.pt'))

        print('Loaded model, f_type: ', self.args.f_type, ' g_type: ', self.args.g_type,
              'at idx: ', self.model_checkpoint_idx, ' from ', self.model_dir)

        f, g = f.to(self.args.device), g.to(self.args.device)
        return f, g

    def _load_classifier(self, model_dir, n_classes):
        from torchvision.models import resnet18
        classifier = resnet18()
        classifier.fc = torch.nn.Linear(512, n_classes)
        # checking if images are grayscale, and if so, changing the first layer to accept 1 channel instead of 3
        if len(self.dataset[0][0].squeeze().shape) == 2:
            classifier.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        classifier = classifier.to(self.args.device)
        classifier.load_state_dict(torch.load(Path(model_dir) / 'model.pt'))
        classifier.eval()
        return classifier

    def _init_models(self):
        from models import F_VAE_dense, F_VAE_can, F_VAE_relax_can
        from models import GBetaVAE, GIndpendentBetaVAE
        if self.args.f_type == 'dense':
            f = F_VAE_dense(self.args)
        elif self.args.f_type == 'can':
            f = F_VAE_can(self.args)
        elif self.args.f_type == 'relax_can':
            f = F_VAE_relax_can(self.args)

        if self.args.g_type == 'beta':
            g = GBetaVAE(self.args)
        elif self.args.g_type == 'independent':
            g = GIndpendentBetaVAE(self.args)
        return f, g

def get_metrics_for_all_ild_types(ild_types: list, ild_model_dirs: list, ild_checkpoints: list,
                                  domain_classifier_dir, class_classifier_dir,
                                  dataset, save_dir: str,
                                  train_classifiers_if_need_be=True):
    """Get the counterfactual metrics for each ILD type.
    exp_dir should be the experiment directory (e.g. 'causal3dident'). This is used to load the correct models.py file.
    ild_types should be a list of strings.
    ild_model_dirs should be a list of model directories for each ild_type.
    ild_checkpoints should be a list of model checkpoint idxs for each ild_type.
    """
    print(f'Getting all counterfactual metrics and saving to {save_dir}...\n\n')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if train_classifiers_if_need_be:
        # checking if the classifiers need to be trained
        if not Path(domain_classifier_dir).exists():
            # get seed from config of ild model
            seed = vars(yaml.load((Path(ild_model_dirs[0]) / 'config.yml').open('r'), Loader=yaml.UnsafeLoader))['seed']
            from utils import Trainer
            Trainer(dataset, save_dir=domain_classifier_dir, batch_size=4096, seed=seed).train_model()
        if not Path(class_classifier_dir).exists():
            # get seed from config of ild model
            seed = vars(yaml.load((Path(ild_model_dirs[0]) / 'config.yml').open('r'), Loader=yaml.UnsafeLoader))['seed']
            from utils import Trainer
            Trainer(dataset, save_dir=class_classifier_dir, batch_size=4096, seed=seed,
                    use_y_as_label=True).train_model()

    ild_metrics = {'ild_type': [], 'effectiveness': [], 'preservation': [], 'reversibility': [], 'composition': []}
    # gather cf metrics for all ild_types
    for ild_type, model_dir, ild_checkpoint in zip(ild_types, ild_model_dirs, ild_checkpoints):
        print(f'Calculating metrics for {ild_type}...')
        cf_metric_maker = CounterFactualMetrics(dataset,
                                                causal_model_dir=model_dir,
                                                causal_model_checkpoint_idx=ild_checkpoint,
                                                domain_classifier_dir=domain_classifier_dir,
                                                class_classifier_dir=class_classifier_dir)
        metrics = cf_metric_maker.calculate_metrics()
        for k in ild_metrics.keys():
            if k == 'ild_type':
                ild_metrics[k].append(ild_type)
            else:
                ild_metrics[k].append(metrics[k])
    # save metrics
    pd.DataFrame(ild_metrics).to_csv(Path(save_dir) / 'counterfactual_metrics.csv')
    return ild_metrics


def plot_counterfactual_metrics(ild_types: list, metrics: dict, save_dir: str):
    """Plot the counterfactual metrics for each ILD type.
    Metrics should be a dict with keys:
    'effectiveness', 'preservation', 'reversibility', 'composition' and values as a list of floats."""
    # Define the x-axis positions for the bar chart
    x = np.arange(len(ild_types))

    # Define the width of the bars
    width = 0.2

    # Create the figure and subplot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create the bar chart for each group
    # plotting the effectiveness, composition, and reversibility for each ILD type
    ax.bar(x - width, metrics['composition'], width, label='Composition')
    ax.bar(x, metrics['reversibility'], width, label='Reversibility')
    ax.bar(x + width, metrics['preservation'], width, label='Preservation')
    ax.bar(x + 2 * width, metrics['effectiveness'], width, label='Effectiveness')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_NAME_TO_FIGURE_NAME[ild_type] for ild_type in ild_types], fontsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_ylabel('Classifier Accuracy (%)', fontsize=25)
    ax.legend(fontsize=25, loc='upper right', bbox_to_anchor=(1.4, 1.0))

    # Adjust subplot parameters to give specified padding.
    plt.tight_layout()

    # Save the plot
    plt.savefig(Path(save_dir) / 'counterfactual_metrics_bar_plot.png')
    print('Saved plot to ', str(Path(save_dir) / 'counterfactual_metrics_bar_plot.png'))
    return ax


def get_last_checkpoint_idx(model_dir):
    """Get the last checkpoint idx for the model in model_dir."""
    # return max([int(p.name.split('.pt')[0].split('_')[1]) for p in Path(model_dir).glob('*.pt')])
    raise NotImplementedError


def build_model_info_df(root_model_dir, checkpoint_dir=None):

    def get_model_info_from_model_dir(model_dir, checkpoint_dir=None):
        model_dir = Path(model_dir)
        ICLR_NAME_TO_SHORT_NAME = {
            'independent': 'indp',
            'relax_can': 'relax_can',
            'dense': 'dense',
            'can': 'can',
        }
        model_dict = {
            'model_dir': str(model_dir),
            'ild_type': ICLR_NAME_TO_SHORT_NAME[model_dir.name.split(']')[0][1:]],
            'k_spa': int(model_dir.name.split('_k')[1].split('_')[0]),
            'seed': int(model_dir.name.split('_seed')[1].split('_')[0]),
        }
        # now add model checkpoint to dict
        if checkpoint_dir is not None:
            # load checkpoint
            with open(checkpoint_dir / f'k{model_dict["k_spa"]}.pkl', 'rb') as f:
                checkpoint = pickle.load(f)[model_dict["ild_type"]][int(model_dict["seed"])]
        else:
            checkpoint = 'best'
            #checkpoint = get_last_checkpoint_idx(model_dir)
        model_dict['checkpoint_idx'] = checkpoint
        return model_dict

    all_models_df = pd.DataFrame(
        [get_model_info_from_model_dir(str(p), checkpoint_dir=checkpoint_dir)
         for p in root_model_dir.glob('*')]
    )
    print('Found Models: ', all_models_df)
    return all_models_df


def get_metrics_for_all_models(root_model_dir, root_classifier_dir, root_save_dir, dataset,
                               checkpoint_dir=None):
    # load models to test
    all_models_df = build_model_info_df(root_model_dir, checkpoint_dir)
    # run for all seeds and k_spas
    for k_spa in all_models_df['k_spa'].unique():
        for seed in all_models_df['seed'].unique():
            print(f'Starting k_spa: {k_spa}, seed: {seed}')
            save_dir = Path(root_save_dir) / f'k_spa_{k_spa}_seed_{seed}'
            matching_exps_df = all_models_df[(all_models_df['k_spa'] == k_spa) & (all_models_df['seed'] == seed)]
            assert len(matching_exps_df) > 0, f'No matching experiments found for k_spa: {k_spa}, seed: {seed}'
            ild_metrics = get_metrics_for_all_ild_types(ild_types=matching_exps_df['ild_type'].tolist(),
                                                        ild_model_dirs=matching_exps_df['model_dir'].tolist(),
                                                        ild_checkpoints=matching_exps_df['checkpoint_idx'].tolist(),
                                                        domain_classifier_dir=str(
                                                            root_classifier_dir / 'domain_classifier' / f'seed_{seed}'),
                                                        class_classifier_dir=str(
                                                            root_classifier_dir / 'class_classifier' / f'seed_{seed}'),
                                                        dataset=dataset,
                                                        save_dir=save_dir)
            plot_counterfactual_metrics(ild_metrics['ild_type'], ild_metrics, save_dir)

    # now that all the results are aggregated, built a plot of the results accross multiple seeds
    build_metric_box_plot_across_seeds(root_save_dir)
    print('\n\n\nDone! All results saved to: ', root_save_dir)
    return None


def build_metric_box_plot_across_seeds(results_dir, plot_save_name=None):
    """Build a box plot of the metrics across multiple seeds."""
    # get all the saved metrics
    results_dir = Path(results_dir)
    all_metrics_df = pd.DataFrame(columns=['ild_type', 'effectiveness', 'preservation', 'reversibility',
                                           'composition', 'k_spa', 'seed'])
    for result_dir in [p for p in results_dir.glob('*') if p.is_dir()]:
        print(result_dir)
        k_spa = int(result_dir.name.split('k_spa_')[1].split('_')[0])
        seed = int(result_dir.name.split('seed_')[1][0])
        metrics = pd.read_csv(result_dir / 'counterfactual_metrics.csv', index_col=0)
        metrics['k_spa'] = k_spa
        metrics['seed'] = seed
        metrics['ild_type'] = metrics['ild_type'].map(MODEL_NAME_TO_FIGURE_NAME)
        all_metrics_df = pd.concat([all_metrics_df, metrics], axis=0).reset_index(drop=True)
    all_metrics_df.rename(columns={'ild_type': 'Model Type', 'k_spa': 'K sparsity',
                                   'effectiveness': 'Effectiveness', 'composition': 'Composition',
                                   'preservation': 'Preservation', 'reversibility': 'Reversibility', }, inplace=True)
    # make box_plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    i = 0
    for metric in ['Composition', 'Reversibility', 'Preservation', 'Effectiveness']:
        sns.boxplot(x='K sparsity', y=metric, hue='Model Type', data=all_metrics_df,
                    ax=axes[i // 2, i % 2], showfliers=False)
        i += 1
    plt.tight_layout()
    sns.set_context('paper', font_scale=2)
    for ax in axes.flatten()[:-1]: ax.legend([], [], frameon=False)
    if plot_save_name is not None:
        plt.savefig(results_dir / plot_save_name)
    else:
        plt.savefig(results_dir / 'box_plot_across_seeds.png')
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rmnist')
    parser.add_argument('--model_dir', type=str, default='saved/ib')
    args = parser.parse_args()


    # RMNIST
    root_save_dir = Path(f'result/{args.dataset}')
    root_model_dir = Path(args.model_dir) / f'{args.dataset}'
    root_classifier_dir = Path(f'classifier/{args.dataset}')
    # the seed specific {domain,class} classifier will saved to root_classifier_dir/{domain,class}_classifier

    # load dataset
    if args.dataset in ['rmnist', 'rfmnist', 'rscmnist']:
        dataset = MnistRotated(['0', '15', '30', '45', '60'], 'data', mnist_type=args.dataset ,train=True)
    elif args.dataset in ['crmnist']:
        dataset = MnistColorRotated(['0','90'], 'data', train=True)
    else:
        raise NotImplementedError(f'Experiment dir: {args.dataset} not implemented.')

    get_metrics_for_all_models(root_model_dir, root_classifier_dir, root_save_dir, dataset)
