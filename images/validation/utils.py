from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

import argparse

import copy
import time
from tqdm.auto import tqdm
import yaml


class Trainer:

    def __init__(self, dataset, save_dir, batch_size=4096, lr=1e-3, n_epochs=20, seed=42, use_y_as_label=False,
                 image_shape=[3, 64, 64]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.save_dir = save_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dir = Path(self.save_dir)
        self.use_y_as_label = use_y_as_label

    def train_model(self):

        print('Training ' + ('label' if self.use_y_as_label else 'domain') + ' classifier.')

        # prepare the data
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(self.dataset, [train_size, val_size],
                                                           generator=torch.Generator().manual_seed(self.seed))
        # set a data augmentation transform for train set
        train_transform = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            lambda x: x,  # NoOp
            lambda x: x,  # NoOp
            lambda x: x,  # NoOp
        ])

        # apply the transform to the train set
        class TransformedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                item = list(self.subset[index])
                if self.transform:
                    item[0] = self.transform(item[0])
                return tuple(item)

            def __len__(self):
                return len(self.subset)

        train_set = TransformedSubset(train_set, train_transform)

        # getting dataset statistics
        try:
            self.n_domains = len(np.unique(self.dataset.train_domain))
            self.n_labels = len(np.unique(self.dataset.train_labels))
        except AttributeError:
            self.n_domains = len(np.unique(self.dataset.domain))
            self.n_labels = len(np.unique(self.dataset.label))
        if self.use_y_as_label:
            self.n_classes = self.n_labels
        else:
            self.n_classes = self.n_domains

        print(f'Found {self.n_classes} {"domains" if not self.use_y_as_label else "y classes"}')
        dataloaders = {
            'train': torch.utils.data.DataLoader(train_set, self.batch_size, shuffle=True, num_workers=16),
            'val': torch.utils.data.DataLoader(val_set, self.batch_size, shuffle=False, num_workers=16)
        }

        # initialize model
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, self.n_classes)
        # checking if images are grayscale, and if so, changing the first layer to accept 1 channel instead of 3
        if len(self.dataset[0][0].squeeze().shape) == 2:
            print('Images are grayscale, changing ResNet18 first layer to accept 1 channel instead of 3.')
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = nn.DataParallel(model).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model, train_history, val_history = self._train(model, dataloaders,
                                                        criterion, optimizer,
                                                        use_y_as_label=self.use_y_as_label,
                                                        model_dir=self.model_dir, device=self.device,
                                                        num_epochs=self.n_epochs)

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(train_history)
        axes[0].set_title('Train History')
        axes[1].plot(val_history)
        axes[1].set_title('Val History')
        plt.tight_layout()
        plt.savefig(str(Path(self.model_dir) / 'training_history_plots.png'))
        print('Fin training!')

    @staticmethod
    def _train(model, dataloaders, criterion,
               optimizer, model_dir, use_y_as_label=False,
               device=None, num_epochs=20, verbose=True, log=True):
        """A general(ish) training function which does training + a per-epoch validation for a given model,
        dataloader dict (a dict containing training and val dataloaders), criterion (loss function), and
        an already initialized optimizer (e.g. Adam which has model.learnable_parameters() already registered).
        Note: `model` can be parallelized by wrapping it in nn.DataParallel before passing it to this function.
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        if log:
            with open(model_dir / 'training_log.txt', 'w') as f:
                # overwritting the current log file
                from datetime import date
                f.write(f'Training log for : {str(date.today())}\n')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        start_time = time.time()

        val_acc_history = []
        train_acc_history = []

        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch} / {num_epochs}')
            print('-' * 10)

            # we're combining train+val into one call
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # set model to training mode
                    # wrapping the training epoch with tqdm to get a progress bar
                    data_iterator = tqdm(dataloaders['train'])
                    print_on_batch = int(np.ceil(len(data_iterator) / 10))

                else:
                    model.eval()  # set model to evaluate mode
                    # validation should be quite fast, so no need for tqdm wrapper
                    data_iterator = dataloaders['val']

                running_loss = 0.0
                running_corrects = 0.0

                for batch_idx, (inputs, y_labels, domain_labels) in enumerate(data_iterator):
                    inputs = inputs.to(device)
                    if use_y_as_label:
                        classification_labels = y_labels.to(
                            device)  # a hack to train a label classifier instead of domain classifier
                    else:
                        classification_labels = domain_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # enable gradient tracking only if we are training
                    with torch.set_grad_enabled(phase == 'train'):
                        # get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, classification_labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == classification_labels.data)

                    # if verbose: printing training batch progress
                    if phase == 'train' and (batch_idx + 1) % print_on_batch == 0 and verbose == True:
                        n_samples_so_far = (batch_idx + 1) * dataloaders['train'].batch_size
                        # print(f'\tBatch {batch_idx}/{len(data_iterator)}: ',
                        #       f'running loss={running_loss / n_samples_so_far:.3f} ',
                        #       f'acc={100*running_corrects / n_samples_so_far:2f}%')

                # end epoch iteration
                epoch_loss = running_loss / len(dataloaders[phase].sampler)
                epoch_acc = 100 * running_corrects / len(dataloaders[phase].sampler)

                print(f'Total {phase}\tLoss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}%')

                if log:
                    with open(model_dir / 'training_log.txt', 'a') as f:
                        f.write(f'Epoch {epoch}, {phase}\tLoss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}%\n')
                        if phase == 'val' and epoch_acc > best_val_acc:
                            f.write(f'--- New best validation! Model saved to {str(model_dir / "model.pt")}. ---\n')

                if phase == 'val' and epoch_acc > best_val_acc:
                    # if best validation phase so far, deep copy the model
                    best_val_acc = epoch_acc
                    torch.save(model.module.state_dict(), model_dir / 'model.pt')
                    print(f'New best validation! Model saved.')
                if phase == 'val':
                    val_acc_history.append(epoch_acc.cpu())
                else:
                    train_acc_history.append(epoch_acc.cpu())

            print()

        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best model saved to {str(model_dir / "model.pt")}')
        print(f'\tBest model val Acc: {best_val_acc:.4f}')

        if log:
            with open(model_dir / 'training_log.txt', 'a') as f:
                f.write('\n\n\n Finished training!')
                f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
                f.write(f'Best model saved to {str(model_dir / "model.pt")}\n')
                f.write(f'\tBest model val Acc: {best_val_acc:.4f}\n')

        # load best model weights
        model.module.load_state_dict(torch.load(model_dir / 'model.pt'))
        return model, train_acc_history, val_acc_history


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
        f, g = self._init_models()

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