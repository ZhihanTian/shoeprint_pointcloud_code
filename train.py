import time
import csv
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import torch.nn.init as init
from model import *
import torch
import random



class Config:
    data_root = r"/hy-tmp/tzh/downsampled_4096"
    num_classes = 91
    max_points = 4096
    chunk_size = 4096
    batch_size = 4
    num_workers = 4
    lr = 1e-3
    epochs = 500
    weight_decay = 1e-4
    dropout_rate = 0.2
    gpu_ids = [0]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    voxel_size = 128
    projection_size = 256
    fpfh_radius = 0.1


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class ShoeDataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.65, random_seed=24):
        """
        Ensure both training and test sets contain all classes of shoe types.
        Key points:
        1. Global stratified sampling (not per-class)
        2. Enforce all classes appear in both sets
        3. Validate class completeness
        """
        self.class_folders = sorted(
            [d for d in os.listdir(root_dir) if d.isdigit()],
            key=lambda x: int(x)
        )
        self.mode = mode
        self.samples = []

        all_files = []
        all_labels = []

        for class_folder in self.class_folders:
            class_id = int(class_folder) - 1
            class_path = os.path.join(root_dir, class_folder)

            files = [
                f for f in os.listdir(class_path)
                if f.endswith('.xyz') and f.startswith(f"{class_folder}_")
            ]

            if len(files) < 2:
                raise ValueError(
                    f"Class {class_folder} has only {len(files)} samples, at least 2 required for split"
                )

            for f in files:
                full_path = os.path.join(class_path, f)
                all_files.append(full_path)
                all_labels.append(class_id)

        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files,
            all_labels,
            train_size=split_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=all_labels  # key
        )

        self._validate_class_coverage(
            train_labels=train_labels,
            val_labels=val_labels,
            total_classes=len(self.class_folders)
        )

        if mode == 'train':
            self.samples = list(zip(train_files, train_labels))
        else:
            self.samples = list(zip(val_files, val_labels))

    def _validate_class_coverage(self, train_labels, val_labels, total_classes):
        """Verify class coverage in train/val set"""
        train_unique = np.unique(train_labels)
        val_unique = np.unique(val_labels)

        if len(train_unique) != total_classes:
            missing = set(range(total_classes)) - set(train_unique)
            raise RuntimeError(
                f"Train set missing classes: {missing}"
            )

        if len(val_unique) != total_classes:
            missing = set(range(total_classes)) - set(val_unique)
            raise RuntimeError(
                f"Validation set missing classes: {missing}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_xyz(self, path):
        """Load .xyz file and handle exceptions"""
        try:
            points = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if not points:
                raise ValueError(f"Empty file: {path}")
            return np.array(points, dtype=np.float32)
        except Exception as e:
            print(f"Failed to load file {path}: {str(e)}")
            return np.zeros((100, 3), dtype=np.float32)  # Return dummy data to keep pipeline running

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        points = self._load_xyz(path)
        num_points = len(points)

        points = self._normalize_points(points)
        points = self._handle_point_quantity(points)

        return torch.from_numpy(points).float(), label

    def _normalize_points(self, points):
        """Point cloud normalization: zero-centering + unit sphere scaling"""
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_distance = np.max(np.linalg.norm(points, axis=1))
        if max_distance > 0:
            points /= max_distance
        return points

    def _handle_point_quantity(self, points):
        """Ensure consistent number of points"""
        num_points = len(points)

        if num_points > Config.max_points:
            indices = np.random.choice(num_points, Config.max_points, replace=False)
        else:
            original_indices = np.arange(num_points)
            remaining = Config.max_points - num_points
            supplement_indices = np.random.choice(num_points, remaining, replace=True)
            indices = np.concatenate([original_indices, supplement_indices])

        return points[indices]


class ExperimentRunner:
    def __init__(self, config, dataset_type='shoe'):
        self.config = config
        self.dataset_type = dataset_type
        self.template_dict = self._create_template_dict()
        self.best_results = {}
        self.models = {
            'OriginalNet': ('original', Net(config.num_classes)),
        }

    def record_result(self, model_name, top1, top3, time_used, model):
        """Update best result, prioritize Top1 then Top3"""
        params = sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else 'N/A'
        current_result = {
            'top1': top1,
            'top3': top3,
            'time': time_used,
            'params': params
        }

        if model_name not in self.best_results:
            self.best_results[model_name] = current_result
        else:
            best = self.best_results[model_name]
            if (current_result['top1'] > best['top1']) or \
                    (current_result['top1'] == best['top1'] and current_result['top3'] > best['top3']):
                self.best_results[model_name] = current_result

    def _create_template_dict(self):
        """Create template point clouds for each class"""
        template_dict = {}
        base_dataset = ShoeDataset(self.config.data_root, 'train')
        class_ids = list(set([s[1] for s in base_dataset.samples]))
        for class_id in class_ids:
            class_samples = [s for s in base_dataset.samples if s[1] == class_id]
            template_path = random.choice(class_samples)[0]
            points = base_dataset._load_xyz(template_path)
            template_dict[class_id] = points
        return template_dict

    def get_data_loaders(self, data_type):
        base_dataset = {
            'shoe': lambda train: ShoeDataset(self.config.data_root, 'train' if train else 'val'),
        }[self.dataset_type]

        dataset_map = {
            'original': base_dataset,
        }

        train_set = dataset_map[data_type](True)
        val_set = dataset_map[data_type](False)

        return (
            DataLoader(train_set, batch_size=self.config.batch_size,
                       shuffle=True, num_workers=self.config.num_workers, drop_last=True),
            DataLoader(val_set, batch_size=self.config.batch_size,
                       shuffle=False, num_workers=2, drop_last=True)
        )

    def run_all(self):
        for model_name, (data_type, model) in self.models.items():
            print(f"\n=== Running {model_name} on {self.dataset_type} ===")

            if isinstance(model, nn.Module):
                self.run_deep_learning_model(model_name, data_type, model)

        self.save_results()

    def run_deep_learning_model(self, model_name, data_type, model):
        config = self.config
        model = model.to(config.device)
        model.apply(weights_init)

        train_loader, val_loader = self.get_data_loaders(data_type)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=config.epochs * len(train_loader))
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        scaler = GradScaler()

        best_top1 = 0.0
        best_top3 = 0.0
        start_time = time.time()

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            for points, labels in train_loader:
                points = points.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()
                with autocast():
                    chunks = torch.split(points, config.chunk_size, dim=2)
                    chunk_logits = [model(chunk) for chunk in chunks]
                    final_logits = torch.mean(torch.stack(chunk_logits, dim=1), dim=1)
                    loss = criterion(final_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            train_top1_acc, train_top3_acc = self.evaluate_model(model, train_loader)
            test_top1_acc, test_top3_acc = self.evaluate_model(model, val_loader)
            avg_loss = total_loss / len(train_loader)

            if test_top1_acc > best_top1 or (test_top1_acc == best_top1 and test_top3_acc > best_top3):
                best_top3 = test_top3_acc
                best_top1 = test_top1_acc
                torch.save(model.state_dict(), f'best_{model_name}_{self.dataset_type}.pth')
            training_time = time.time() - start_time
            self.record_result(model_name, best_top1, best_top3, training_time, model)
            print(f"{model_name} Epoch {epoch + 1}/{config.epochs} | "
                  f"Loss: {avg_loss:.4f} | Train Top1: {train_top1_acc * 100:.2f}% | "
                  f"Train Top3: {train_top3_acc * 100:.2f}% | Test Top1: {test_top1_acc * 100:.2f}% | Test Top3: {test_top3_acc*100:.2f} | Best Top3: {best_top3 * 100:.2f}%")


    def evaluate_model(self, model, val_loader):
        model.eval()
        top1_correct = 0
        top3_correct = 0
        total = 0

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.config.device)
                labels = labels.to(self.config.device)

                with autocast():
                    chunks = torch.split(points, self.config.chunk_size, dim=2)
                    chunk_logits = [model(chunk) for chunk in chunks]
                    outputs = torch.mean(torch.stack(chunk_logits, dim=1), dim=1)

                _, predicted = torch.max(outputs.data, 1)
                top1_correct += (predicted == labels).sum().item()

                _, top3_pred = torch.topk(outputs, k=3, dim=1)
                top3_correct += (top3_pred == labels.view(-1, 1)).sum().item()
                total += labels.size(0)

        return top1_correct / total, top3_correct / total

    def save_results(self):
        """Append best results to CSV file"""
        os.makedirs('results', exist_ok=True)

        results = []
        for model_name, data in self.best_results.items():
            params_str = f"{data['params'] / 1e6:.2f}M" if isinstance(data['params'], int) else 'N/A'
            results.append({
                'Model': model_name,
                'Dataset': self.dataset_type,
                'Top1 Accuracy': f"{data['top1'] * 100:.2f}%",
                'Top3 Accuracy': f"{data['top3'] * 100:.2f}%",
                'Training Time (h)': f"{data['time'] / 3600:.2f}",
                'Params (M)': params_str
            })

        file_exists = os.path.isfile('results/comparison.csv')
        with open('results/comparison.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Model', 'Dataset', 'Top1 Accuracy', 'Top3 Accuracy',
                'Training Time (h)', 'Params (M)'
            ])
            if not file_exists or f.tell() == 0:
                writer.writeheader()
            writer.writerows(results)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(log_preds)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        loss = -torch.sum(targets * log_preds, dim=-1)
        return torch.mean(loss)


if __name__ == '__main__':
    config = Config()

    with open('results/comparison.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('')

    for dataset in ['shoe']:
        print(f"\n===== Starting {dataset.upper()} Experiments =====")
        runner = ExperimentRunner(config, dataset_type=dataset)
        runner.run_all()

        print(f"\n=== {dataset.upper()} Final Results ===")
        for model_name, data in runner.best_results.items():
            params_str = f"{data['params'] / 1e6:.2f}M" if isinstance(data['params'], int) else 'N/A'
            print(f"{model_name}:")
            print(f"  Top1: {data['top1'] * 100:.2f}%")
            print(f"  Top3: {data['top3'] * 100:.2f}%")
            print(f"  Params: {params_str}")
            print(f"  Time: {data['time'] / 3600:.2f}h\n")
