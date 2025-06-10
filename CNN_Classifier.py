import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 数据预处理
def preprocess_lightcurves(df, num_points=100, expected_passbands=6):
    processed = []
    grouped = df.groupby('object_id')

    for obj_id, group in grouped:
        group = group.sort_values('mjd')
        interpolated = []

        for pb in range(expected_passbands):
            if pb in group['passband'].values:
                pb_data = group[group['passband'] == pb]
                if len(pb_data) < 2:
                    flux_interp = np.zeros(num_points)
                else:
                    f = interp1d(pb_data['mjd'], pb_data['flux'],
                                 kind='linear', fill_value='extrapolate')
                    time_grid = np.linspace(pb_data['mjd'].min(),
                                            pb_data['mjd'].max(),
                                            num_points)
                    flux_interp = f(time_grid)
            else:
                flux_interp = np.zeros(num_points)

            interpolated.append(flux_interp)

        processed.append((obj_id, np.array(interpolated)))

    return processed


# 数据集类
class LightCurveDataset(Dataset):
    def __init__(self, data, targets, object_ids):
        self.data = data
        self.targets = targets
        self.object_ids = object_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.data[idx]),
                torch.LongTensor([self.targets[idx]]))


# CNN模型
class LightCurveCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # 加载训练数据
    train_df = pd.read_csv('training_set.csv')
    meta_df = pd.read_csv('training_set_metadata.csv')

    # 预处理
    processed = preprocess_lightcurves(train_df)
    object_ids = np.array([x[0] for x in processed])
    features = np.array([x[1] for x in processed])

    # 标签处理
    meta = meta_df.set_index('object_id')
    targets = np.array([meta.loc[obj_id]['target'] for obj_id in object_ids])
    unique_targets = np.unique(targets)
    label_mapping = {old: new for new, old in enumerate(unique_targets)}
    targets = np.vectorize(label_mapping.get)(targets)
    num_classes = len(unique_targets)

    # 标准化
    channel_means = []
    channel_stds = []
    for i in range(features.shape[1]):
        mean = np.mean(features[:, i, :])
        std = np.std(features[:, i, :])
        features[:, i, :] = (features[:, i, :] - mean) / std
        channel_means.append(mean)
        channel_stds.append(std)

    # 交叉验证
    group_kfold = GroupKFold(n_splits=5)
    results = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(features, targets, groups=object_ids)):
        # 训练验证流程...
        pass  # 原始交叉验证代码

    # 全量训练
    print("\nTraining final model...")
    final_model = LightCurveCNN(features.shape[1], num_classes)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    full_loader = DataLoader(
        LightCurveDataset(features, targets, object_ids),
        batch_size=32,
        shuffle=True
    )

    for epoch in range(50):
        final_model.train()
        for inputs, labels in full_loader:
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

    # 测试流程
    test_df = pd.read_csv('test_set_sample.csv')
    test_meta = pd.read_csv('test_set_metadata.csv')

    # 测试数据预处理
    processed_test = preprocess_lightcurves(test_df)
    test_features = np.array([x[1] for x in processed_test])
    test_ids = np.array([x[0] for x in processed_test])

    # 应用训练数据的标准化
    for i in range(test_features.shape[1]):
        test_features[:, i, :] = (test_features[:, i, :] - channel_means[i]) / channel_stds[i]

    # 预测
    test_loader = DataLoader(
        LightCurveDataset(test_features, np.zeros(len(test_features)), test_ids),
        batch_size=32,
        shuffle=False
    )

    final_model.eval()
    predictions = []
    confidence_threshold = 0.3
    inverse_mapping = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = final_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for prob, pred in zip(max_probs, preds):
                if prob < confidence_threshold:
                    predictions.append('unknown')
                else:
                    predictions.append(inverse_mapping[pred.item()])

    # 保存结果
    result_df = pd.DataFrame({
        'object_id': test_ids,
        'predicted_class': predictions
    }).merge(test_meta[['object_id', 'ddf']], on='object_id')

    result_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to test_predictions.csv")


if __name__ == '__main__':
    main()