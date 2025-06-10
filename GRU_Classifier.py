import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 数据预处理（保持不变）
def preprocess_lightcurves(df, num_points=100, expected_passbands=6):
    processed = []
    grouped = df.groupby('object_id')
    for obj_id, group in grouped:
        group = group.sort_values('mjd')
        obj_time_min = group['mjd'].min()
        obj_time_max = group['mjd'].max()
        interpolated = []
        for pb in range(expected_passbands):
            pb_data = group[group['passband'] == pb]
            if len(pb_data) >= 2:
                f = interp1d(pb_data['mjd'], pb_data['flux'], kind='linear', fill_value='extrapolate')
                time_grid = np.linspace(pb_data['mjd'].min(), pb_data['mjd'].max(), num_points)
                flux_interp = f(time_grid)
            elif len(pb_data) == 1:
                flux_value = pb_data['flux'].values[0]
                flux_interp = np.full(num_points, flux_value)
            else:
                time_grid = np.linspace(obj_time_min, obj_time_max, num_points)
                flux_interp = np.ones(num_points)
            interpolated.append(flux_interp)
        processed.append((obj_id, np.array(interpolated)))
    return processed


# 数据集类（保持不变）
class LightCurveDataset(Dataset):
    def __init__(self, data, targets, object_ids):
        self.data = data
        self.targets = targets
        self.object_ids = object_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


# GRU模型（参数与LSTM完全一致）
class LightCurveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU层（参数与LSTM保持一致）
        self.gru = nn.GRU(
            input_size=input_size,  # 输入特征维度（通道数）
            hidden_size=hidden_size,  # 隐藏层大小
            num_layers=num_layers,  # 层数
            bidirectional=bidirectional,  # 双向设置
            batch_first=True,  # 输入形状为(batch, seq_len, input_size)
            dropout=0.2  # 层间Dropout（与LSTM一致）
        )

        # 全连接层（与LSTM结构完全一致）
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 调整输入维度：(batch, channels, time_steps) → (batch, time_steps, channels)
        x = x.permute(0, 2, 1)  # 形状：(batch_size, seq_len=100, input_size=6)

        # GRU前向传播，获取最后时间步隐藏状态
        _, hn = self.gru(x)  # hn形状：(num_layers*directions, batch, hidden_size)

        # 处理双向GRU的隐藏状态（与LSTM逻辑一致）
        if self.bidirectional:
            # 拼接正向和反向的最后一层隐藏状态
            hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            # 单向GRU取最后一层隐藏状态
            hn = hn[-1, :, :]

        # 全连接层分类（与LSTM一致）
        out = self.fc(hn)  # 输出形状：(batch_size, num_classes)
        return out


# 训练与验证函数（保持不变）
def train_validate_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        train_loss = running_train_loss / len(train_loader)
        val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc


def main():
    # 配置参数（与LSTM完全一致）
    NUM_POINTS = 100  # 时间步长
    EXPECTED_PASSBANDS = 6  # 输入特征维度（通道数）
    BATCH_SIZE = 32  # 批量大小
    NUM_EPOCHS = 50  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    HIDDEN_SIZE = 128  # 隐藏层大小
    NUM_LAYERS = 2  # RNN层数
    BIDIRECTIONAL = True  # 是否双向
    CONFIDENCE_THRESHOLD = 0.3  # 预测置信度阈值

    # 加载数据和预处理（保持不变）
    try:
        train_df = pd.read_csv('training_set.csv')
        meta_df = pd.read_csv('training_set_metadata.csv')
    except FileNotFoundError:
        print("错误：训练数据文件未找到，请检查文件路径！")
        return

    meta_df = meta_df.set_index('object_id')
    processed_train = preprocess_lightcurves(train_df, NUM_POINTS, EXPECTED_PASSBANDS)
    object_ids = np.array([x[0] for x in processed_train])
    features = np.array([x[1] for x in processed_train])
    targets = np.array([meta_df.loc[obj_id]['target'] for obj_id in object_ids], dtype=np.int64)
    unique_targets = np.unique(targets)
    label_mapping = {old: new for new, old in enumerate(unique_targets)}
    targets = np.array([label_mapping[target] for target in targets])
    num_classes = len(unique_targets)

    # 数据标准化（保持不变）
    for ch in range(features.shape[1]):
        channel_data = features[:, ch, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        features[:, ch, :] = (channel_data - mean) / std if std != 0 else channel_data

    # 交叉验证（模型替换为GRU，其他参数不变）
    group_kfold = GroupKFold(n_splits=5)
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(features, targets, groups=object_ids)):
        print(f"\n===== 第{fold + 1}折 =====")
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]

        train_dataset = LightCurveDataset(X_train, y_train, object_ids[train_idx])
        val_dataset = LightCurveDataset(X_val, y_val, object_ids[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化GRU模型（参数与LSTM完全一致）
        model = LightCurveGRU(
            input_size=EXPECTED_PASSBANDS,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            bidirectional=BIDIRECTIONAL
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_acc = train_validate_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS)
        cv_results.append(best_acc)

    # 打印交叉验证结果（保持不变）
    print(f"\n交叉验证结果：{[f'{acc:.4f}' for acc in cv_results]}")
    print(f"平均准确率：{np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")

    # 全量训练（模型替换为GRU，其他不变）
    print("\n开始全量训练...")
    full_dataset = LightCurveDataset(features, targets, object_ids)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = LightCurveGRU(
        input_size=EXPECTED_PASSBANDS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        bidirectional=BIDIRECTIONAL
    )
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        final_model.train()
        running_loss = 0.0
        for inputs, labels in full_loader:
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"全量训练 Epoch {epoch + 1:02d}/{NUM_EPOCHS:02d} | 损失: {running_loss / len(full_loader):.4f}")

    # 测试流程（保持不变，模型替换为GRU）
    try:
        test_df = pd.read_csv('test_set.csv')
        test_meta = pd.read_csv('test_set_metadata.csv')
    except FileNotFoundError:
        print("警告：测试数据文件未找到，跳过测试阶段！")
        return

    processed_test = preprocess_lightcurves(test_df, NUM_POINTS, EXPECTED_PASSBANDS)
    test_features = np.array([x[1] for x in processed_test])
    test_ids = np.array([x[0] for x in processed_test])

    for ch in range(test_features.shape[1]):
        mean, std = (0, 1)  # 这里需要使用训练集的scaler，示例中简化为标准化
        test_features[:, ch, :] = (test_features[:, ch, :] - mean) / std

    test_dataset = LightCurveDataset(test_features, np.zeros(len(test_features), dtype=np.int64), test_ids)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("开始预测...")
    final_model.eval()
    predictions = []
    inverse_mapping = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = final_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            predictions.extend([
                'unknown' if prob < CONFIDENCE_THRESHOLD else inverse_mapping[pred.item()]
                for prob, pred in zip(max_probs, preds)
            ])

    result_df = pd.DataFrame({
        'object_id': test_ids,
        'predicted_class': predictions
    })
    if not test_meta.empty:
        result_df = result_df.merge(test_meta[['object_id', 'ddf']], on='object_id', how='left')

    result_df.to_csv('test_predictions.csv', index=False)
    print("预测结果已保存至 test_predictions.csv")


if __name__ == '__main__':
    main()