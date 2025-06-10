import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 数据预处理（线性插值，无零值填充）
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
                # 线性插值（至少2个点）
                f = interp1d(pb_data['mjd'], pb_data['flux'], kind='linear', fill_value='extrapolate')
                time_grid = np.linspace(pb_data['mjd'].min(), pb_data['mjd'].max(), num_points)
                flux_interp = f(time_grid)
            elif len(pb_data) == 1:
                # 单个点时，用该点值填充整个时间序列
                flux_value = pb_data['flux'].values[0]
                flux_interp = np.full(num_points, flux_value)
            else:
                # 无数据时，用对象整体时间范围，假设通量为1（非零常数）
                time_grid = np.linspace(obj_time_min, obj_time_max, num_points)
                flux_interp = np.ones(num_points)  # 非零填充，可根据需求调整

            interpolated.append(flux_interp)

        processed.append((obj_id, np.array(interpolated)))

    return processed


# 数据集类（正确返回一维标签）
class LightCurveDataset(Dataset):
    def __init__(self, data, targets, object_ids):
        self.data = data
        self.targets = targets
        self.object_ids = object_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 输入形状：(通道数, 时间步)，标签形状：(1,)（一维张量）
        return (
            torch.FloatTensor(self.data[idx]),
            torch.tensor(self.targets[idx], dtype=torch.long)  # 直接转换为一维张量
        )


# RNN模型（LSTM）
class LightCurveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层配置
        self.lstm = nn.LSTM(
            input_size=input_size,          # 每个时间步的特征数（通道数）
            hidden_size=hidden_size,        # 隐藏层大小
            num_layers=num_layers,          # LSTM层数
            bidirectional=bidirectional,    # 是否双向
            batch_first=True,               # 输入形状为 (batch, seq_len, input_size)
            dropout=0.2                     # 层间Dropout防止过拟合
        )

        # 全连接层
        fc_input_size = hidden_size * (2 if bidirectional else 1)  # 双向时隐藏层大小翻倍
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 调整输入维度：(batch, channels, time_steps) → (batch, time_steps, channels)
        x = x.permute(0, 2, 1)  # 形状变为 (batch_size, seq_len=100, input_size=6)

        # LSTM前向传播，获取最后一层隐藏状态
        _, (hn, _) = self.lstm(x)  # hn形状：(num_layers*directions, batch, hidden_size)

        # 处理双向LSTM的隐藏状态
        if self.bidirectional:
            # 拼接正向和反向的最后一层隐藏状态
            hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            # 单向LSTM取最后一层隐藏状态
            hn = hn[-1, :, :]

        # 全连接层分类
        out = self.fc(hn)  # 输出形状：(batch_size, num_classes)
        return out


# 训练与验证函数
def train_validate_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # 验证阶段
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

        # 打印训练进度
        print(f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # 记录最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc


def main():
    # 配置参数（包含RNN特有参数）
    NUM_POINTS = 100           # 时间步长（序列长度）
    EXPECTED_PASSBANDS = 6     # 输入特征维度（每个时间步的特征数，即通道数）
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128         # LSTM隐藏层大小
    NUM_LAYERS = 2            # LSTM层数
    BIDIRECTIONAL = True      # 是否使用双向LSTM
    CONFIDENCE_THRESHOLD = 0.3 # 预测置信度阈值

    # 加载训练数据和元数据
    try:
        train_df = pd.read_csv('training_set.csv')
        meta_df = pd.read_csv('training_set_metadata.csv')
    except FileNotFoundError:
        print("错误：训练数据文件未找到，请检查文件路径！")
        return
    except Exception as e:
        print(f"数据加载错误：{str(e)}")
        return

    # 检查元数据与训练数据的object_id一致性
    meta_df = meta_df.set_index('object_id')
    train_objects = set(train_df['object_id'].unique())
    missing_in_meta = train_objects - set(meta_df.index)
    if missing_in_meta:
        raise ValueError(f"元数据缺失 {len(missing_in_meta)} 个训练数据中的object_id！")

    # 预处理训练数据
    print("预处理训练数据...")
    processed_train = preprocess_lightcurves(train_df, NUM_POINTS, EXPECTED_PASSBANDS)
    object_ids = np.array([x[0] for x in processed_train])
    features = np.array([x[1] for x in processed_train])  # 形状：(样本数, 通道数, 时间步)

    # 处理标签
    targets = []
    for obj_id in object_ids:
        targets.append(meta_df.loc[obj_id]['target'])
    targets = np.array(targets, dtype=np.int64)
    unique_targets = np.unique(targets)
    label_mapping = {old: new for new, old in enumerate(unique_targets)}
    targets = np.array([label_mapping[target] for target in targets])
    num_classes = len(unique_targets)

    # 数据标准化（每个通道独立标准化）
    print("标准化训练数据...")
    channel_scalers = []
    for ch in range(features.shape[1]):
        channel_data = features[:, ch, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        features[:, ch, :] = (channel_data - mean) / std if std != 0 else channel_data
        channel_scalers.append((mean, std))

    # 交叉验证
    print(f"\n开始{NUM_EPOCHS}轮5折交叉验证...")
    group_kfold = GroupKFold(n_splits=5)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(features, targets, groups=object_ids)):
        print(f"\n===== 第{fold + 1}折 =====")
        train_idx = train_idx.astype(int)
        val_idx = val_idx.astype(int)

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]

        # 创建数据集和DataLoader
        train_dataset = LightCurveDataset(X_train, y_train, object_ids[train_idx])
        val_dataset = LightCurveDataset(X_val, y_val, object_ids[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化RNN模型
        model = LightCurveRNN(
            input_size=EXPECTED_PASSBANDS,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            bidirectional=BIDIRECTIONAL
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # 训练并验证
        best_acc = train_validate_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS)
        cv_results.append(best_acc)
        print(f"第{fold + 1}折最佳验证准确率：{best_acc:.4f}")

    # 打印交叉验证汇总
    print(f"\n交叉验证结果：{[f'{acc:.4f}' for acc in cv_results]}")
    print(f"平均准确率：{np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")

    # 全量训练（使用全部训练数据）
    print("\n开始全量训练...")
    full_dataset = LightCurveDataset(features, targets, object_ids)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = LightCurveRNN(
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
            loss = criterion(outputs, labels)  # 直接使用一维标签，无需squeeze
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"全量训练 Epoch {epoch + 1:02d}/{NUM_EPOCHS:02d} | 损失: {running_loss / len(full_loader):.4f}")

    # 测试流程（假设存在测试数据）
    try:
        test_df = pd.read_csv('test_set.csv')
        test_meta = pd.read_csv('test_set_metadata.csv')
    except FileNotFoundError:
        print("警告：测试数据文件未找到，跳过测试阶段！")
        return

    # 预处理测试数据
    print("预处理测试数据...")
    processed_test = preprocess_lightcurves(test_df, NUM_POINTS, EXPECTED_PASSBANDS)
    test_features = np.array([x[1] for x in processed_test])
    test_ids = np.array([x[0] for x in processed_test])

    # 应用训练集的标准化
    for ch in range(test_features.shape[1]):
        mean, std = channel_scalers[ch]
        test_features[:, ch, :] = (test_features[:, ch, :] - mean) / std if std != 0 else test_features[:, ch, :]

    # 创建测试DataLoader（测试集无标签，用零填充）
    test_dataset = LightCurveDataset(test_features, np.zeros(len(test_features), dtype=np.int64), test_ids)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 预测
    print("开始预测...")
    final_model.eval()
    predictions = []
    inverse_mapping = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = final_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for prob, pred in zip(max_probs, preds):
                if prob < CONFIDENCE_THRESHOLD:
                    predictions.append('unknown')  # 低置信度标记为未知
                else:
                    predictions.append(inverse_mapping[pred.item()])

    # 保存结果
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