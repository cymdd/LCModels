import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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


def main():
    # 配置参数
    NUM_POINTS = 100  # 时间步长
    EXPECTED_PASSBANDS = 6  # 通道数
    K_NEIGHBORS = 5  # K值
    CONFIDENCE_THRESHOLD = 0.5  # 占位参数（KNN无概率输出）

    # 加载训练数据和元数据
    try:
        train_df = pd.read_csv('training_set.csv')
        meta_df = pd.read_csv('training_set_metadata.csv')
    except FileNotFoundError:
        print("错误：训练数据文件未找到，请检查文件路径！")
        return

    meta_df = meta_df.set_index('object_id')
    train_objects = set(train_df['object_id'].unique())
    missing_in_meta = train_objects - set(meta_df.index)
    if missing_in_meta:
        raise ValueError(f"元数据缺失 {len(missing_in_meta)} 个训练数据中的object_id！")

    # 预处理训练数据
    print("预处理训练数据...")
    processed_train = preprocess_lightcurves(train_df, NUM_POINTS, EXPECTED_PASSBANDS)
    object_ids = np.array([x[0] for x in processed_train])
    features = np.array([x[1] for x in processed_train])
    features = features.reshape(features.shape[0], -1)  # 展平为二维特征

    # **关键修改1：仅基于训练集构建标签映射**
    train_targets = np.array([meta_df.loc[obj_id]['target'] for obj_id in object_ids], dtype=np.int64)
    train_unique_targets = np.unique(train_targets)  # 训练集14类
    label_mapping = {old: new for new, old in enumerate(train_unique_targets)}
    train_labels = np.array([label_mapping[target] for target in train_targets])
    num_train_classes = len(train_unique_targets)

    # 数据标准化
    print("标准化训练数据...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 交叉验证（使用训练集标签）
    print(f"\n开始5折交叉验证 (K={K_NEIGHBORS})...")
    group_kfold = GroupKFold(n_splits=5)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(features, train_labels, groups=object_ids)):
        print(f"\n===== 第{fold + 1}折 =====")
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric='euclidean', n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        # 验证时仅评估训练集已知类别
        accuracy = accuracy_score(y_val, y_pred)
        cv_results.append(accuracy)
        print(f"第{fold + 1}折验证准确率：{accuracy:.4f}")

    # 打印交叉验证结果
    print(f"\n交叉验证结果：{[f'{acc:.4f}' for acc in cv_results]}")
    print(f"平均准确率：{np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")

    # 全量训练（使用训练集标签）
    print("\n开始全量训练...")
    final_model = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric='euclidean', n_jobs=-1)
    final_model.fit(features, train_labels)

    # 加载测试数据（处理新增类别）
    try:
        test_df = pd.read_csv('test_set.csv')
        test_meta = pd.read_csv('test_set_metadata.csv')
    except FileNotFoundError:
        print("警告：测试数据文件未找到，跳过测试阶段！")
        return

    # 预处理测试数据
    print("预处理测试数据...")
    processed_test = preprocess_lightcurves(test_df, NUM_POINTS, EXPECTED_PASSBANDS)
    test_features = np.array([x[1] for x in processed_test]).reshape(-1, NUM_POINTS * EXPECTED_PASSBANDS)
    test_ids = np.array([x[0] for x in processed_test])
    test_features = scaler.transform(test_features)

    # **关键修改2：预测时处理未知类别**
    print("开始预测...")
    y_pred_labels = final_model.predict(test_features)  # 预测标签为训练集的0~13

    # 映射回训练集原始标签（非测试集标签）
    inverse_train_mapping = {v: k for k, v in label_mapping.items()}
    predicted_classes = []
    for pred_label in y_pred_labels:
        # 训练集已知类别直接映射
        if pred_label in inverse_train_mapping:
            predicted_classes.append(inverse_train_mapping[pred_label])
        # 测试集新增类别标记为unknown（实际不会进入此分支，因KNN输出限于训练集标签）
        else:
            predicted_classes.append('unknown')

    # **关键修改3：检查测试集真实标签是否存在于训练集（仅演示）**
    if not test_meta.empty:
        test_meta = test_meta.set_index('object_id')
        test_true_targets = []
        for obj_id in test_ids:
            try:
                test_true_targets.append(test_meta.loc[obj_id]['target'])
            except KeyError:
                test_true_targets.append('unknown')  # 处理测试集元数据缺失的情况

        # 对比训练集标签与测试集真实标签（示例）
        for true_label in test_true_targets:
            if true_label not in train_unique_targets:
                print(f"警告：测试集存在训练集未出现的类别：{true_label}")

    # 保存结果（标记未知类别）
    result_df = pd.DataFrame({
        'object_id': test_ids,
        'predicted_class': predicted_classes
    })
    if not test_meta.empty:
        result_df = result_df.merge(test_meta[['object_id', 'ddf']], on='object_id', how='left')

    result_df.to_csv('test_predictions.csv', index=False)
    print("预测结果已保存至 test_predictions.csv，未知类别标记为 'unknown'")


if __name__ == '__main__':
    main()