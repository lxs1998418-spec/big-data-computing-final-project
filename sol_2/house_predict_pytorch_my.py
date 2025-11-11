import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import psutil
import os

# 设置随机种子以确保可重复性
def set_seed(seed=36):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(36)


# 内存监控函数
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # 转换为MB


# 绘制内存使用图表
def plot_memory_usage(memory_history):
    if not memory_history:
        return
    
    labels = [item['label'] for item in memory_history]
    memory_mb = [item['memory_mb'] for item in memory_history]
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(memory_mb)), memory_mb, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('memory usage (MB)', fontsize=12)
    plt.title('Memory Usage During Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(range(len(memory_mb)), memory_mb)):
        plt.annotate(f'{y:.1f}MB', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # 添加峰值内存标注
    peak_idx = np.argmax(memory_mb)
    peak_value = memory_mb[peak_idx]
    plt.annotate(f'峰值: {peak_value:.1f}MB', 
                xy=(peak_idx, peak_value), 
                xytext=(peak_idx, peak_value + max(memory_mb) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('memory_usage.png', dpi=150, bbox_inches='tight')
    plt.close()


# 数据预处理函数
def preprocess_data(train_df, test_df):
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    print("开始数据预处理...")
    
    # 处理分类变量
    categorical_cols = ['country', 'property_type', 'furnishing_status']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined_data)
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        label_encoders[col] = le

    # 特征工程
    
    # 可负担性比率
    train_processed['affordability_ratio'] = train_processed['customer_salary'] / (train_processed['price'] + 1)
    test_processed['affordability_ratio'] = test_processed['customer_salary'] / (test_processed['price'] + 1)

    # 贷款价值比
    train_processed['loan_to_value'] = train_processed['loan_amount'] / (train_processed['price'] + 1)
    test_processed['loan_to_value'] = test_processed['loan_amount'] / (test_processed['price'] + 1)

    # 房产年龄
    current_year = 2025
    train_processed['property_age'] = current_year - train_processed['constructed_year']
    test_processed['property_age'] = current_year - test_processed['constructed_year']

    # 支付能力
    train_processed['payment_capacity'] = train_processed['customer_salary'] - train_processed['monthly_expenses']
    test_processed['payment_capacity'] = test_processed['customer_salary'] - test_processed['monthly_expenses']

    # 首付比率
    train_processed['down_payment_ratio'] = train_processed['down_payment'] / (train_processed['price'] + 1)
    test_processed['down_payment_ratio'] = test_processed['down_payment'] / (test_processed['price'] + 1)

    # 风险评分
    train_processed['risk_score'] = train_processed['crime_cases_reported'] + train_processed['legal_cases_on_property']
    test_processed['risk_score'] = test_processed['crime_cases_reported'] + test_processed['legal_cases_on_property']

    # 质量评分
    train_processed['quality_score'] = train_processed['satisfaction_score'] + train_processed['neighbourhood_rating'] + \
                                       train_processed['connectivity_score']
    test_processed['quality_score'] = test_processed['satisfaction_score'] + test_processed['neighbourhood_rating'] + \
                                      test_processed['connectivity_score']

    print("数据预处理完成!")
    return train_processed, test_processed, label_encoders


# 数据集类 用于加载和预处理数据
class HouseDataset(Dataset):

    def __init__(self, features, labels=None):
        # 转换为numpy数组
        if isinstance(features, pd.DataFrame):
            self.features = features.values.astype(np.float32)
        else:
            self.features = features.astype(np.float32)
        
        # 如果有标签，转换为numpy数组
        if labels is not None:
            if isinstance(labels, pd.Series):
                self.labels = labels.values.astype(np.float32)
            else:
                self.labels = labels.astype(np.float32)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.features)
    
    # 获取单个样本 返回特征和标签
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # 如果有标签，返回特征和标签
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features, label
        else:
            # 如果没有标签，只返回特征
            return features


# 自定义深度神经网络模型 包含多层全连接网络、批归一化、Dropout等
class CustomDeepNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], 
                 output_size=1, dropout_rate=0.3, use_batch_norm=True,
                 activation='relu'):
        """
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表，例如 [256, 128, 64] 表示三层隐藏层
            output_size: 输出维度（二分类为1）
            dropout_rate: Dropout比率，用于防止过拟合
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型 ('relu', 'tanh', 'leaky_relu')
        """
        super(CustomDeepNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        # 构建隐藏层
        for i in range(len(layer_sizes) - 1):
            # 全连接层
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # 批归一化（可选）
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            # 激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())
            
            # Dropout（可选）
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())  # 二分类使用Sigmoid激活
        
        # 将层组合成Sequential模块
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        自定义权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001,
                device='cpu', early_stopping=True, patience=15, memory_history=None):
    """
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备 ('cpu' 或 'cuda')
        early_stopping: 是否使用早停
        patience: 早停的耐心值
    
    Returns:
        training_history: 训练历史记录（损失和准确率）
    """
    # 将模型移到指定设备
    model = model.to(device)
    
    # 定义损失函数（二分类交叉熵）
    criterion = nn.BCELoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 学习率调度器（可选，用于动态调整学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 跟踪学习率变化
    current_lr = learning_rate
    
    print(f"开始训练模型...")
    print(f"设备: {device}")
    print(f"优化器: Adam, 学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    
    if memory_history is not None:
        memory_history.append({'label': '训练开始前', 'memory_mb': get_memory_usage()})
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)  # 添加维度以匹配输出
            
            # 前向传播
            optimizer.zero_grad()  # 清零梯度
            outputs = model(batch_features)
            
            # 计算损失
            loss = criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        
        # 模型 验证阶段
        model.eval()  # 设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 验证时不需要计算梯度
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # 更新学习率
        old_lr = current_lr
        scheduler.step(avg_val_loss)
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 如果学习率变化了，打印出来
        if old_lr != current_lr:
            print(f"学习率从 {old_lr:.6f} 调整为 {current_lr:.6f}")
        
        # 早停检查
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                # 恢复最佳模型
                model.load_state_dict(best_model_state)
                break
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            if memory_history is not None:
                memory_history.append({'label': f'Epoch {epoch + 1}', 'memory_mb': get_memory_usage()})
    
    training_time = time.time() - start_time
    print(f"\n训练完成, 耗时: {training_time:.2f}秒")
    
    if memory_history is not None:
        memory_history.append({'label': '训练完成后', 'memory_mb': get_memory_usage()})
    
    return history


# 评估模型性能
def evaluate_model(model, data_loader, device='cpu', memory_history=None):
    """
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        (loss, accuracy, predictions, probabilities)
    """
    model.eval()
    criterion = nn.BCELoss()
    
    total_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(batch_labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    if memory_history is not None:
        memory_history.append({'label': '评估完成后', 'memory_mb': get_memory_usage()})
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_probabilities)




# 预测函数
def predict(model, data_loader, device='cpu', memory_history=None):
    """对测试集进行预测"""
    model.eval()
    all_predictions = []
    
    if memory_history is not None:
        memory_history.append({'label': '预测开始前', 'memory_mb': get_memory_usage()})
    
    with torch.no_grad():
        for batch_features in data_loader:
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
            batch_features = batch_features.to(device)
            
            outputs = model(batch_features)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            all_predictions.extend(predictions)
    
    if memory_history is not None:
        memory_history.append({'label': '预测完成后', 'memory_mb': get_memory_usage()})
    
    return np.array(all_predictions).astype(int)


# 绘制训练历史
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 初始化内存记录列表
    memory_history = []
    memory_history.append({'label': '程序启动', 'memory_mb': get_memory_usage()})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    memory_history.append({'label': '数据加载后', 'memory_mb': get_memory_usage()})
    
    print(f"训练数据形状: {train_df.shape}, 测试数据形状: {test_df.shape}")
    
    # 数据预处理
    print("\n预处理数据...")
    train_processed, test_processed, encoders = preprocess_data(train_df, test_df)
    memory_history.append({'label': '数据预处理后', 'memory_mb': get_memory_usage()})
    
    # 准备特征和目标
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    X_train = train_processed[feature_cols]
    y_train = train_processed['label']
    X_test = test_processed[feature_cols]
    
    
    # 数据标准化
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    memory_history.append({'label': '数据标准化后', 'memory_mb': get_memory_usage()})
    
    # 分割数据用于验证
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=36, stratify=y_train
    )
    
    print(f"训练集大小: {X_train_split.shape[0]}")
    print(f"验证集大小: {X_val_split.shape[0]}")
    

    print("创建数据集和数据加载器...")
    # 创建数据集
    train_dataset = HouseDataset(X_train_split, y_train_split)
    val_dataset = HouseDataset(X_val_split, y_val_split)
    test_dataset = HouseDataset(X_test_scaled)
    
    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    memory_history.append({'label': '数据集创建后', 'memory_mb': get_memory_usage()})

    
    # 创建模型
    model = CustomDeepNeuralNetwork(
        input_size=X_train_scaled.shape[1],
        hidden_sizes=[256, 128, 64, 32],  # 四层隐藏层
        output_size=1,
        dropout_rate=0.3,
        use_batch_norm=True,
        activation='relu'
    )
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    memory_history.append({'label': '模型创建后', 'memory_mb': get_memory_usage()})
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=0.001,
        device=device,
        early_stopping=True,
        patience=15,
        memory_history=memory_history
    )
    
    # 评估模型
    print("评估模型性能...")
    val_loss, val_accuracy, val_predictions, val_probabilities = evaluate_model(
        model, val_loader, device=device, memory_history=memory_history
    )
    
    # 计算Macro-F1
    val_macro_f1 = f1_score(y_val_split, val_predictions, average='macro')
    
    print(f"\n验证集性能:")
    print(f"损失: {val_loss:.4f}")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"Macro-F1: {val_macro_f1:.4f}")
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    plot_training_history(history)
    
    # 生成测试集预测
    print("\n" + "="*60)
    print("生成测试集预测...")
    start_time = time.time()
    test_predictions = predict(model, test_loader, device=device, memory_history=memory_history)
    prediction_time = time.time() - start_time
    
    print(f"测试集预测完成，耗时: {prediction_time:.2f}秒")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': test_predictions
    })
    
    submission_filename = 'submission_pytorch.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"测试预测分布:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本 ({count / len(test_predictions) * 100:.1f}%)")
    
    # 保存模型（可选）
    model_save_path = 'best_model_pytorch.pth'
    torch.save(model.state_dict(), model_save_path)
    
    # 记录最终内存并绘制图表
    memory_history.append({'label': '程序结束', 'memory_mb': get_memory_usage()})
    
    # 绘制内存使用图表
    print("\n绘制内存使用图表...")
    plot_memory_usage(memory_history)
    
    # 打印内存统计
    memory_values = [item['memory_mb'] for item in memory_history]
    peak_memory = max(memory_values)
    start_memory = memory_values[0]
    total_increase = peak_memory - start_memory
    
 
    print("内存使用总结:")
    print(f"初始内存: {start_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB ({peak_memory/1024:.2f} GB)")
    print(f"最终内存: {memory_values[-1]:.2f} MB")
    print(f"总内存增长: {total_increase:.2f} MB ({total_increase/1024:.2f} GB)")
    
    print("\n" + "="*60)
    print("任务完成！")

