
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== 数据预处理函数 ====================
def preprocess_data(train_df, test_df):
    """
    数据预处理函数
    包括分类变量编码和特征工程
    """
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    print("开始数据预处理...")
    
    # 1. 处理分类变量
    print("1. 处理分类变量...")
    categorical_cols = ['country', 'property_type', 'furnishing_status']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined_data)
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        label_encoders[col] = le

    # 2. 特征工程 - 创建有意义的特征
    print("2. 创建衍生特征...")
    
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


# ==================== 自定义数据集类 ====================
class HouseDataset(Dataset):
    """
    自定义PyTorch数据集类
    用于加载和预处理数据
    """
    def __init__(self, features, labels=None):
        """
        初始化数据集
        
        Args:
            features: 特征数据 (numpy array 或 pandas DataFrame)
            labels: 标签数据 (可选，用于训练集)
        """
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
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            如果是有标签数据（训练集），返回 (features, label)
            如果是无标签数据（测试集），只返回 features
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features, label
        else:
            return features


# ==================== 自定义深度神经网络模型 ====================
class CustomDeepNeuralNetwork(nn.Module):
    """
    自定义深度神经网络模型
    使用PyTorch实现，包含多层全连接网络、批归一化、Dropout等
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], 
                 output_size=1, dropout_rate=0.3, use_batch_norm=True,
                 activation='relu'):
        """
        初始化神经网络
        
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
        使用Xavier初始化（也称为Glorot初始化）来初始化权重
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
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_size)
        
        Returns:
            输出张量 (batch_size, output_size)，经过Sigmoid激活，值在[0,1]之间
        """
        return self.network(x)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001,
                optimizer_type='adam', device='cpu', early_stopping=True, patience=15):
    """
    训练模型的完整实现
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        optimizer_type: 优化器类型 ('adam', 'sgd')
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
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
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
    print(f"优化器: {optimizer_type}, 学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()  # 设置为训练模式
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            # 将数据移到设备
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
        
        # ========== 验证阶段 ==========
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
    
    training_time = time.time() - start_time
    print(f"\n训练完成！耗时: {training_time:.2f}秒")
    
    return history


# ==================== 评估函数 ====================
def evaluate_model(model, data_loader, device='cpu'):
    """
    评估模型性能
    
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
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_probabilities)


# ==================== 预测函数 ====================
def predict(model, data_loader, device='cpu'):
    """
    对测试集进行预测
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        predictions: 预测结果（0或1）
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_features in data_loader:
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
            batch_features = batch_features.to(device)
            
            outputs = model(batch_features)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            all_predictions.extend(predictions)
    
    return np.array(all_predictions).astype(int)


# ==================== 可视化训练历史 ====================
def plot_training_history(history, figsize=(12, 5)):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n" + "="*60)
    print("加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 数据预处理
    print("\n" + "="*60)
    print("预处理数据...")
    train_processed, test_processed, encoders = preprocess_data(train_df, test_df)
    
    # 准备特征和目标
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    X_train = train_processed[feature_cols]
    y_train = train_processed['label']
    X_test = test_processed[feature_cols]
    
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols[:5]}... (共{len(feature_cols)}个)")
    
    # 数据标准化（对神经网络很重要）
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 分割数据用于验证
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"训练集大小: {X_train_split.shape[0]}")
    print(f"验证集大小: {X_val_split.shape[0]}")
    
    # 创建数据集和数据加载器
    print("\n" + "="*60)
    print("创建数据集和数据加载器...")
    train_dataset = HouseDataset(X_train_split, y_train_split)
    val_dataset = HouseDataset(X_val_split, y_val_split)
    test_dataset = HouseDataset(X_test_scaled)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"批次大小: {batch_size}")
    
    # 创建模型
    print("\n" + "="*60)
    print("创建深度神经网络模型...")
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
    
    # 训练模型
    print("\n" + "="*60)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=0.001,
        optimizer_type='adam',
        device=device,
        early_stopping=True,
        patience=15
    )
    
    # 评估模型
    print("\n" + "="*60)
    print("评估模型性能...")
    val_loss, val_accuracy, val_predictions, val_probabilities = evaluate_model(
        model, val_loader, device=device
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
    test_predictions = predict(model, test_loader, device=device)
    prediction_time = time.time() - start_time
    
    print(f"测试集预测完成，耗时: {prediction_time:.2f}秒")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': test_predictions
    })
    
    submission_filename = 'submission_pytorch.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\n提交文件 '{submission_filename}' 创建成功!")
    print(f"测试预测分布:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本 ({count / len(test_predictions) * 100:.1f}%)")
    
    # 保存模型（可选）
    model_save_path = 'best_model_pytorch.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\n模型已保存到: {model_save_path}")
    
    print("\n" + "="*60)
    print("任务完成！")

