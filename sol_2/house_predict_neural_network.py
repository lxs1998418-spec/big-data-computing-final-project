# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据预处理函数 ====================
def preprocess_data(train_df, test_df):
    """数据预处理函数"""
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


# ==================== 激活函数 ====================
class ReLU:
    """ReLU激活函数"""
    @staticmethod
    def forward(x):
        """前向传播"""
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x, grad_output):
        """反向传播"""
        grad_input = grad_output.copy()
        grad_input[x <= 0] = 0
        return grad_input


class Sigmoid:
    """Sigmoid激活函数"""
    @staticmethod
    def forward(x):
        """前向传播"""
        # 防止溢出
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def backward(x, grad_output):
        """反向传播"""
        sigmoid_x = Sigmoid.forward(x)
        return grad_output * sigmoid_x * (1 - sigmoid_x)


class Tanh:
    """Tanh激活函数"""
    @staticmethod
    def forward(x):
        """前向传播"""
        return np.tanh(x)
    
    @staticmethod
    def backward(x, grad_output):
        """反向传播"""
        tanh_x = Tanh.forward(x)
        return grad_output * (1 - tanh_x ** 2)


# ==================== 损失函数 ====================
class BinaryCrossEntropy:
    """二分类交叉熵损失函数"""
    @staticmethod
    def forward(y_pred, y_true):
        """前向传播 - 计算损失"""
        # 防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # 二分类交叉熵
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    @staticmethod
    def backward(y_pred, y_true):
        """反向传播 - 计算梯度"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # 二分类交叉熵的梯度
        grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)
        return grad


# ==================== 全连接层 ====================
class LinearLayer:
    """全连接层"""
    def __init__(self, input_size, output_size, use_bias=True):
        """
        初始化全连接层
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            use_bias: 是否使用偏置
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Xavier初始化权重
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        
        # 初始化偏置
        if use_bias:
            self.bias = np.zeros((1, output_size))
        else:
            self.bias = None
        
        # 用于存储前向传播的输入，用于反向传播
        self.last_input = None
        
        # 用于存储梯度
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, input_size)
        
        Returns:
            输出数据 (batch_size, output_size)
        """
        self.last_input = x
        # 矩阵乘法: x @ weights + bias
        output = np.dot(x, self.weights)
        if self.use_bias:
            output += self.bias
        return output
    
    def backward(self, grad_output):
        """
        反向传播
        
        Args:
            grad_output: 来自上一层的梯度 (batch_size, output_size)
        
        Returns:
            传递给下一层的梯度 (batch_size, input_size)
        """
        batch_size = grad_output.shape[0]
        
        # 计算权重的梯度
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        
        # 计算偏置的梯度
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        # 计算输入的梯度（传递给前一层）
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate):
        """更新权重和偏置"""
        self.weights -= learning_rate * self.grad_weights
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias


# ==================== Dropout层 ====================
class Dropout:
    """Dropout正则化层"""
    def __init__(self, dropout_rate=0.5):
        """
        初始化Dropout层
        
        Args:
            dropout_rate: dropout比率
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            # 训练时应用dropout
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
            return x * self.mask
        else:
            # 测试时不应用dropout
            return x
    
    def backward(self, grad_output):
        """反向传播"""
        if self.training:
            return grad_output * self.mask
        else:
            return grad_output


# ==================== Batch Normalization层 ====================
class BatchNorm:
    """批归一化层"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        """
        初始化BatchNorm层
        
        Args:
            num_features: 特征数量
            momentum: 移动平均的动量
            eps: 防止除零的小值
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # 可学习参数
        self.gamma = np.ones((1, num_features))  # 缩放参数
        self.beta = np.zeros((1, num_features))  # 偏移参数
        
        # 运行时的均值和方差
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # 训练时的中间变量
        self.training = True
        self.last_input = None
        self.normalized = None
        self.std = None
    
    def forward(self, x):
        """前向传播"""
        self.last_input = x
        
        if self.training:
            # 训练时使用当前batch的统计量
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # 更新运行时的统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            mean = batch_mean
            var = batch_var
        else:
            # 测试时使用运行时的统计量
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        self.std = np.sqrt(var + self.eps)
        self.normalized = (x - mean) / self.std
        
        # 缩放和偏移
        output = self.gamma * self.normalized + self.beta
        return output
    
    def backward(self, grad_output):
        """反向传播"""
        batch_size = grad_output.shape[0]
        
        # 计算gamma和beta的梯度
        self.grad_gamma = np.sum(grad_output * self.normalized, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        
        # 计算输入的梯度
        grad_normalized = grad_output * self.gamma
        grad_var = np.sum(grad_normalized * (self.last_input - np.mean(self.last_input, axis=0, keepdims=True)), 
                         axis=0, keepdims=True) * (-0.5) * (self.std ** -3)
        grad_mean = np.sum(grad_normalized * (-1 / self.std), axis=0, keepdims=True) + \
                   grad_var * np.mean(-2 * (self.last_input - np.mean(self.last_input, axis=0, keepdims=True)), 
                                    axis=0, keepdims=True)
        grad_input = grad_normalized / self.std + grad_var * 2 * (self.last_input - np.mean(self.last_input, axis=0, keepdims=True)) / batch_size + \
                    grad_mean / batch_size
        
        return grad_input
    
    def update_weights(self, learning_rate):
        """更新参数"""
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta


# ==================== 优化器 ====================
class SGD:
    """随机梯度下降优化器"""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        初始化SGD优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, layer, layer_id):
        """更新层参数"""
        if layer_id not in self.velocity:
            self.velocity[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'bias': np.zeros_like(layer.bias) if layer.use_bias else None
            }
        
        # 更新权重
        if self.momentum > 0:
            self.velocity[layer_id]['weights'] = self.momentum * self.velocity[layer_id]['weights'] + layer.grad_weights
            layer.weights -= self.learning_rate * self.velocity[layer_id]['weights']
        else:
            layer.weights -= self.learning_rate * layer.grad_weights
        
        # 更新偏置
        if layer.use_bias:
            if self.momentum > 0:
                self.velocity[layer_id]['bias'] = self.momentum * self.velocity[layer_id]['bias'] + layer.grad_bias
                layer.bias -= self.learning_rate * self.velocity[layer_id]['bias']
            else:
                layer.bias -= self.learning_rate * layer.grad_bias


class Adam:
    """Adam优化器"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        初始化Adam优化器
        
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的衰减率
            beta2: 二阶矩估计的衰减率
            eps: 防止除零的小值
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = {}  # 时间步
    
    def update(self, layer, layer_id):
        """更新层参数"""
        if layer_id not in self.m:
            self.m[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'bias': np.zeros_like(layer.bias) if layer.use_bias else None
            }
            self.v[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'bias': np.zeros_like(layer.bias) if layer.use_bias else None
            }
            self.t[layer_id] = 0
        
        self.t[layer_id] += 1
        t = self.t[layer_id]
        
        # 更新权重
        self.m[layer_id]['weights'] = self.beta1 * self.m[layer_id]['weights'] + (1 - self.beta1) * layer.grad_weights
        self.v[layer_id]['weights'] = self.beta2 * self.v[layer_id]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2)
        
        m_hat = self.m[layer_id]['weights'] / (1 - self.beta1 ** t)
        v_hat = self.v[layer_id]['weights'] / (1 - self.beta2 ** t)
        
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # 更新偏置
        if layer.use_bias:
            self.m[layer_id]['bias'] = self.beta1 * self.m[layer_id]['bias'] + (1 - self.beta1) * layer.grad_bias
            self.v[layer_id]['bias'] = self.beta2 * self.v[layer_id]['bias'] + (1 - self.beta2) * (layer.grad_bias ** 2)
            
            m_hat = self.m[layer_id]['bias'] / (1 - self.beta1 ** t)
            v_hat = self.v[layer_id]['bias'] / (1 - self.beta2 ** t)
            
            layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


# ==================== 深度神经网络模型 ====================
class DeepNeuralNetwork:
    """自定义深度神经网络"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, 
                 activation='relu', dropout_rate=0.3, use_batch_norm=False, 
                 use_bias=True, random_state=42):
        """
        初始化深度神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度（二分类为1）
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
            dropout_rate: dropout比率
            use_batch_norm: 是否使用批归一化
            use_bias: 是否使用偏置
            random_state: 随机种子
        """
        np.random.seed(random_state)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层使用Sigmoid（二分类）
        self.output_activation = Sigmoid()
        
        # 构建网络层
        self.layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        
        # 输入层到第一个隐藏层
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            # 全连接层
            layer = LinearLayer(layer_sizes[i], layer_sizes[i+1], use_bias=use_bias)
            self.layers.append(layer)
            
            # Batch Normalization
            if use_batch_norm:
                bn_layer = BatchNorm(layer_sizes[i+1])
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)
            
            # Dropout
            if dropout_rate > 0:
                dropout_layer = Dropout(dropout_rate)
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)
        
        # 输出层
        output_layer = LinearLayer(hidden_sizes[-1], output_size, use_bias=use_bias)
        self.layers.append(output_layer)
        
        # 损失函数
        self.loss_fn = BinaryCrossEntropy()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def forward(self, x, training=True):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, input_size)
            training: 是否为训练模式
        
        Returns:
            输出 (batch_size, output_size)
        """
        # 设置所有层的训练模式
        for dropout_layer in self.dropout_layers:
            if dropout_layer is not None:
                dropout_layer.training = training
        for bn_layer in self.batch_norm_layers:
            if bn_layer is not None:
                bn_layer.training = training
        
        # 前向传播通过隐藏层
        # 保存激活函数的输入（用于反向传播）
        self.activation_inputs = []
        for i in range(len(self.layers) - 1):
            x = self.layers[i].forward(x)
            
            # Batch Normalization
            if self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i].forward(x)
            
            # 保存激活函数的输入
            self.activation_inputs.append(x.copy())
            
            # 激活函数
            x = self.activation.forward(x)
            
            # Dropout
            if self.dropout_layers[i] is not None:
                x = self.dropout_layers[i].forward(x)
        
        # 输出层
        x = self.layers[-1].forward(x)
        # 保存输出层的输入（用于反向传播）
        self.output_layer_input = x.copy()
        x = self.output_activation.forward(x)
        
        return x
    
    def backward(self, y_pred, y_true):
        """
        反向传播
        
        Args:
            y_pred: 预测值 (batch_size, output_size)
            y_true: 真实值 (batch_size, output_size)
        """
        # 计算损失函数的梯度
        grad = self.loss_fn.backward(y_pred, y_true)
        
        # 输出层的反向传播
        # 使用保存的输出层输入（激活函数的输入）
        grad = self.output_activation.backward(self.output_layer_input, grad)
        grad = self.layers[-1].backward(grad)
        
        # 隐藏层的反向传播（从后往前）
        for i in range(len(self.layers) - 2, -1, -1):
            # Dropout
            if self.dropout_layers[i] is not None:
                grad = self.dropout_layers[i].backward(grad)
            
            # 激活函数
            # 使用保存的激活函数输入
            activation_input = self.activation_inputs[i]
            grad = self.activation.backward(activation_input, grad)
            
            # Batch Normalization
            if self.batch_norm_layers[i] is not None:
                grad = self.batch_norm_layers[i].backward(grad)
            
            # 全连接层
            grad = self.layers[i].backward(grad)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=100, batch_size=32, learning_rate=0.001, 
            optimizer='adam', verbose=True, early_stopping=True, patience=10):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            optimizer: 优化器类型 ('sgd', 'adam')
            verbose: 是否打印训练过程
            early_stopping: 是否使用早停
            patience: 早停的耐心值
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        
        if X_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val).reshape(-1, 1)
        
        # 初始化优化器
        if optimizer == 'adam':
            optimizers = [Adam(learning_rate=learning_rate) for _ in self.layers]
        else:
            optimizers = [SGD(learning_rate=learning_rate) for _ in self.layers]
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        print(f"开始训练深度神经网络...")
        print(f"网络结构: {self.input_size} -> {' -> '.join(map(str, self.hidden_sizes))} -> {self.output_size}")
        print(f"训练参数: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, optimizer={optimizer}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 打乱数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批次训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward(batch_X, training=True)
                
                # 计算损失
                loss = self.loss_fn.forward(y_pred, batch_y)
                train_loss += loss
                
                # 计算准确率
                predictions = (y_pred > 0.5).astype(int)
                train_correct += np.sum(predictions == batch_y)
                train_total += len(batch_y)
                
                # 反向传播
                self.backward(y_pred, batch_y)
                
                # 更新权重
                for j, layer in enumerate(self.layers):
                    optimizers[j].update(layer, j)
                
                # 更新BatchNorm参数
                for j, bn_layer in enumerate(self.batch_norm_layers):
                    if bn_layer is not None:
                        bn_layer.update_weights(learning_rate)
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / (len(X_train) / batch_size)
            train_accuracy = train_correct / train_total
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # 验证阶段
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                
                # 早停检查
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳权重
                        best_weights = self._get_weights()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                        # 恢复最佳权重
                        self._set_weights(best_weights)
                        break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成！耗时: {training_time:.2f}秒")
        
        return self
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征
            y: 标签
        
        Returns:
            (loss, accuracy)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # 前向传播（测试模式）
        y_pred = self.forward(X, training=False)
        
        # 计算损失
        loss = self.loss_fn.forward(y_pred, y)
        
        # 计算准确率
        predictions = (y_pred > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        
        return loss, accuracy
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征
        
        Returns:
            预测类别 (0 or 1)
        """
        X = np.array(X)
        y_pred = self.forward(X, training=False)
        predictions = (y_pred > 0.5).astype(int).flatten()
        return predictions
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X: 特征
        
        Returns:
            预测概率
        """
        X = np.array(X)
        y_pred = self.forward(X, training=False)
        return y_pred.flatten()
    
    def _get_weights(self):
        """获取所有层的权重"""
        weights = []
        for layer in self.layers:
            weights.append({
                'weights': layer.weights.copy(),
                'bias': layer.bias.copy() if layer.use_bias else None
            })
        return weights
    
    def _set_weights(self, weights):
        """设置所有层的权重"""
        for i, layer in enumerate(self.layers):
            layer.weights = weights[i]['weights'].copy()
            if layer.use_bias:
                layer.bias = weights[i]['bias'].copy()
    
    def plot_training_history(self, figsize=(12, 5)):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        if len(self.val_losses) > 0:
            axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(self.train_accuracies, label='Train Accuracy', color='blue')
        if len(self.val_accuracies) > 0:
            axes[1].plot(self.val_accuracies, label='Val Accuracy', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    print("加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 数据预处理
    print("\n预处理数据...")
    train_processed, test_processed, encoders = preprocess_data(train_df, test_df)
    
    # 准备特征和目标
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    X_train = train_processed[feature_cols]
    y_train = train_processed['label']
    X_test = test_processed[feature_cols]
    
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols[:5]}... (共{len(feature_cols)}个)")
    
    # 数据标准化
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
    
    # 创建模型
    print("\n" + "="*60)
    print("创建深度神经网络模型...")
    model = DeepNeuralNetwork(
        input_size=X_train_scaled.shape[1],
        hidden_sizes=[128, 64, 32],  # 三层隐藏层
        output_size=1,
        activation='relu',
        dropout_rate=0.3,
        use_batch_norm=True,
        use_bias=True,
        random_state=42
    )
    
    # 训练模型
    print("\n" + "="*60)
    model.fit(
        X_train_split, y_train_split,
        X_val=X_val_split, y_val=y_val_split,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        optimizer='adam',
        verbose=True,
        early_stopping=True,
        patience=15
    )
    
    # 评估模型
    print("\n" + "="*60)
    print("评估模型性能...")
    val_loss, val_accuracy = model.evaluate(X_val_split, y_val_split)
    val_predictions = model.predict(X_val_split)
    val_macro_f1 = f1_score(y_val_split, val_predictions, average='macro')
    
    print(f"\n验证集性能:")
    print(f"损失: {val_loss:.4f}")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"Macro-F1: {val_macro_f1:.4f}")
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    model.plot_training_history()
    
    # 生成测试集预测
    print("\n" + "="*60)
    print("生成测试集预测...")
    start_time = time.time()
    test_predictions = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    
    print(f"测试集预测完成，耗时: {prediction_time:.2f}秒")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': test_predictions
    })
    
    submission_filename = 'submission_neural_network.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\n提交文件 '{submission_filename}' 创建成功!")
    print(f"测试预测分布:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本 ({count / len(test_predictions) * 100:.1f}%)")
    
    print("\n" + "="*60)
    print("任务完成！")

