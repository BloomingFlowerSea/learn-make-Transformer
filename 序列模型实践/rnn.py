import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. 手动实现单层 RNN (不调用 nn.RNN)
# ==========================================
class MyManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyManualRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义权重和偏置 (nn.Parameter 会自动加入反向传播)
        self.w_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev=None):
        # x 形状: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # 存储所有时间步的隐藏状态
        h_states = []

        # 核心逻辑：随时间循环
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步的数据
            # RNN 公式: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
            h_t = torch.tanh(
                torch.matmul(x_t, self.w_ih.t()) + self.b_ih +
                torch.matmul(h_prev, self.w_hh.t()) + self.b_hh
            )
            h_states.append(h_t.unsqueeze(1))
            h_prev = h_t  # 更新隐藏状态，传给下一步

        return torch.cat(h_states, dim=1), h_prev


# ==========================================
# 2. 构建预测模型容器
# ==========================================
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.rnn = MyManualRNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # 映射到未来7天

    def forward(self, x):
        _, last_h = self.rnn(x)
        out = self.fc(last_h)
        return out


# ==========================================
# 3. 数据准备与预处理
# ==========================================
# 这里模拟你的数据结构
np.random.seed(42)
dates = pd.date_range('2004-08-19', periods=200)
# 模拟一段带有趋势和波动的开盘价
mock_open = np.linspace(2.5, 5.0, 200) + np.random.normal(0, 0.1, 200)
df = pd.DataFrame({'Date': dates, 'Open': mock_open})

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Open'].values.reshape(-1, 1))


# 滑动窗口创建数据集
def create_dataset(data, lookback, forecast=7):
    X, y = [], []
    for i in range(len(data) - lookback - forecast):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback: i + lookback + forecast])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).squeeze(-1)


LOOKBACK = 30  # 用过去30天
FORECAST = 7  # 预测未来7天
X_train, y_train = create_dataset(scaled_data, LOOKBACK, FORECAST)

# ==========================================
# 4. 训练阶段
# ==========================================
model = StockPredictor(input_size=1, hidden_size=64, output_size=FORECAST)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

# ==========================================
# 5. 预测与可视化
# ==========================================
model.eval()
# 获取最后30天的数据来预测未来
last_sequence = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
last_sequence = torch.FloatTensor(last_sequence)

with torch.no_grad():
    pred_scaled = model(last_sequence)
    # 逆归一化回真实价格
    pred_real = scaler.inverse_transform(pred_scaled.numpy().reshape(-1, 1))

print("\n预测未来7天的开盘价为：")
print(pred_real.flatten())

# 画图对比
plt.figure(figsize=(10, 5))
plt.plot(df['Open'].values[-50:], label='Historical Open Price')
# 生成未来7天的横坐标
future_indices = np.arange(50, 50 + 7)
plt.plot(future_indices, pred_real, 'r-o', label='Predicted Next 7 Days')
plt.axvline(x=49, color='gray', linestyle='--')
plt.legend()
plt.title("Stock Price Prediction using Manual RNN")
plt.show()