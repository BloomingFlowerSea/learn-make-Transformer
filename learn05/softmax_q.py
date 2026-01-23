import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)



net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10


if __name__ == '__main__':
    for epoch in range(num_epochs):
        net.train()
        tall_l = 0.0
        tall_num = 0

        true_num = 0.0
        test_num = 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

            with torch.no_grad():

                tall_l += float(l) * len(y)
                tall_num += len(y)

        net.eval()
        for X, y in test_iter:
            test_num += len(y)
            y_hat = net(X)
            y_pred = y_hat.argmax(dim=1)
            correct_count = (y_pred == y).type(torch.float).sum().item()
            true_num += correct_count
        print(f'epoch {epoch + 1}, loss {tall_l / tall_num : f}, acc {true_num / test_num : f}')
