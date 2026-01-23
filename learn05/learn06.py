import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
# params = [W1, b1, W2, b2]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


loss = nn.CrossEntropyLoss()


net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, num_outputs))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


num_epochs, lr = 10, 0.1
trainer = torch.optim.SGD(net.parameters(), lr=lr)

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
    torch.save(net.state_dict(), 'model_weights.pth')
    print("模型参数已经保存~！")