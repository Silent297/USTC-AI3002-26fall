"""需补全：MLP 的模型构造与前向传播。"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, widths, activation="relu"):
        super().__init__()
        sizes = [input_size, *widths]

        # TODO 1：补全三个属性的构造，保留赋值左侧的名称。
        # hidden：按顺序保存带偏置的隐藏层 Linear，类型为 nn.ModuleList。
        # activation：按参数选择 nn.ReLU() 或 nn.Sigmoid()。
        # output：从最后一个隐藏层映射到 1；无隐藏层时从 input_size 映射到 1。
        hidden_layers = []
        for i in range(len(sizes) - 1):
            in_dim = sizes[i]
            out_dim = sizes[i+1]
            hidden_layers.append(nn.Linear(in_dim, out_dim))
        self.hidden = nn.ModuleList(hidden_layers)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Illegal input of the activation function")
        
        self.output = nn.Linear(sizes[-1], 1)

    def forward(self, x):
        # TODO 2：依次经过隐藏层及激活，最后经过线性输出层。
        # 输入形状为 (B, input_size)，返回形状为 (B, 1)。
        for layer in self.hidden:
            y = layer(x)
            x = self.activation(y)
        return self.output(x)
    
