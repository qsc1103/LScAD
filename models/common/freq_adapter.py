import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class HEAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, num_mlps=4, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)

        # 全连接层 (FC) 处理 Image PE
        self.fc_image = nn.Linear(D_features, D_features)
        self.act = act_layer

        # 定义 Shared MLP
        self.shared_mlp = MLP(input_dim=D_features, hidden_dim=D_hidden_features, output_dim=D_features)

        # 定义多个独立的 MLP 层
        self.individual_mlps = nn.ModuleList(
            [MLP(input_dim=D_features, hidden_dim=D_hidden_features, output_dim=D_features) for _ in range(num_mlps)])

    def forward(self, x_img, x_f):
        """
        输入：
        - x_img: 图像的 token
        - x_f: 频率特征的 token
        """
        # 1. 对图像 token 进行全连接处理
        x_img_fc = self.fc_image(x_img)
        #
        # # 2. 将经过全连接层的 Image PE 和 Frequency PE 相加
        # x = x_img_fc

        x_com = x_img_fc + x_f

        # 3. 通过 Shared MLP 处理
        output = self.fc_image(x_com)

        # 4. 将共享特征传递给多个独立 MLP 层
        #mlp_outputs = [mlp(shared_output) for mlp in self.individual_mlps]

        # 5. 合并输出（这里我们简单返回多个 MLP 的输出，你可以根据需要进一步处理）
        #combined_output = torch.stack(mlp_outputs, dim=0).mean(dim=0)

        return output

