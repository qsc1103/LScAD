import torch
import torch.nn as nn


# class Adapter(nn.Module):
#     def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.skip_connect = skip_connect
#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#
#         # Use 1x1 Conv2d for dimensionality reduction (like a linear layer)
#         self.conv_down = nn.Conv2d(D_features, D_hidden_features, kernel_size=1)
#
#         # Convolutional layers with different kernel sizes
#         self.conv1x1 = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=5, padding=2)
#
#         # Use 1x1 Conv2d for dimensionality increase (like a linear layer)
#         self.conv_up = nn.Conv2d(D_hidden_features, D_features, kernel_size=1)
#
#     def forward(self, x):
#         # x is (BT, H, W, D)
#         # Reshape x to (BT, D, H, W) to apply Conv2d
#         x = x.permute(0, 3, 1, 2)  # (BT, D, H, W)
#
#         # Step 1: Dimensionality reduction using 1x1 conv
#         x_down = self.conv_down(x)
#
#         # Step 2: Apply convolutional layers
#         x_11 = self.conv1x1(x_down)
#         x_22 = self.conv3x3(x_down)
#         x_33 = self.conv5x5(x_down)
#
#         # Step 3: Combine the outputs from different convolutional layers
#         x_add = x_11 + x_22 + x_33
#
#         # Step 4: Residual connection and activation
#         x_res = x_down + x_add
#         xs = self.act(x_res)
#
#         # Step 5: Dimensionality increase using 1x1 conv
#         xs = self.conv_up(xs)
#
#         # Step 6: Skip connection (if enabled)
#         if self.skip_connect:
#             x = x + xs
#         else:
#             x = xs
#
#         # Step 7: Reshape back to (BT, H, W, D)
#         x = x.permute(0, 2, 3, 1)  # (BT, H, W, D)
#
#         return x


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x