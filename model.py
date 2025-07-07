import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS)
    Args:
        xyz: point cloud data, [B, N, 3]
        npoint: number of sample points
    Returns:
        centroids: sampled point indices, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class MultiScaleSetAbstraction(nn.Module):
    """Set abstraction layer supporting multi-scale feature extraction (using FPS)"""

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.branches = nn.ModuleList()
        for radius, nsample, mlp in zip(radius_list, nsample_list, mlp_list):
            self.branches.append(
                SingleScaleSA(radius, nsample, in_channel, mlp)
            )
        self.fusion_conv = nn.Conv1d(sum([mlp[-1] for mlp in mlp_list]), sum([mlp[-1] for mlp in mlp_list]), 1)
        self.fusion_bn = nn.BatchNorm1d(sum([mlp[-1] for mlp in mlp_list]))

    def forward(self, xyz, points):
        B, N, _ = xyz.shape

        if self.npoint is not None:
            centroids = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, centroids)
        else:
            new_xyz = xyz.mean(1, keepdim=True)

        branch_features = []
        for branch in self.branches:
            _, feat = branch(xyz, points, new_xyz)
            branch_features.append(feat)

        combined_feat = torch.cat(branch_features, dim=1)
        fused_feat = F.relu(self.fusion_bn(self.fusion_conv(combined_feat)))
        return new_xyz, fused_feat


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.sa1 = MultiScaleSetAbstraction(
            npoint=4096, radius_list=[0.05, 0.1, 0.2], nsample_list=[16, 32, 64], in_channel=0,
            mlp_list=[[16, 16, 32], [32, 32, 64], [64, 64, 128]]
        )
        self.cas = CAS(channels=32 + 64 + 128)
        self.sa2 = MultiScaleSetAbstraction(
            npoint=1024, radius_list=[0.2, 0.4, 0.6], nsample_list=[32, 64, 128], in_channel=32 + 64 + 128,
            mlp_list=[[32, 64], [64, 128], [128, 256]]
        )
        self.sa3 = MultiScaleSetAbstraction(
            npoint=None, radius_list=[0.4, 0.6, 0.8], nsample_list=[64, 128, 256], in_channel=64 + 128 + 256,
            mlp_list=[[64, 128], [128, 256], [256, 512]]
        )
        self.fc = nn.Sequential(
            nn.Linear((128 + 256 + 512) * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.attention_weights_sa1 = None
        self.sa1_xyz = None

    def forward(self, xyz):
        B, N, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        attentive_feat_sa1 = self.cas(l1_points)
        self.sa1_xyz = l1_xyz

        l1_points_att = attentive_feat_sa1.permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_att)
        l2_points = l2_points.permute(0, 2, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        max_pool = F.adaptive_max_pool1d(l3_points, 1).view(B, -1)
        avg_pool = F.adaptive_avg_pool1d(l3_points, 1).view(B, -1)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        return self.fc(combined)


class SingleScaleSA(nn.Module):
    """Single-scale feature extraction module"""

    def __init__(self, radius, nsample, in_channel, mlp):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points, new_xyz):
        B, S, _ = new_xyz.shape

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.view(B, S, 1, 3)

        if points is not None:
            grouped_points = index_points(points, idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]
        return new_xyz, new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query implementation"""
    device = xyz.device
    B, S, _ = new_xyz.shape
    N = xyz.shape[1]

    dist = torch.cdist(new_xyz, xyz)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    mask = dist > radius
    group_idx[mask] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    invalid_mask = group_idx == N
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    group_idx[invalid_mask] = group_first[invalid_mask]

    return group_idx


def index_points(points, idx):
    """Index point data"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class CAS(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = self.sigmoid(avg_out + max_out)
        return out.unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return attn


if __name__ == "__main__":
    model = Net(num_classes=10)
    xyz = torch.rand(4, 2048, 3)  # Assuming input has 4 samples, each with 2048 points
    out = model(xyz)
    print("Output shape:", out.shape)  # Expected output is [4, 10]
