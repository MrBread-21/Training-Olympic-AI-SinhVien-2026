"""Mô hình được cấp sẵn cho bài 25 (không sửa file này). Đầu vào: ảnh RGB float trong [0, 1], shape (B, 3, 64, 64)."""
import torch
import torch.nn as nn


def khoi(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class CNN(nn.Module):
    def __init__(self, n_cls=6):
        super().__init__()
        self.features = nn.Sequential(khoi(3, 32), nn.MaxPool2d(2), khoi(32, 64), nn.MaxPool2d(2), khoi(64, 128))   # (B,128,16,16)
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        f = self.features((x - 0.5) / 0.25)
        return self.head(f.mean((2, 3)))


def nap(path="mo_hinh.pt", dev="cpu"):
    m = CNN()
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.to(dev).eval()
