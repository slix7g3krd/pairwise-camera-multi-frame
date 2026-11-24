# pairwise-camera-multi-frame

## 领域

游戏、图形与多媒体

## 主题

双人相机取景与遮挡

## 说明

可见性取景、按帧号遮挡与滚动冻结。

初始状态下 `gameplay/camera_multi_frame.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
