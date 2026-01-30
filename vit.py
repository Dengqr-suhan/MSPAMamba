import numpy as np

def mask_to_rgb(mask, colormap):
    """
    将单通道 mask 图像转换为 RGB 图像。

    Args:
        mask (np.ndarray): 输入的单通道 mask 图像，每个像素值代表一个类别 ID。
        colormap (dict): 颜色映射字典，键是类别 ID，值是对应的 RGB 颜色元组 (R, G, B)。

    Returns:
        np.ndarray: 转换后的 RGB 图像。
    """
    # 确保 mask 是整数类型
    mask = mask.astype(np.int32)

    # 创建一个空的 RGB 图像，通道数为 3
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # 遍历颜色映射，将每个类别 ID 对应的像素设置为相应的颜色
    for class_id, color in colormap.items():
        rgb_image[mask == class_id] = color

    return rgb_image

# 示例用法：
if __name__ == "__main__":
    # 创建一个示例 mask 图像 (例如，3x3 的图像，有 3 个类别)
    # 类别 0: 背景
    # 类别 1: 物体 A
    # 类别 2: 物体 B
    sample_mask = np.array([
        [0, 1, 1],
        [0, 0, 2],
        [2, 2, 0]
    ], dtype=np.uint8)

    print("原始 mask 图像:")
    print(sample_mask)

    # 定义颜色映射
    # 键是类别 ID，值是 RGB 颜色 (0-255)
    my_colormap = {
        0: (0, 0, 0),       # 类别 0 (背景) -> 黑色
        1: (255, 0, 0),     # 类别 1 (物体 A) -> 红色
        2: (0, 255, 0)      # 类别 2 (物体 B) -> 绿色
    }

    # 转换 mask 到 RGB
    rgb_output = mask_to_rgb(sample_mask, my_colormap)

    print("\n转换后的 RGB 图像:")
    print(rgb_output)

    # 你可以使用 PIL (Pillow) 或 OpenCV 来保存或显示图像
    try:
        from PIL import Image
        img = Image.fromarray(rgb_output, 'RGB')
        img.save("output_rgb_mask.png")
        print("\nRGB 图像已保存为 output_rgb_mask.png")
    except ImportError:
        print("\nPillow 库未安装。请运行 'pip install Pillow' 来保存或显示图像。")

    try:
        import cv2
        cv2.imwrite("output_rgb_mask_cv2.png", cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR))
        print("RGB 图像已保存为 output_rgb_mask_cv2.png (使用 OpenCV)。")
    except ImportError:
        print("OpenCV 库未安装。请运行 'pip install opencv-python' 来保存或显示图像。")