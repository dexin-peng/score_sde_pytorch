import torch
import torch.nn.functional as F


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    """
    纯 Python 实现 upfirdn2d 操作：
      1. 上采样：将输入扩展到 (B, C, H*up_y, W*up_x)（在新位置填 0）。
      2. 填充：利用 F.pad 填充（pad_x0 为左侧、pad_x1 为右侧，pad_y0 为上侧、pad_y1 为下侧），
         若为负则做裁剪。
      3. 卷积滤波：先将 kernel 翻转后，再用 conv2d 进行卷积，此处设置 stride = (down_y, down_x) 内嵌下采样，
         且 groups = C 保证各通道独立。
    最终输出尺寸应为  
      out_h = ((H*up_y + pad_y0 + pad_y1 - kernel_h) // down_y) + 1  
      out_w = ((W*up_x + pad_x0 + pad_x1 - kernel_w) // down_x) + 1
    """
    B, C, H, W = input.shape
    kernel_h, kernel_w = kernel.shape

    # 1. 上采样：构造大小为 (B, C, H*up_y, W*up_x) 的张量，并将原始值复制到对应位置
    out_h = H * up_y
    out_w = W * up_x
    upsampled = torch.zeros(B, C, out_h, out_w, device=input.device, dtype=input.dtype)
    upsampled[:, :, ::up_y, ::up_x] = input

    # 2. 填充：F.pad 的顺序为 (左, 右, 上, 下)
    padded = F.pad(upsampled, (pad_x0, pad_x1, pad_y0, pad_y1))
    if pad_x0 < 0 or pad_x1 < 0 or pad_y0 < 0 or pad_y1 < 0:
        start_x = -min(pad_x0, 0)
        end_x = padded.shape[3] + min(pad_x1, 0)
        start_y = -min(pad_y0, 0)
        end_y = padded.shape[2] + min(pad_y1, 0)
        padded = padded[:, :, start_y:end_y, start_x:end_x]

    # 3. 卷积滤波：先翻转 kernel，再扩展到各通道（groups=C），同时采用 stride 进行下采样
    w = torch.flip(kernel, dims=[0, 1]).view(1, 1, kernel_h, kernel_w)
    w = w.repeat(C, 1, 1, 1)
    conv_out = F.conv2d(padded, w, stride=(down_y, down_x), padding=0, groups=C)
    return conv_out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    参数说明：
      - input: (B, C, H, W)
      - kernel: 滤波核，形状 (kH, kW)
      - up: 上采样因子（同时用于 x 和 y 方向）
      - down: 下采样因子（同时用于 x 和 y 方向）
      - pad: 形式为 (pad_left/上, pad_right/下)
             注意：此处假定左右填充相同、上下填充相同，如有需要可单独调整
    """
    # 注意调用时将 pad[0] 分别作为左/上填充，pad[1] 作为右/下填充
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])