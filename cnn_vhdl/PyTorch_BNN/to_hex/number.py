from PIL import Image
import numpy as np

def process_image(input_path, output_image_path, output_hex_path):
    # 打开图片并转换为灰度图
    img = Image.open(input_path).convert('L')
    # 调整图片大小为28x28
    img = img.resize((28, 28), Image.LANCZOS)
    # 获取图片的像素值
    pixel_values = np.array(img)
    # 将像素值保存到hex文件
    with open(output_hex_path, 'w') as hex_file:
        address = 0
        for row in pixel_values:
            for pixel in row:
                # 写入Intel HEX格式的数据记录
                data = f'{pixel:02X}'
                # record = f':01{address:04X}00{data}{(1 + (address >> 8) + (address & 0xFF) + pixel) & 0xFF:02X}'
                record = f':01{address:04X}00{data}'
                record = record + calculate_checksum(record)
                hex_file.write(record + '\n')
                address += 1
        # 写入文件结束记录
        hex_file.write(':00000001FF\n')
    # 保存处理后的图片
    img.save(output_image_path)

def calculate_checksum(record):
    # 去掉起始的冒号和最后的校验和字节
    record_data = record[1:]
    # 将记录数据转换为字节数组
    byte_data = bytes.fromhex(record_data)
    # 计算所有字节的总和
    total = sum(byte_data)
    # 取总和的低8位
    low_byte = total & 0xFF
    # 计算补码
    checksum = (512 - low_byte) & 0xFF
    return f'{checksum:02X}'

# 示例用法
input_path = 'DataImages-Test/9917-label-8.png'  # 输入图片路径
output_image_path = 'output_image.png'  # 输出图片路径
output_hex_path = 'number8.hex'  # 输出hex文件路径

process_image(input_path, output_image_path, output_hex_path)