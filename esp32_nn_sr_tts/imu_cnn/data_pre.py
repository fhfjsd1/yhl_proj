import os
import re

def rename_txt_files(directory):
    # 正则表达式匹配文件名格式：XX_123.txt，其中XX_可能是任意字符，123为数字部分
    pattern = re.compile(r'^(.*_)(\d+)(\.txt)$')
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            match = pattern.match(filename)
            if match:
                prefix, number_str, suffix = match.groups()
               
                new_number = int(number_str) +100
                new_filename = f"{prefix}{new_number}{suffix}"
                
                src = os.path.join(directory, filename)
                dst = os.path.join(directory, new_filename)
                
                print(f"重命名: {src} -> {dst}")
                os.rename(src, dst)

if __name__ == "__main__":
    # 指定需要处理的文件夹路径
    folder_path = "TraningData_8_23" # 例如: /path/to/your/txt_files
    if os.path.isdir(folder_path):
        rename_txt_files(folder_path)
        print("重命名完成")
    else:
        print("指定的路径不是文件夹")