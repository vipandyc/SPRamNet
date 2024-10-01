import os

# 定义原始文件名和目标文件名的范围
start_original = 501
end_original = 1000
start_target = 464
end_target = 963

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

os.chdir(current_dir)

# get the changed working directory
updated_dir = os.getcwd()
print("Updated working directory:", updated_dir)

# 确保原始范围和目标范围的长度一致
assert (end_original - start_original) == (end_target - start_target), "范围长度不一致"

# 遍历原始文件名并生成新的文件名
for i in range(end_original - start_original + 1):
    original_filename = f"rdf_{start_original + i}.npy"
    target_filename = f"rdf_{start_target + i}.npy"
    
    # 检查原始文件是否存在
    if os.path.exists(original_filename):
        # 重命名文件
        os.rename(original_filename, target_filename)
        print(f"Renamed {original_filename} to {target_filename}")
    else:
        print(f"File {original_filename} does not exist")