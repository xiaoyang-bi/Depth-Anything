import re
import matplotlib.pyplot as plt
import sys

# 1. 读取log文件
log_file = sys.argv[1]  # 替换为你的log文件名
with open(log_file, 'r', encoding='utf-8') as f:
    log_text = f.read()

# 2. 用正则表达式提取所有Metrics/rmse:后的数值
rmse_list = re.findall(r'Metrics/rmse:\s*([0-9.eE+-]+)', log_text)
rmse_list = [float(x) for x in rmse_list]

# 3. 画图
plt.figure(figsize=(10, 5))
plt.plot(rmse_list, marker='o')
plt.xlabel('Index')
plt.ylabel('Metrics/rmse')
plt.title('Metrics/rmse over time')
plt.grid(True)
plt.tight_layout()

# 4. 保存图片
plt.savefig('metrics_rmse_plot.png')
plt.show()
