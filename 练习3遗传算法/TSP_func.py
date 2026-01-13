import numpy as np
import time
import random
import matplotlib.pyplot as plt
import re

plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 使用微软雅黑
plt.rcParams["axes.unicode_minus"] = False         # 解决负号显示问题


def load_data_re(file_path):
    """
    用正则表达式
    从txt文件中读取以分号分隔的城市x, y坐标, 返回城市坐标列表

    :param file_path: txt文件路径
    :return city_coords: 城市坐标列表 [(x1, y1), (x2, y2), ...]
    """
    with open(file_path, 'r') as f:
        data = f.read()

    pattern = re.compile(r'(\d+)\s+(\d+)')   # x y
    matches = pattern.findall(data)

    city_coords = []
    for city in matches:                
        x = city[0]   
        y = city[1]
        city_coords.append((float(x), float(y)))
    
    return city_coords

def cal_dist_matrix(city_coords):
    """ 
    根据城市坐标, 计算并返回以距离为权重的邻接矩阵

    :param city_coords: 城市坐标列表 [(x1, y1), (x2, y2), ...]
    :return dist_matrix: 距离邻接矩阵
    """
    n = len(city_coords)                       # 城市数量
    dist_matrix = np.zeros((n, n))             # 初始化距离矩阵
    for i in range(n):                         # 对每个城市
        x1, y1 = city_coords[i]
        for j in range(n):                     # 计算与其他城市的距离
            if i != j:
                x2, y2 = city_coords[j]
                dist_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist_matrix

# 需要的函数      
def cal_distance(dist_matrix, path):
    """
    计算TSP可行路径的总距离
    
    :param dist_matrix: 城市邻接矩阵
    :param path: 可行路径
    :return distance: 路径总距离
    """
    distance = 0
    for i in range(len(dist_matrix)):
        distance += dist_matrix[path[i]][path[i+1]]
    return distance

def plot_tsp_path(city_coords, path, title='TSP路径'):
    """
    可视化TSP路径
    
    :param city_coords: 城市坐标列表，每个元素为(x, y)元组
    :param path: 路径列表，包含城市索引（如[0, 3, 1, ..., 0]表示闭环路径）
    :param title: 绘图的标题
    """
    # 提取路径中所有城市的坐标
    x = [city_coords[i][0] for i in path]
    y = [city_coords[i][1] for i in path]
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制路径（连接线）
    plt.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, label='路径')
    
    # 绘制城市点（普通城市）
    plt.scatter(x, y, c='skyblue', s=80, edgecolors='black', linewidths=0.8, label='城市')
    
    
    # 添加标签和标题
    plt.title(title, fontsize=14)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 显示图表
    plt.show()

def plot_distance_record(distance_record, title='每轮迭代的路径距离'):
    """ 
    可视化每轮迭代的路径距离

    :param distance_record: list, 记录每轮迭代的路径距离
    :param title: 绘图的标题
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(distance_record)

    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('路径距离')
    plt.show()

# 构造新解的方法
def change_two_cities(path):
    """  
    随机交换两个城市位置生成新路径

    :param path: 可行路径
    :return new_path: 新可行路径
    """
    new_path = path.copy()                           # 复制，不修改原路径
    i, j = random.sample(range(1, len(path)-1), 2)   # 从列表中随机抽取不重复元素, path头尾固定为城市0，不用交换
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def reverse_subpath(path):
    """
    将路径切分为两条子路径, 其中一条翻转后再次拼接, 生成新路径
    
    :param path: 可行路径
    :return new_path: 新可行路径
    """
    i, j = random.sample(range(1, len(path)-1), 2)   # [1, 29] 随机抽取不重复元素
    if i > j:         
        i, j = j, i  # 同时交换

    new_path = path.copy()
    subpath = new_path[i:j+1]
    new_path = new_path[:i] + subpath[::-1] + new_path[j+1:]     
    return new_path

def select_insert_one_city(path):
    """ 
    随机选取某个城市, 并将其插入到另一位置

    :param path: 可行路径
    :return new_path: 新可行路径
    """
    i, j = random.sample(range(1, len(path)-1), 2)   # [1, 29] 随机抽取不重复元素

    new_path = path.copy()
    city = new_path.pop(i)
    new_path.insert(j, city)
    return new_path