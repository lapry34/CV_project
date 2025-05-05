import numpy as np
import sys

if __name__ == '__main__':
    print('test')
    # 1. 定义一个二维数组，求数组中所有元素的和
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(array)
    print(array.sum())
    # 2. 定义一个二维数组，求数组中所有元素的平均值
    print(array.mean())
    # 3. 定义一个二维数组，求数组中所有元素的最大值
    print(array.max())


    sys.exit(0)

