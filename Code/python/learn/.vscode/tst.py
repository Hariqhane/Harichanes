class CountDown:
    def __init__(self, start):
        self.count = start  # 初始化计数器

    def __iter__(self):
        return self  # 返回迭代器本身

    def __next__(self):
        if self.count <= 0:
            raise StopIteration  # 当计数器到达0或以下时，抛出StopIteration
        else:
            current = self.count  # 当前数值为计数器值
            self.count -= 1  # 计数器减1
            print("test")
            return current  # 返回当前数值

# 使用CountDown迭代器
counter = CountDown(10)  # 创建CountDown实例，初始值为10
print(type(8))
print(type(counter))  # 打印counter的类型
# for num in counter:  # 遍历迭代器
#     print(num)  # 输出 3, 2, 1
import sys
