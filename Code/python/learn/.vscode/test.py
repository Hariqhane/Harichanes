class MyNumbers:
    # 定义一个迭代器类
    def __iter__(self):
        # 初始化迭代起始值
        self.a = 1
        return self

    # 定义迭代器的下一个值
    def __next__(self):
        x = self.a
        self.a += 1
        return x
 
myclass = MyNumbers()
myiter = iter(myclass)
 
print(next(myiter))  # 打印迭代器的下一个值
print(next(myiter))  # 打印迭代器的下一个值
print(next(myiter))  # 打印迭代器的下一个值
print(next(myiter))  # 打印迭代器的下一个值
print(next(myiter))  # 打印迭代器的下一个值