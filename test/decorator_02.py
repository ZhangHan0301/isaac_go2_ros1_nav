import time
import functools

def time_factory(a):
    def timer(func):
        @functools.wraps(func)  # 保持原函数元信息
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            print("start ",a)
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            print(f"{func.__name__} 执行耗时: {elapsed:.4f}秒")
            return result
        return wrapper
    return timer
# 使用装饰器
@time_factory("s")
def calculate_sum(n):
    return sum(range(n+1))

calculate_sum(1000000)
# 输出：calculate_sum 执行耗时: 0.0382秒