def decorator(func):          # ← 接受被装饰函数
    def wrapper(*args, **kwargs):  # ← 新函数
        # 前置操作（如日志记录、权限校验）
        result = func(*args, **kwargs)  # ← 调用原函数
        # 后置操作（如结果处理、资源清理）
        return result
    return wrapper            # ← 返回增强后的函数

import time
import functools

def timer(func):
    @functools.wraps(func)  # 保持原函数元信息
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"{func.__name__} 执行耗时: {elapsed:.4f}秒")
        return result
    return wrapper

# 使用装饰器
@timer
def calculate_sum(n):
    return sum(range(n+1))


result = calculate_sum(2)
print(result)
# 输出：calculate_sum 执行耗时: 0.0382秒