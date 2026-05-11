import importlib
import traceback


with open("DocTer_api.txt", "r", encoding="utf-8") as f:
    apis = [line.strip() for line in f if line.strip()]

available = []
unavailable = []

for api in apis:
    try:
        # 拆分模块和属性
        parts = api.split(".")
        module_path = ".".join(parts[:-1])
        attr_name = parts[-1]

        # 导入模块
        module = importlib.import_module(module_path)

        # 检查属性是否存在
        if hasattr(module, attr_name):
            # 如果存在，尝试访问/调用
            obj = getattr(module, attr_name)

            try:
                # 试调用（如果是可调用）
                if callable(obj):
                    obj()
                available.append(api)
            except Exception as call_err:
                # 这个 API 存在，但调用有错误（属于可用）
                available.append(api)
        else:
            unavailable.append(api)

    except Exception as e:
        # 无法导入或其他错误
        unavailable.append(api)

# 输出结果
print(f"可用的 API ({len(available)}):")
for a in available:
    print("  ", a)

print(f"\n不可用的 API ({len(unavailable)}):")
for u in unavailable:
    print("  ", u)
