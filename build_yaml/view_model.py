import openai

# 设置 API 密钥
openai.api_key = "sk-WXtqOuBZPY096KTcDdE866275274464d88943d068aA7Ff5d"

# 可选：设置自定义的 base_url（如果有需要）
openai.base_url = "https://api.gpt.ge/v1/"

# 获取所有可用模型
response = openai.models.list()
# 打印模型名称
for model in response['data']:
    print(model['id'])
