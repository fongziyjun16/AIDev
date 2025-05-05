import requests

# Flask服务器的地址
FLASK_SERVER_URL = 'http://localhost:5000'

# 翻译服务接口
translation_url = f'{FLASK_SERVER_URL}/translation'

# 要上传的文件路径
file_path = 'test.pdf'  # 修改为你的文件路径

# 构建请求参数
params = {
    'source_language': 'English',  # 修改为你的源语言
    'target_language': 'Chinese'   # 修改为你的目标语言
}

# 发送POST请求
with open(file_path, 'rb') as file:
    files = {'input_file': file}
    response = requests.post(translation_url, files=files, data=params)


# 翻译后文件
output_filename = "translated_output.md"

# 处理响应
if response.status_code == 200:
    # 保存翻译后的文件
    with open(output_filename, 'wb') as output_file:
        output_file.write(response.content)
    print(f"Translation completed. Translated file saved as {output_filename}.")
else:
    print(f"Translation failed. Status code: {response.status_code}")
    print(response.text)