import re
import json
import os

# 定义要搜索的文件路径
gui_file_path = "gui.py"
# 定义输出的JSON文件路径
i18n_folder = "i18n/locale"

# 创建locale文件夹（如果不存在）
os.makedirs(i18n_folder, exist_ok=True)

# 提取i18n文本的正则表达式
i18n_pattern = r'i18n\(\s*["\'](.*?)["\']\s*\)'

# 读取gui.py文件内容
with open(gui_file_path, "r", encoding="utf-8") as file:
    content = file.read()

# 提取所有i18n文本
matches = re.findall(i18n_pattern, content)

# 去重并排序
i18n_texts = sorted(list(set(matches)))

# 输出提取的文本数量
print(f"Extracted {len(i18n_texts)} unique i18n texts from {gui_file_path}")

# 创建英文基础JSON文件（如果不存在）
en_json_path = os.path.join(i18n_folder, "en_US.json")
if not os.path.exists(en_json_path):
    # 英文JSON就是键值对相同的结构
    en_json_data = {text: text for text in i18n_texts}
    with open(en_json_path, "w", encoding="utf-8") as file:
        json.dump(en_json_data, file, ensure_ascii=False, indent=4)
    print(f"Created English base translation file: {en_json_path}")
else:
    # 如果文件存在，读取现有内容并合并新提取的文本
    with open(en_json_path, "r", encoding="utf-8") as file:
        en_json_data = json.load(file)

    # 添加新的文本条目
    new_entries = 0
    for text in i18n_texts:
        if text not in en_json_data:
            en_json_data[text] = text
            new_entries += 1

    # 保存更新后的文件
    if new_entries > 0:
        with open(en_json_path, "w", encoding="utf-8") as file:
            json.dump(en_json_data, file, ensure_ascii=False, indent=4)
        print(
            f"Updated English translation file: {en_json_path} (added {new_entries} new entries)"
        )
    else:
        print(f"English translation file {en_json_path} is already up to date")

# 创建或更新中文JSON文件
zh_json_path = os.path.join(i18n_folder, "zh_CN.json")
if not os.path.exists(zh_json_path):
    # 创建中文JSON，初始时键值对相同，之后可以手动翻译
    zh_json_data = {text: text for text in i18n_texts}
    with open(zh_json_path, "w", encoding="utf-8") as file:
        json.dump(zh_json_data, file, ensure_ascii=False, indent=4)
    print(f"Created Chinese translation file: {zh_json_path}")
    print("Please manually translate the Chinese JSON file")
else:
    # 如果文件存在，读取现有内容并合并新提取的文本
    with open(zh_json_path, "r", encoding="utf-8") as file:
        zh_json_data = json.load(file)

    # 添加新的文本条目
    new_entries = 0
    for text in i18n_texts:
        if text not in zh_json_data:
            zh_json_data[text] = text
            new_entries += 1

    # 保存更新后的文件
    if new_entries > 0:
        with open(zh_json_path, "w", encoding="utf-8") as file:
            json.dump(zh_json_data, file, ensure_ascii=False, indent=4)
        print(
            f"Updated Chinese translation file: {zh_json_path} (added {new_entries} new entries)"
        )
        print(
            f"Please translate the {new_entries} new entries in the Chinese JSON file"
        )
    else:
        print(f"Chinese translation file {zh_json_path} is already up to date")

# 检查是否有缺失的翻译
if os.path.exists(en_json_path) and os.path.exists(zh_json_path):
    missing_translations = []
    for text in i18n_texts:
        if text not in zh_json_data or zh_json_data[text] == text:
            missing_translations.append(text)

    if missing_translations:
        print(
            f"\nFound {len(missing_translations)} texts that need translation in Chinese JSON:"
        )
        for text in missing_translations[:10]:  # 只显示前10个
            print(f"  - {text}")
        if len(missing_translations) > 10:
            print(f"  ... and {len(missing_translations) - 10} more")
    else:
        print("\nAll texts are properly translated in Chinese JSON")

# 生成差异报告
en_keys = set(en_json_data.keys())
zh_keys = set(zh_json_data.keys())

if en_keys != zh_keys:
    only_in_en = en_keys - zh_keys
    only_in_zh = zh_keys - en_keys

    if only_in_en:
        print(f"\nTexts only in English JSON ({len(only_in_en)}):")
        for text in sorted(list(only_in_en))[:5]:  # 只显示前5个
            print(f"  - {text}")
        if len(only_in_en) > 5:
            print(f"  ... and {len(only_in_en) - 5} more")

    if only_in_zh:
        print(f"\nTexts only in Chinese JSON ({len(only_in_zh)}):")
        for text in sorted(list(only_in_zh))[:5]:  # 只显示前5个
            print(f"  - {text}")
        if len(only_in_zh) > 5:
            print(f"  ... and {len(only_in_zh) - 5} more")
else:
    print("\nBoth JSON files have the same set of texts")
