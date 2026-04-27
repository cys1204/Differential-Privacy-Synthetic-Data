import markdown
import os

# 讀取 Markdown 內容
with open('HW1_Report.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 轉換為 HTML
html_content = markdown.markdown(text, extensions=['extra', 'tables', 'toc'])

# 加入漂亮的 CSS 樣式 (GitHub 風格)
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HW1 實驗報告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 45px;
            color: #24292e;
        }}
        h1, h2, h3 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        table, th, td {{ border: 1px solid #dfe2e5; }}
        th, td {{ padding: 10px; text-align: left; }}
        th {{ background-color: #f6f8fa; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
        blockquote {{ padding: 0 1em; color: #6a737d; border-left: 0.25em solid #dfe2e5; margin: 0; }}
        code {{ background-color: rgba(27,31,35,0.05); border-radius: 3px; padding: 0.2em 0.4em; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# 寫入 HTML 檔案
with open('HW1_Report.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("轉換成功！已生成 HW1_Report.html")
