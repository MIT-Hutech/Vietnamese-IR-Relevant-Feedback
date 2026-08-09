# Wikipedia dataset crawler

Công cụ Flask thu thập dữ liệu tiếng Việt từ Wikipedia và lưu vào file dùng
chung `../data/dataset.json`.

Chạy từ root repository:

```powershell
py -m venv .venv-crawler
.\.venv-crawler\Scripts\Activate.ps1
python -m pip install -r crawler_tool/requirements.txt
python crawler_tool/app.py
```

Mở `http://localhost:5050`.

Crawler sử dụng background thread và ghi file cục bộ, vì vậy không được thiết kế
để chạy như Vercel Function.
