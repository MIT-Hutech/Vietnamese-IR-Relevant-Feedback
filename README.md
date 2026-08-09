# Vietnamese IR with Relevant Feedback

Repository gồm hai thành phần dùng chung tập dữ liệu tiếng Việt tại
`data/dataset.json`:

- Web truy hồi thông tin và phản hồi liên quan Rocchio, được deploy trên Vercel.
- Công cụ crawler Wikipedia chạy local trong `crawler_tool/`.

## Chạy web IR/Rocchio

```powershell
py -m venv .venv-ir
.\.venv-ir\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ir_app.py
```

Mở `http://localhost:5051`.

## Chạy crawler local

```powershell
py -m venv .venv-crawler
.\.venv-crawler\Scripts\Activate.ps1
python -m pip install -r crawler_tool/requirements.txt
python crawler_tool/app.py
```

Mở `http://localhost:5050`. Crawler đọc và cập nhật trực tiếp file
`data/dataset.json`; sau khi crawl xong, khởi động lại web IR để nạp chỉ mục mới.

Crawler không được route bởi `vercel.json` và chỉ dành để chạy local.
