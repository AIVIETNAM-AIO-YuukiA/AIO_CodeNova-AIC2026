# AIO_CodeNova-AIC2026

## Yêu cầu

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) — quản lý môi trường và package

---

## Cài đặt lần đầu

```bash
# 1. Clone repo
git clone <repo-url>
cd AIO_CodeNova-AIC2026

# 2. Cài uv (nếu chưa có)
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Tạo môi trường và cài dependencies
uv sync

# 4. Cài pre-commit hooks
uv run pre-commit install
```

---

## Quản lý dependencies với uv

```bash
# Thêm package mới
uv add <package-name>

# Thêm package chỉ dùng khi dev (lint, test,...)
uv add --dev <package-name>

# Xóa package
uv remove <package-name>

# Đồng bộ môi trường theo uv.lock (dùng khi pull code mới về)
uv sync

# Chạy script/lệnh trong môi trường
uv run python main.py
```

> Sau khi `uv add` / `uv remove`, nhớ commit cả `pyproject.toml` và `uv.lock`.

---

## Git Workflow

### Quy tắc đặt tên branch

```
feature/<tên-tính-năng>     # thêm tính năng mới
fix/<tên-lỗi>               # sửa bug
chore/<công-việc>           # cập nhật config, docs,...
```

Ví dụ: `feature/data-preprocessing`, `fix/model-output-error`

### Các bước làm việc

```bash
# 1. Luôn cập nhật branch main trước
git checkout main
git pull origin main

# 2. Tạo branch mới từ main
git checkout -b feature/<tên-tính-năng>

# 3. Làm việc, sau đó commit
git add .
git commit -m "feat: mô tả ngắn thay đổi"

# 4. Push branch lên remote
git push origin feature/<tên-tính-năng>

# 5. Tạo Pull Request trên GitHub để merge vào main
```

### Quy tắc commit message

| Prefix | Dùng khi |
|--------|----------|
| `feat:` | Thêm tính năng mới |
| `fix:` | Sửa bug |
| `docs:` | Cập nhật tài liệu |
| `chore:` | Thay đổi config, dependencies |
| `refactor:` | Refactor code |

---

## Makefile shortcuts

```bash
make install      # uv sync — cài/đồng bộ dependencies
make lint         # kiểm tra lỗi code
make format       # tự động format code
make pre-commit   # chạy tất cả pre-commit hooks
make test         # chạy tests
```
