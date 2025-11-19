# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/10 17:40

# 将 .env 文件中的环境变量加载到 os.environ 中
import os
from pathlib import Path

env_file = Path(__file__).parent / ".env"

if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(env_file))
    except ImportError:
        # 手动解析 .env
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
