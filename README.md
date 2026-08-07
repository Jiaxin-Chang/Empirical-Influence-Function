# 终端 A：annotation 后端
cd tools\annotation-viewer
python -m server.main

# 终端 B：annotation 前端
cd tools\annotation-viewer
pnpm dev

# 终端 C：correlation-report 后端
cd ..
$env:EIF_ADAPTER_PATH = "..."
$env:EIF_BASE_MODEL_PATH = "..."
python -m src.ttav_bundle_api --host 0.0.0.0 --port 8766

# 终端 D：correlation-report前端
cd tools/correlation-report
pnpm dev