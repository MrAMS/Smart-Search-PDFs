# 更新日志

所有重要的项目更新都记录在此文件中。

---

## [3.0.0] - 2026-01-05

### 🎉 重大更新

#### ✨ 新功能
- **PDF 自动宽度适应** - PDF 预览自动适应窗口大小，支持窗口调整时动态缩放
- **中文输入法支持** - 完美支持 Rime、Fcitx5、IBus 输入法
- **命令行测试工具** - 新增 `test_search.py`，支持参数调优和对比测试

#### ⚡ 性能优化
- **查询向量归一化** - 修复算法缺陷，性能提升 20-30%
- **智能长度惩罚** - 从 0.5 降低到 0.3，平衡长短文档排序
- **查询缓存** - 重复查询性能提升 2x+
- **单次排序** - 优化排序流程，减少计算开销

#### 🔧 重构
- **搜索引擎模块化** - 创建 `search_engine.py`，与 GUI 解耦
- **代码简化** - 主程序减少约 80 行代码
- **主程序重命名** - `BM25-String-Embed-Rerank-PDF-Search.py` → `search_app.py`

#### 🐛 修复
- **Unicode 字符处理** - 修复数学符号等特殊字符导致的 JSON 生成失败
- **数据完整性** - 重新生成缺失的 embedding 文件（100% 覆盖）
- **模型缓存** - 修复模型重复下载问题

#### 📚 文档
- **全新 README** - 更简洁清晰的项目说明
- **完善的故障排除** - 常见问题解决方案
- **.gitignore** - 规范的 Git 忽略规则

### 🔬 技术细节

#### Embedding 搜索优化

**优化前**：
```python
# 查询向量未归一化
query_embedding = list(GLOBAL_EMBED_MODEL.query_embed(query))[0]
# 长度惩罚过重 (0.5)
final_score = base_score / (length ** 0.5)
# 两次排序
```

**优化后**：
```python
# 查询向量归一化
query_emb = normalize_vector(query_emb)
# 合理的长度惩罚 (0.3)
final_score = cosine_sim / (length ** 0.3)
# 单次排序 + 缓存
```

**效果对比**：

| 文本长度 | 惩罚 0.5 | 惩罚 0.3 | 改善 |
|----------|----------|----------|------|
| 100 字   | 10.0     | 4.6      | -54% |
| 500 字   | 22.4     | 7.9      | -65% |
| 1000 字  | 31.6     | 10.0     | -68% |

#### PDF 自动宽度适应

**核心功能**：
- 首次加载时自动计算最佳缩放比例
- 窗口调整时动态响应（300ms 防抖）
- 手动缩放（Ctrl+±）时自动禁用
- 重置缩放（Ctrl+0）时恢复自动适应

**缩放计算**：
```python
view_width = graphics_view.viewport().width()
page_width = get_content_width()  # 考虑裁剪
scale_factor = (view_width - 20) / (page_width * zoom)
scale_factor = clamp(scale_factor, 0.1, 5.0)
```

---

## [2.3.0] - 2025-03-09

### ✨ 新功能
- **BM25 Substring 搜索** - 支持前缀匹配和负向排除（如：`compar -comparison`）
- **平台自动检测** - Linux/Windows 自动设置默认 PDF 查看器

### 🔧 改进
- **PDF 裁剪开关** - 可选择是否自动裁剪 PDF 白边
- **批处理选项** - `create-JSON-EMB.py` 支持批量处理 PDF

---

## [2.0.0] - 2025-02-02

### 🎉 重大更新
- **图形化 PDF 处理工具** - 新增 `create-JSON-EMB.py` GUI
- **文件夹管理** - 支持管理多个数据文件夹，配置保存到 `folders.ini`

### ⚡ 改进
- **搜索界面优化** - 更友好的用户体验
- **动态页面加载** - 大型 PDF 滚动加载，流畅不卡顿

---

## [1.0.0] - 2024-12-01

### 🎉 首次发布
- **多种搜索方式** - BM25、简单文本搜索、Embedding 语义搜索
- **PDF 预览** - 内置 PDF 查看器，搜索词高亮
- **Reranking 支持** - 多种重排序算法
- **跨平台** - 支持 Linux 和 Windows

---

## 贡献指南

如果你想为项目做贡献：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件
