#!/usr/bin/env python3
import os
import sys
import json
import re
import numpy as np

from PyQt5 import QtCore, QtWidgets

# Global flag for batch processing; default is disabled.
BATCH_PROCESSING_ENABLED = False

# ---------------------------
# Utility functions
# ---------------------------
def clean_text_for_json(text):
    """
    清理文本中的无效 Unicode 字符（如 surrogate pairs）
    这些字符无法被 UTF-8 编码，会导致 JSON 保存失败
    """
    if not text:
        return text

    # 移除 surrogate pairs（0xD800-0xDFFF 范围的字符）
    # 这些通常来自 PDF 中的特殊数学符号
    cleaned = ""
    for char in text:
        code = ord(char)
        # 跳过 surrogate pairs 范围的字符
        if 0xD800 <= code <= 0xDFFF:
            # 可以选择替换为空格或特殊标记
            cleaned += " "  # 或使用 "[MATH]" 等标记
        else:
            cleaned += char

    return cleaned


def normalize_embedding(embedding):
    """
    归一化 embedding 向量到单位长度
    这确保余弦相似度计算的准确性
    """
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    else:
        return embedding

# ---------------------------
# External library functions
# ---------------------------
import pymupdf4llm

# Check if fastembed library is installed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None

##########################################
# Functions for PDF-to-JSON processing
##########################################

def extract_page_chunks(file_path, log_callback=None):
    """
    Extracts page-length chunks from a PDF file using PyMuPDF4LLM.
    Returns a list of dicts (one per page).
    """
    chunks = []
    try:
        data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        for page in data:
            # 清理文本中的无效 Unicode 字符
            raw_text = page["text"]
            cleaned_text = clean_text_for_json(raw_text)

            chunks.append({
                "text": cleaned_text,
                "page_number": page.get("metadata", {}).get("page", None),
                "filename": os.path.basename(file_path),
                "chunk_type": "page",           # 新增：标识chunk类型
                "chunk_index": len(chunks)       # 新增：chunk序号
            })
    except Exception as e:
        if log_callback:
            log_callback(f"Error extracting chunks from {file_path}: {e}")
        else:
            print(f"Error extracting chunks from {file_path}: {e}")
    return chunks


def extract_paragraph_chunks(file_path, log_callback=None, min_paragraph_length=100):
    """
    按段落提取chunks，适合双栏论文等结构化文档

    Args:
        file_path: PDF文件路径
        log_callback: 日志回调函数
        min_paragraph_length: 最小段落长度（字符数），短段落会合并

    Returns:
        List[Dict]: chunk列表
    """
    chunks = []
    try:
        data = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        chunk_index = 0

        for page in data:
            page_num = page.get("metadata", {}).get("page", None)
            cleaned_text = clean_text_for_json(page["text"])

            # 按双换行符分段（Markdown段落分隔符）
            paragraphs = re.split(r'\n\n+', cleaned_text)

            current_buffer = []
            buffer_length = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_length = len(para)

                # 合并短段落
                if para_length < min_paragraph_length:
                    current_buffer.append(para)
                    buffer_length += para_length

                    # 缓冲区达到最小长度，输出
                    if buffer_length >= min_paragraph_length:
                        chunks.append({
                            "text": "\n\n".join(current_buffer),
                            "page_number": page_num,
                            "filename": os.path.basename(file_path),
                            "chunk_type": "paragraph",
                            "chunk_index": chunk_index
                        })
                        chunk_index += 1
                        current_buffer = []
                        buffer_length = 0
                else:
                    # 先输出缓冲区
                    if current_buffer:
                        chunks.append({
                            "text": "\n\n".join(current_buffer),
                            "page_number": page_num,
                            "filename": os.path.basename(file_path),
                            "chunk_type": "paragraph",
                            "chunk_index": chunk_index
                        })
                        chunk_index += 1
                        current_buffer = []
                        buffer_length = 0

                    # 输出当前段落
                    chunks.append({
                        "text": para,
                        "page_number": page_num,
                        "filename": os.path.basename(file_path),
                        "chunk_type": "paragraph",
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1

            # 处理页面末尾的缓冲区
            if current_buffer:
                chunks.append({
                    "text": "\n\n".join(current_buffer),
                    "page_number": page_num,
                    "filename": os.path.basename(file_path),
                    "chunk_type": "paragraph",
                    "chunk_index": chunk_index
                })
                chunk_index += 1

    except Exception as e:
        if log_callback:
            log_callback(f"Error extracting paragraph chunks from {file_path}: {e}")
        else:
            print(f"Error extracting paragraph chunks from {file_path}: {e}")
    return chunks


def extract_fixed_chunks(file_path, log_callback=None, chunk_size=2000, overlap=200):
    """
    按固定长度切分chunks，支持滑动窗口重叠

    Args:
        file_path: PDF文件路径
        log_callback: 日志回调函数
        chunk_size: 每个chunk的字符数
        overlap: 相邻chunk的重叠字符数

    Returns:
        List[Dict]: chunk列表
    """
    chunks = []
    try:
        data = pymupdf4llm.to_markdown(file_path, page_chunks=True)

        # 构建全文本和页码映射
        full_text = ""
        char_to_page = []

        for page in data:
            page_num = page.get("metadata", {}).get("page", None)
            page_text = clean_text_for_json(page["text"])
            char_to_page.extend([page_num] * len(page_text))
            full_text += page_text

        # 滑动窗口切分
        chunk_index = 0
        start = 0

        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()

            if chunk_text:
                # 取chunk中点位置的页码
                mid_pos = start + len(chunk_text) // 2
                page_num = char_to_page[mid_pos] if mid_pos < len(char_to_page) else char_to_page[-1]

                chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "filename": os.path.basename(file_path),
                    "chunk_type": "fixed",
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end
                })
                chunk_index += 1

            start += chunk_size - overlap

    except Exception as e:
        if log_callback:
            log_callback(f"Error extracting fixed chunks from {file_path}: {e}")
        else:
            print(f"Error extracting fixed chunks from {file_path}: {e}")
    return chunks


def get_chunk_filename(base_name, granularity, extension):
    """
    生成带粒度标识的文件名

    Args:
        base_name: 基础文件名（不含扩展名）
        granularity: 粒度类型 ("page", "paragraph", "fixed")
        extension: 文件扩展名（如 ".json", ".emb"）

    Returns:
        str: 完整文件名

    Examples:
        ("doc", "page", ".json") -> "doc.json"
        ("doc", "paragraph", ".json") -> "doc.para.json"
        ("doc", "fixed", ".emb") -> "doc.fixed.emb"
    """
    suffix_map = {
        "page": "",           # 保持兼容，page不加后缀
        "paragraph": ".para",
        "fixed": ".fixed"
    }

    suffix = suffix_map.get(granularity, "")
    return f"{base_name}{suffix}{extension}"


def process_pdf_to_json(folder, granularity, log_callback):
    """
    Processes all PDF files in the given folder.
    For each PDF file that does not yet have a corresponding JSON file,
    extracts chunks (based on granularity) and saves them as JSON.

    Args:
        folder: PDF文件夹路径
        granularity: 粒度类型 ("page", "paragraph", "fixed")
        log_callback: 日志回调函数
    """
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    if total_files == 0:
        log_callback(f"No PDF files found in {folder}")
        return

    log_callback(f"\n===== Chunk粒度设置 =====")
    log_callback(f"选择粒度: {granularity}")
    if granularity == "paragraph":
        log_callback(f"  - 最小段落长度: 100 字符")
        log_callback(f"  - 合并短段落: 是")
    elif granularity == "fixed":
        log_callback(f"  - Chunk大小: 2000 字符")
        log_callback(f"  - 重叠区域: 200 字符")
    log_callback(f"========================\n")

    for idx, file_name in enumerate(pdf_files):
        log_callback(f"Processing PDF file {idx + 1} of {total_files}: {file_name}")
        pdf_file_path = os.path.join(folder, file_name)

        # 使用新的文件命名函数
        base_name = os.path.splitext(file_name)[0]
        json_file_name = get_chunk_filename(base_name, granularity, ".json")
        json_file_path = os.path.join(folder, json_file_name)

        if os.path.exists(json_file_path):
            log_callback(f"Skipping {file_name} – {json_file_name} already exists.")
            continue

        try:
            # 根据粒度调用不同的提取函数
            if granularity == "page":
                chunks = extract_page_chunks(pdf_file_path, log_callback)
            elif granularity == "paragraph":
                chunks = extract_paragraph_chunks(pdf_file_path, log_callback)
            elif granularity == "fixed":
                chunks = extract_fixed_chunks(pdf_file_path, log_callback)
            else:
                log_callback(f"Unknown granularity: {granularity}, using page-level")
                chunks = extract_page_chunks(pdf_file_path, log_callback)

            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(chunks, json_file, ensure_ascii=False, indent=2)
            log_callback(f"Saved {len(chunks)} chunks to {json_file_name}")
        except Exception as e:
            log_callback(f"Error processing {file_name}: {e}")

##########################################
# Functions for JSON-to-EMB processing
##########################################

def embed_pages_in_json(json_file_path, embedding_model, log_callback):
    """
    Reads a JSON file containing text chunks (pages),
    generates embeddings for each chunk (using the given embedding_model) one page at a time,
    normalizes embeddings, removes the text field, and returns the updated list.
    """
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        pages = json.load(json_file)

    total_pages = len(pages)
    for i, page in enumerate(pages):
        log_callback(f"Embedding page {i + 1} of {total_pages} in {os.path.basename(json_file_path)}")
        if "text" in page:
            try:
                embedding_gen = embedding_model.passage_embed([page["text"]])
                embedding = list(embedding_gen)[0]

                # 归一化 embedding
                embedding = normalize_embedding(embedding)

                if isinstance(embedding, np.ndarray):
                    page["embedding"] = embedding.tolist()
                else:
                    page["embedding"] = embedding
            except Exception as e:
                log_callback(f"Error embedding page {i + 1} in {json_file_path}: {e}")
            del page["text"]
    return pages

def embed_pages_in_json_batch(json_file_path, embedding_model, log_callback):
    """
    Batch embeds all pages from the JSON file at once.
    Normalizes embeddings for better quality.
    If an error occurs, falls back to page-by-page processing.
    """
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        pages = json.load(json_file)

    total_pages = len(pages)
    texts = []
    for i, page in enumerate(pages):
        texts.append(page.get("text", ""))
    try:
        log_callback(f"Batch embedding {total_pages} pages from {os.path.basename(json_file_path)}")
        embedding_gen = embedding_model.passage_embed(texts)
        embeddings = list(embedding_gen)
        for i, embedding in enumerate(embeddings):
            # 归一化 embedding
            embedding = normalize_embedding(embedding)

            if isinstance(embedding, np.ndarray):
                pages[i]["embedding"] = embedding.tolist()
            else:
                pages[i]["embedding"] = embedding
            if "text" in pages[i]:
                del pages[i]["text"]
    except Exception as e:
        log_callback(f"Batch embedding error for {os.path.basename(json_file_path)}: {e}")
        log_callback("Falling back to page-by-page embedding for this file.")
        pages = embed_pages_in_json(json_file_path, embedding_model, log_callback)
    return pages

def process_json_to_emb(folder, log_callback):
    """
    Processes all JSON files in the folder.
    For each JSON file that does not have a corresponding .emb file,
    generates embeddings and saves the result as a .emb file.
    Uses jinaai/jina-embeddings-v2-base-zh for Chinese-English mixed text support.
    """
    if not FASTEMBED_AVAILABLE:
        log_callback("Fastembed library not installed, EMB files creation disabled.")
        return

    try:
        # 设置模型缓存目录（使用用户主目录下的 .cache/fastembed）
        cache_dir = os.path.expanduser("~/.cache/fastembed")
        model_name = "jinaai/jina-embeddings-v2-base-zh"

        # 检查模型是否已缓存
        model_cache_path = os.path.join(cache_dir, model_name.replace("/", "--"))
        if os.path.exists(model_cache_path):
            log_callback(f"✅ 检测到缓存的模型: {model_name}")
            log_callback(f"   缓存路径: {model_cache_path}")
        else:
            log_callback(f"⏬ 首次使用，需要下载模型: {model_name}")
            log_callback(f"   将缓存到: {cache_dir}")
            log_callback("   这可能需要几分钟，请耐心等待...")

        # 使用 jinaai/jina-embeddings-v2-base-zh - 专门支持中英文混合
        # 特点:
        # - 768 维度 embedding
        # - 8192 token 长文本支持（适合 PDF 页面）
        # - 2024 年发布，较新
        # - 不需要特殊前缀
        #
        # 备选模型：
        # - "jinaai/jina-embeddings-v3" - 多语言（~100种），1024维，8192 token
        # - "BAAI/bge-small-zh-v1.5" - 中文专用，512维，512 token
        # - "intfloat/multilingual-e5-large" - 多语言（~100种），1024维，512 token
        log_callback("初始化 embedding 模型: jinaai/jina-embeddings-v2-base-zh (中英文混合)")
        embedding_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        log_callback("✅ 模型加载成功 (768维, 支持8192 token)")
        log_callback(f"   模型将被复用，下次运行无需重新下载")
    except Exception as e:
        log_callback(f"Error initializing embedding model: {e}")
        log_callback("提示：如果模型下载失败，请检查网络连接或尝试使用镜像")
        return

    json_files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    total_files = len(json_files)
    if total_files == 0:
        log_callback(f"No JSON files found in {folder}")
        return

    for idx, file_name in enumerate(json_files):
        emb_file_name = os.path.splitext(file_name)[0] + ".emb"
        emb_file_path = os.path.join(folder, emb_file_name)
        json_file_path = os.path.join(folder, file_name)
        log_callback(f"Processing JSON file {idx + 1} of {total_files}: {file_name}")
        if os.path.exists(emb_file_path):
            log_callback(f"Skipping {file_name} – EMB already exists.")
            continue
        try:
            if BATCH_PROCESSING_ENABLED:
                embedded_pages = embed_pages_in_json_batch(json_file_path, embedding_model, log_callback)
            else:
                embedded_pages = embed_pages_in_json(json_file_path, embedding_model, log_callback)
            with open(emb_file_path, "w", encoding="utf-8") as emb_file:
                json.dump(embedded_pages, emb_file, ensure_ascii=False, indent=2)
            log_callback(f"Saved EMB to {emb_file_name}")
        except Exception as e:
            log_callback(f"Error processing {file_name} for EMB: {e}")

##########################################
# Combined processing per folder (Ordered)
##########################################

def process_folder(folder, process_json, process_emb, granularity, log_callback):
    """
    Processes a single folder in a strict order.
    If process_json is True, run PDF-to-JSON extraction.
    Then, if process_emb is True, run JSON-to-EMB creation.
    This order ensures that EMB processing is done only after JSON files are generated.

    Args:
        folder: 文件夹路径
        process_json: 是否处理JSON
        process_emb: 是否处理EMB
        granularity: 粒度类型 ("page", "paragraph", "fixed")
        log_callback: 日志回调
    """
    log_callback(f"--- Starting processing for folder: {folder} ---")
    if process_json:
        log_callback(">>> Starting PDF-to-JSON extraction...")
        process_pdf_to_json(folder, granularity, log_callback)  # 传递granularity
    else:
        log_callback(">>> Skipping PDF-to-JSON extraction (not selected).")

    if process_emb:
        log_callback(">>> Starting JSON-to-EMB creation (after JSON extraction)...")
        process_json_to_emb(folder, log_callback)
    else:
        log_callback(">>> Skipping JSON-to-EMB creation (not selected).")

    log_callback(f"--- Finished processing folder: {folder} ---\n")

##########################################
# Worker for Background Processing (PyQt5)
##########################################

class Worker(QtCore.QObject):
    logSignal = QtCore.pyqtSignal(str)
    progressSignal = QtCore.pyqtSignal(int)
    finishedSignal = QtCore.pyqtSignal()

    def __init__(self, queue_items, parent=None):
        super().__init__(parent)
        self.queue_items = queue_items

    @QtCore.pyqtSlot()
    def run(self):
        total = len(self.queue_items)
        for i, item in enumerate(self.queue_items):
            folder = item["folder"]
            process_json = item["process_json"]
            process_emb = item["process_emb"]
            granularity = item.get("granularity", "page")  # 新增：获取粒度，默认page

            self.logSignal.emit(f"\n=== Processing folder {i + 1} of {total}: {folder} ===")
            process_folder(folder, process_json, process_emb, granularity, self.logSignal.emit)

            progress_percent = int(((i + 1) / total) * 100)
            self.progressSignal.emit(progress_percent)
        self.logSignal.emit("\nAll processing complete.")
        self.finishedSignal.emit()

##########################################
# Main Application Window (PyQt5)
##########################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 处理工具 - 智能文档索引生成器")
        self.resize(900, 700)
        self.setup_ui()
        self.worker_thread = None

        # If fastembed is not available, display a message in the log
        if not FASTEMBED_AVAILABLE:
            self.append_log("警告：FastEmbed 库未安装，EMB 文件生成功能已禁用")

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- 标题栏 ---
        title_layout = QtWidgets.QVBoxLayout()
        title_label = QtWidgets.QLabel("PDF 文档处理工具")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        subtitle_label = QtWidgets.QLabel("将 PDF 文档转换为可搜索的知识库")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                padding-bottom: 10px;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        main_layout.addLayout(title_layout)

        # --- 分隔线 ---
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0;")
        main_layout.addWidget(line)

        # --- 文件夹管理按钮 ---
        button_layout = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("添加文件夹")
        self.add_button.clicked.connect(self.add_folder)
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        button_layout.addWidget(self.add_button)

        self.clear_button = QtWidgets.QPushButton("清空队列")
        self.clear_button.clicked.connect(self.clear_queue)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # --- 处理选项卡片 ---
        options_group = QtWidgets.QGroupBox("处理选项")
        options_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #34495e;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        options_layout = QtWidgets.QGridLayout()
        options_layout.setSpacing(15)

        # Batch processing
        self.batch_checkbox = QtWidgets.QCheckBox("批量处理模式")
        self.batch_checkbox.setToolTip(
            "批量处理模式：\n"
            "✓ 更快的 EMB 文件生成速度\n"
            "✗ 可能占用较多内存\n"
            "建议：处理大量文件时启用"
        )
        self.batch_checkbox.setChecked(False)
        self.batch_checkbox.toggled.connect(self.update_batch_processing)
        self.batch_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        options_layout.addWidget(self.batch_checkbox, 0, 0)

        # Chunk 粒度选择
        granularity_label = QtWidgets.QLabel("文档切分粒度：")
        granularity_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #34495e;")
        options_layout.addWidget(granularity_label, 1, 0)

        self.granularity_combo = QtWidgets.QComboBox()
        self.granularity_combo.addItem("页面级 - 适合 PPT 和幻灯片")
        self.granularity_combo.addItem("段落级 - 适合学术论文")
        self.granularity_combo.addItem("固定长度块 - 适合长文档")
        self.granularity_combo.setCurrentIndex(0)

        # 详细的工具提示
        self.granularity_combo.setToolTip(
            "页面级：\n"
            "  每页作为一个完整单元\n"
            "  适合：PPT、演示文稿、独立页面内容\n"
            "  生成文件：doc.json\n\n"
            "段落级：\n"
            "  按段落智能切分（自动合并短段落）\n"
            "  适合：双栏论文、学术文章、书籍\n"
            "  生成文件：doc.para.json\n\n"
            "固定长度块：\n"
            "  固定 2000 字符/块，200 字符重叠\n"
            "  适合：长篇报告、小说、连续文档\n"
            "  生成文件：doc.fixed.json"
        )

        self.granularity_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background: white;
                font-size: 13px;
                color: #34495e;
            }
            QComboBox:hover {
                border-color: #3498db;
                background-color: #f8f9fa;
                color: #34495e;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #34495e;
                margin-right: 10px;
            }
        """)
        options_layout.addWidget(self.granularity_combo, 1, 1, 1, 2)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # --- 处理队列表格 ---
        queue_group = QtWidgets.QGroupBox("处理队列")
        queue_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #34495e;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        queue_layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件夹路径", "生成 JSON", "生成 EMB", "操作"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #bdc3c7;
                gridline-color: #ecf0f1;
                background-color: white;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 13px;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        queue_layout.addWidget(self.table)
        queue_group.setLayout(queue_layout)
        main_layout.addWidget(queue_group)

        # --- 开始处理按钮和进度条 ---
        process_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始处理")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 15px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        process_layout.addWidget(self.start_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                font-size: 13px;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        process_layout.addWidget(self.progress_bar, 1)
        main_layout.addLayout(process_layout)

        # --- Log Text Area ---
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

    def update_batch_processing(self, state):
        global BATCH_PROCESSING_ENABLED
        BATCH_PROCESSING_ENABLED = state

    def add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            folder_item = QtWidgets.QTableWidgetItem(folder)
            self.table.setItem(row_position, 0, folder_item)
            json_checkbox = QtWidgets.QCheckBox()
            json_checkbox.setChecked(True)
            self.table.setCellWidget(row_position, 1, json_checkbox)
            emb_checkbox = QtWidgets.QCheckBox()
            if not FASTEMBED_AVAILABLE:
                emb_checkbox.setChecked(False)
                emb_checkbox.setEnabled(False)
            else:
                emb_checkbox.setChecked(True)
            self.table.setCellWidget(row_position, 2, emb_checkbox)
            def update_emb(checked, emb=emb_checkbox):
                emb.setEnabled(checked and FASTEMBED_AVAILABLE)
                if not (checked and FASTEMBED_AVAILABLE):
                    emb.setChecked(False)
            json_checkbox.toggled.connect(update_emb)
            remove_button = QtWidgets.QPushButton("移除")
            remove_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 4px 12px;
                    border-radius: 3px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            remove_button.clicked.connect(lambda _, row=row_position: self.remove_row(row))
            self.table.setCellWidget(row_position, 3, remove_button)

    def remove_row(self, row):
        self.table.removeRow(row)
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 3)
            if widget:
                try:
                    widget.clicked.disconnect()
                except Exception:
                    pass
                widget.clicked.connect(lambda checked, row=r: self.remove_row(row))

    def clear_queue(self):
        self.table.setRowCount(0)

    def append_log(self, message):
        self.log_text.append(message)

    def update_progress(self, percent):
        self.progress_bar.setValue(percent)

    def start_processing(self):
        if self.table.rowCount() == 0:
            self.append_log("队列中没有文件夹")
            return

        self.start_button.setEnabled(False)

        # 获取粒度选择
        granularity_index = self.granularity_combo.currentIndex()
        granularity_map = {0: "page", 1: "paragraph", 2: "fixed"}
        granularity = granularity_map[granularity_index]

        queue_items = []
        for row in range(self.table.rowCount()):
            folder = self.table.item(row, 0).text()
            json_widget = self.table.cellWidget(row, 1)
            emb_widget = self.table.cellWidget(row, 2)
            process_json = json_widget.isChecked() if json_widget else False
            process_emb = emb_widget.isChecked() if emb_widget else False
            queue_items.append({
                "folder": folder,
                "process_json": process_json,
                "process_emb": process_emb,
                "granularity": granularity  # 新增：传递粒度
            })

        self.worker = Worker(queue_items)
        self.worker_thread = QtCore.QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.logSignal.connect(self.append_log)
        self.worker.progressSignal.connect(self.update_progress)
        self.worker.finishedSignal.connect(self.on_processing_finished)
        self.worker.finishedSignal.connect(self.worker_thread.quit)
        self.worker.finishedSignal.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()

    def on_processing_finished(self):
        self.append_log("\n所有处理已完成")
        self.start_button.setEnabled(True)

    def on_thread_finished(self):
        self.worker_thread = None

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

##########################################
# Main
##########################################

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
