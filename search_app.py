import os
import re
import sys
import json
import subprocess
import platform
import math                              # ← added
import numpy as np
from collections import Counter         # ← added

# 导入优化的搜索引擎模块
from search_engine import SearchEngine, EmbeddingSearcher  # ← added

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QStatusBar,
    QComboBox,
    QShortcut,
    QScrollBar,
    QMenuBar,
    QAction,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QAbstractItemView,
    QSplitter
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QKeySequence
from PyQt5.QtCore import Qt, QRectF, QTimer
import fitz  # PyMuPDF
import unicodedata

# --- BM25s imports ---
import bm25s

###############################################################################
# Attempt fastembed import
###############################################################################
FASTEMBED_AVAILABLE = False
FASTEMBED_ENCODER = None

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    TextEmbedding = None

###############################################################################
# Global variables for corpus and BM25 model
###############################################################################
GLOBAL_CORPUS = []
GLOBAL_BM25_MODEL = None

# We'll store a fastembed.TextEmbedding model here if needed
GLOBAL_EMBED_MODEL = None

# The maximum number of BM25 search hits to return before any re-ranking.
MAX_SEARCH_RESULTS = 50

# For convenience, we store the folders database in memory (list of dicts):
FOLDERS_DB = []

###############################################################################
# Helper functions
###############################################################################
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def init_embedding_model(log_callback=None):
    """
    初始化全局 embedding 模型（如果尚未初始化）。
    使用缓存目录避免重复下载。
    返回 True 表示成功，False 表示失败。
    """
    global GLOBAL_EMBED_MODEL, FASTEMBED_AVAILABLE

    # 如果已经初始化，直接返回
    if GLOBAL_EMBED_MODEL is not None:
        if log_callback:
            log_callback("✅ Embedding 模型已加载，无需重新初始化")
        return True

    # 检查 fastembed 是否可用
    if not FASTEMBED_AVAILABLE:
        if log_callback:
            log_callback("FastEmbed not installed. Embeddings won't be used.")
        return False

    try:
        # 设置模型缓存目录
        cache_dir = os.path.expanduser("~/.cache/fastembed")
        model_name = "jinaai/jina-embeddings-v2-base-zh"

        # 检查模型是否已缓存
        model_cache_path = os.path.join(cache_dir, model_name.replace("/", "--"))
        if os.path.exists(model_cache_path):
            if log_callback:
                log_callback(f"✅ 使用缓存的模型: {model_name}")
        else:
            if log_callback:
                log_callback(f"⏬ 首次使用，下载模型: {model_name}")
                log_callback("   这可能需要几分钟，请耐心等待...")

        if log_callback:
            log_callback("Initializing embedding model (jinaai/jina-embeddings-v2-base-zh)...")

        GLOBAL_EMBED_MODEL = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

        if log_callback:
            log_callback("✅ Embedding 模型加载成功 (768维, 8192 token)")
            log_callback("   模型已缓存，下次运行将直接使用")
        return True

    except Exception as e:
        if log_callback:
            log_callback(f"Error initializing embedding model: {e}")
        return False


def load_folders_database():
    """
    Attempts to load 'folders.ini'. 
    If it doesn't exist, returns None => "not initialized".
    If it exists but is invalid or empty, returns empty list => valid but no data.
    Otherwise, returns the list.
    """
    if not os.path.exists("folders.ini"):
        return None

    try:
        with open("folders.ini", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                # If the JSON is not a list, treat it as invalid
                return []
            return data
    except Exception as e:
        print(f"Error reading folders.ini: {e}")
        return []


def save_folders_database(folders_list):
    """
    Saves the given folders_list to 'folders.ini'.
    """
    try:
        with open("folders.ini", "w", encoding="utf-8") as f:
            json.dump(folders_list, f, indent=2)
    except Exception as e:
        print(f"Error writing folders.ini: {e}")


def load_corpus_and_initialize_bm25(folders_list):
    """
    Given a list of folder entries (each with {checked, path, description}),
    load all .json (and matching .emb) from the *checked* folders into GLOBAL_CORPUS,
    and build a BM25 index.
    
    If a folder does not exist, we store "Folder xxxxxx not found" in error_messages.
    Returns (error_messages, status_message).
    """
    global GLOBAL_CORPUS, GLOBAL_BM25_MODEL

    GLOBAL_CORPUS.clear()
    GLOBAL_BM25_MODEL = None
    error_messages = []

    # Collect all JSON files from the checked folders
    all_json_files = []
    for folder_entry in folders_list:
        if not folder_entry.get("checked"):
            continue
        folder_path = folder_entry.get("path", "")
        if not os.path.isdir(folder_path):
            # Folder not found
            error_messages.append(f"Folder {folder_path} not found")
            continue

        json_files_in_folder = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.endswith(".json")
        ]
        all_json_files.extend(json_files_in_folder)

    if not all_json_files:
        return error_messages, "No JSON files found in the selected folders."

    # Load data from each JSON file
    for file_path in all_json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                docs = json.load(json_file)

            # For each doc, if 'filename' is given, make it absolute
            folder_of_json = os.path.dirname(file_path)
            for doc in docs:
                pdf_name = doc.get('filename', '')
                if pdf_name and not os.path.isabs(pdf_name):
                    doc['filename'] = os.path.join(folder_of_json, pdf_name)

            GLOBAL_CORPUS.extend(docs)
        except Exception as e:
            error_messages.append(f"Error reading {file_path}: {e}")

    if not GLOBAL_CORPUS:
        return error_messages, "No documents found in any JSON file."

    # Build BM25 index
    texts = [doc['text'] for doc in GLOBAL_CORPUS if 'text' in doc]
    if texts:
        GLOBAL_BM25_MODEL = bm25s.BM25()
        tokenized_corpus = bm25s.tokenize(texts, stopwords="en")
        GLOBAL_BM25_MODEL.index(tokenized_corpus)
    else:
        return error_messages, "No textual data to build BM25 model."

    # Attempt to load embeddings for each JSON
    load_embeddings_for_corpus(all_json_files)

    # 验证 embedding 质量
    valid_count, issues = validate_corpus_embeddings(GLOBAL_CORPUS)
    if valid_count > 0:
        print(f"✅ 加载了 {valid_count} 个文档的 embeddings")
        if issues:
            print(f"⚠️  发现 {len(issues)} 个质量问题:")
            for issue in issues[:10]:  # 只显示前10个
                print(f"   {issue}")
            if len(issues) > 10:
                print(f"   ... 还有 {len(issues) - 10} 个问题")

    return error_messages, "BM25 model successfully initialized."


def load_embeddings_for_corpus(json_file_list):
    """
    For each .json file in 'json_file_list', tries to find a matching .emb file
    in the same folder with the same base name. If present, load the embeddings.
    """
    global GLOBAL_CORPUS
    emb_count = 0
    corpus_index = 0

    for file_path in json_file_list:
        base, _ext = os.path.splitext(file_path)
        emb_file_path = base + ".emb"

        # Count how many pages are in this JSON
        try:
            with open(file_path, "r", encoding="utf-8") as j:
                pages_in_json = json.load(j)
        except:
            pages_in_json = []
        num_pages = len(pages_in_json)

        if not os.path.exists(emb_file_path):
            # Just move corpus_index forward
            corpus_index += num_pages
            continue

        # We found a .emb file
        try:
            with open(emb_file_path, "r", encoding="utf-8") as emb_file:
                pages_with_emb = json.load(emb_file)
        except:
            pages_with_emb = []

        if len(pages_in_json) != len(pages_with_emb):
            print(f"Warning: mismatch in #pages for {file_path} vs {emb_file_path}")
            min_len = min(len(pages_in_json), len(pages_with_emb))
        else:
            min_len = len(pages_in_json)

        # Attach embeddings
        for i in range(min_len):
            doc = GLOBAL_CORPUS[corpus_index + i]
            if 'embedding' in pages_with_emb[i]:
                doc['embedding'] = np.array(pages_with_emb[i]['embedding'], dtype=np.float32)
                emb_count += 1

        corpus_index += num_pages

    print(f"Loaded embeddings for {emb_count} pages total.")


def validate_corpus_embeddings(corpus):
    """
    验证语料库中 embedding 的质量和一致性
    返回：(valid_count, issues_list)
    """
    issues = []
    valid_count = 0
    expected_dim = None

    for idx, doc in enumerate(corpus):
        if 'embedding' not in doc:
            continue

        emb = np.array(doc['embedding'])
        valid_count += 1

        # 检查维度一致性
        if expected_dim is None:
            expected_dim = len(emb)
        elif len(emb) != expected_dim:
            issues.append(f"文档 {idx} ({doc.get('filename', 'unknown')}): "
                         f"维度不匹配 ({len(emb)} vs 预期 {expected_dim})")

        # 检查向量范数（是否归一化）
        emb_norm = np.linalg.norm(emb)
        if emb_norm < 0.01:  # 接近零向量
            issues.append(f"文档 {idx} ({doc.get('filename', 'unknown')}): "
                         f"向量接近零 (norm={emb_norm:.6f})")
        elif not (0.9 < emb_norm < 1.1):  # 不在归一化范围
            issues.append(f"文档 {idx} ({doc.get('filename', 'unknown')}): "
                         f"可能未归一化 (norm={emb_norm:.4f})")

        # 检查是否包含 NaN 或 Inf
        if np.any(np.isnan(emb)):
            issues.append(f"文档 {idx} ({doc.get('filename', 'unknown')}): 包含 NaN 值")
        if np.any(np.isinf(emb)):
            issues.append(f"文档 {idx} ({doc.get('filename', 'unknown')}): 包含 Inf 值")

    return valid_count, issues


###############################################################################
# Minimal span-based scoring functions (unchanged)
###############################################################################
def minimal_span_score(text, query_terms):
    norm_text = remove_accents(text.lower())
    norm_query_terms = [remove_accents(qt.lower()) for qt in query_terms]

    words = norm_text.split()
    positions = {term: [] for term in norm_query_terms}
    for i, w in enumerate(words):
        if w in positions:
            positions[w].append(i)

    for term in norm_query_terms:
        if not positions[term]:
            return 0.0

    all_positions = []
    for t in norm_query_terms:
        all_positions.extend((p, t) for p in positions[t])
    all_positions.sort(key=lambda x: x[0])

    best_span = len(words) + 1
    found_terms = {}
    left = 0
    for right in range(len(all_positions)):
        pos_right, term_right = all_positions[right]
        found_terms[term_right] = pos_right

        while len(found_terms) == len(norm_query_terms):
            span = max(found_terms.values()) - min(found_terms.values()) + 1
            if span < best_span:
                best_span = span
            pos_left, term_left = all_positions[left]
            if found_terms.get(term_left, None) == pos_left:
                del found_terms[term_left]
            left += 1

    return 1.0 / (best_span + 1)


def rerank_minimal_span(top_docs, query_terms):
    global GLOBAL_CORPUS
    doc_scores = []
    for doc_id, bm25_score in top_docs:
        text = GLOBAL_CORPUS[doc_id]['text']
        ms_score = minimal_span_score(text, query_terms)
        doc_scores.append((doc_id, ms_score))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores


###############################################################################
# Exact text search (unchanged)
###############################################################################
def rerank_exact_text(top_docs, query_phrase):
    global GLOBAL_CORPUS
    query_norm = remove_accents(query_phrase.lower())
    matched = []
    unmatched = []

    for doc_id, bm25_score in top_docs:
        doc_text = GLOBAL_CORPUS[doc_id]['text']
        doc_text_norm = remove_accents(doc_text.lower())
        if query_norm in doc_text_norm:
            matched.append((doc_id, bm25_score))
        else:
            unmatched.append((doc_id, bm25_score))

    return matched + unmatched


###############################################################################
# Helper function for "Simple text search"
###############################################################################
def parse_simple_search_query(query_str):
    pattern = r'"([^"]+)"|(\S+)'
    matches = re.findall(pattern, query_str)

    quoted_phrases = []
    unquoted_words = []
    for (phrase, word) in matches:
        if phrase:
            quoted_phrases.append(phrase)
        elif word:
            unquoted_words.append(word)
    return quoted_phrases, unquoted_words


###############################################################################
# A custom QGraphicsView with dynamic page loading support
###############################################################################
class ClickableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pdf_path = None
        self.current_page = 1
        self.total_pages = 1

        # Dynamic loading attributes
        self.rendered_pages = {}  # {page_num: (pixmap_item, y_position, height)}
        self.first_rendered_page = None
        self.last_rendered_page = None
        self.page_gap = 0
        self.is_loading = False

        # Store reference to SearchApp (will be set after init)
        self.search_app = None

        # Auto-fit width control
        self.auto_fit_width = True  # Enable auto-fit by default
        self.resize_timer = None  # Timer to debounce resize events

        # Connect scroll event
        self.verticalScrollBar().valueChanged.connect(self.on_scroll)

    def set_search_app(self, search_app):
        """Set reference to the main SearchApp instance."""
        self.search_app = search_app

    def set_pdf_details(self, pdf_path, page, total_pages):
        self.current_pdf_path = pdf_path
        self.current_page = page
        self.total_pages = total_pages

    def clear_rendered_pages(self):
        """Clear all tracking of rendered pages."""
        self.rendered_pages.clear()
        self.first_rendered_page = None
        self.last_rendered_page = None

    def on_scroll(self, value):
        """Handle scroll events to trigger dynamic loading."""
        if self.is_loading or not self.current_pdf_path or not self.search_app:
            return

        vbar = self.verticalScrollBar()
        threshold = 500  # Trigger loading when within 500 pixels of edge

        # Check if near top
        if value < threshold and self.first_rendered_page and self.first_rendered_page > 1:
            self.load_previous_pages()

        # Check if near bottom
        elif (vbar.maximum() - value) < threshold and self.last_rendered_page and self.last_rendered_page < self.total_pages:
            self.load_next_pages()

    def load_previous_pages(self):
        """Request loading of previous pages from SearchApp."""
        if self.search_app:
            self.is_loading = True
            try:
                self.search_app.load_previous_pages()
            finally:
                self.is_loading = False

    def load_next_pages(self):
        """Request loading of next pages from SearchApp."""
        if self.search_app:
            self.is_loading = True
            try:
                self.search_app.load_next_pages()
            finally:
                self.is_loading = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_pdf_path:
            self.open_pdf_externally()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        """Handle window resize events to auto-fit PDF width."""
        super().resizeEvent(event)

        # Only auto-adjust if enabled and we have a PDF loaded
        if self.auto_fit_width and self.current_pdf_path and self.search_app:
            # Use a timer to debounce resize events (avoid too many re-renders)
            if self.resize_timer is not None:
                self.resize_timer.stop()
                self.resize_timer.deleteLater()

            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self._on_resize_finished)
            self.resize_timer.start(300)  # Wait 300ms after resize stops

    def _on_resize_finished(self):
        """Called after resize event has settled."""
        if self.search_app and self.current_pdf_path:
            # Calculate new scale factor and re-render
            self.search_app.auto_fit_pdf_width()

    def open_pdf_externally(self):
        """Try to open PDF in external viewer with fallback options."""
        if not self.current_pdf_path:
            return

        try:
            if platform.system() == "Windows":
                # Try Adobe Reader, then default viewer
                readers = [
                    ["AcroRd32.exe", "/A", f"page={self.current_page}", self.current_pdf_path],
                    ["start", "", self.current_pdf_path]  # Default Windows viewer
                ]
            else:
                # Try multiple Linux PDF viewers
                readers = [
                    ["okular", self.current_pdf_path, "-p", str(self.current_page)],
                    ["evince", "--page-label=" + str(self.current_page), self.current_pdf_path],
                    ["xdg-open", self.current_pdf_path],  # System default
                ]

            opened = False
            for cmd in readers:
                try:
                    subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)
                    opened = True
                    break
                except FileNotFoundError:
                    continue
                except Exception:
                    continue

            if not opened:
                print(f"Warning: No PDF viewer found. Please install okular, evince, or set a default PDF viewer.")

        except Exception as e:
            print(f"Error opening PDF externally: {e}")


###############################################################################
# Dialog for managing folders (unchanged)
###############################################################################
class FoldersDialog(QDialog):
    def __init__(self, folders_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Data Folders")
        self.folders_list = folders_list  # We'll work on a copy in memory

        # Make this window 3× wider (arbitrary choice: 1200x600)
        self.resize(1200, 600)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Load?", "Folder Path", "Description"])
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.load_data_into_table()

        # Buttons
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add folder")
        self.remove_button = QPushButton("Remove folder")
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)

        self.add_button.clicked.connect(self.add_folder_row)
        self.remove_button.clicked.connect(self.remove_folder_row)

        # OK / Cancel
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept_dialog)
        self.button_box.rejected.connect(self.reject_dialog)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

        # Make it modal
        self.setModal(True)

    def load_data_into_table(self):
        self.table.setRowCount(len(self.folders_list))
        for row, folder_entry in enumerate(self.folders_list):
            # Column 0: checkbox
            check_box = QCheckBox()
            check_box.setChecked(bool(folder_entry.get("checked", False)))
            self.table.setCellWidget(row, 0, check_box)

            # Column 1: folder path
            path_item = QTableWidgetItem(folder_entry.get("path", ""))
            self.table.setItem(row, 1, path_item)

            # Column 2: description
            desc_item = QTableWidgetItem(folder_entry.get("description", ""))
            self.table.setItem(row, 2, desc_item)

    def add_folder_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)

        check_box = QCheckBox()
        check_box.setChecked(True)
        self.table.setCellWidget(row, 0, check_box)

        folder_path_item = QTableWidgetItem("")
        self.table.setItem(row, 1, folder_path_item)

        desc_item = QTableWidgetItem("")
        self.table.setItem(row, 2, desc_item)

        # Optionally open a file dialog right away
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if folder:
            folder_path_item.setText(folder)

    def remove_folder_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def accept_dialog(self):
        new_folders = []
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            checked = w.isChecked() if w else False

            path_item = self.table.item(row, 1)
            path = path_item.text() if path_item else ""

            desc_item = self.table.item(row, 2)
            desc = desc_item.text() if desc_item else ""

            # If user didn't pick a path, prompt now
            if not path:
                folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
                path = folder

            new_folders.append({
                "checked": checked,
                "path": path,
                "description": desc,
            })

        self.folders_list[:] = new_folders  # update in place
        super().accept()

    def reject_dialog(self):
        super().reject()


###############################################################################
# The main GUI application class
###############################################################################
class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Search Interface with PDF Viewer")
        self.current_result_index = 0
        self.results = []
        # We'll keep a dynamic list of words to highlight in the PDF
        self.query_terms = []
        self.font_size = 12
        self.scale_factor = 1.0

        self.embeddings_present = False  # whether we found .emb files
        self.search_engine = None  # ← added: 优化的搜索引擎
        self.init_ui()

        # ---------------------------------------------------------------------
        # Load the folders database if available; if not, message the user
        # ---------------------------------------------------------------------
        global FOLDERS_DB
        loaded_data = load_folders_database()
        if loaded_data is None:
            # None => "folders.ini" not found
            self.result_display.setText("Folder database not initialized")
            FOLDERS_DB = []
        else:
            FOLDERS_DB = loaded_data

        # If we have a valid list, attempt to load the corpus
        if FOLDERS_DB:
            errors, status = load_corpus_and_initialize_bm25(FOLDERS_DB)
            # Show any error messages (e.g. missing folders)
            for err in errors:
                self.result_display.append(err)
            self.result_display.append(status)
        # If FOLDERS_DB is empty and not None, it means folders.ini was present but invalid or empty
        if FOLDERS_DB == [] and loaded_data is not None:
            self.result_display.setText("No folders in database. Please add some folders.")

        # Check if we actually loaded any embeddings
        self.embeddings_present = any(('embedding' in doc) for doc in GLOBAL_CORPUS)

        # Attempt to initialize the global embedding model if we have embeddings
        if self.embeddings_present:
            success = init_embedding_model(lambda msg: self.result_display.append(msg))
            if success:
                # ← added: 初始化优化的搜索引擎
                self.search_engine = SearchEngine(
                    corpus=GLOBAL_CORPUS,
                    bm25_model=GLOBAL_BM25_MODEL,
                    embed_model=GLOBAL_EMBED_MODEL
                )
                if GLOBAL_BM25_MODEL is not None:
                    self.result_display.append("Corpus and Embeddings loaded successfully. Ready to search.")
                else:
                    self.result_display.append("Embeddings loaded successfully (no BM25). Ready to search.")
            else:
                if GLOBAL_BM25_MODEL is not None:
                    self.result_display.append("FastEmbed not installed. Embeddings won't be used.")
                else:
                    self.result_display.append("No BM25 and no FastEmbed. Check your installation.")
        else:
            if GLOBAL_BM25_MODEL is None:
                self.result_display.setText("No corpus or BM25 model available.")
            else:
                self.result_display.append("Corpus loaded successfully. Ready to search.")

    def init_ui(self):
        # ---------------------------------------------------------------------
        # Instead of a simple layout, use a QSplitter with vertical orientation
        # so top = text area, bottom = PDF viewer
        # ---------------------------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)

        # Top widget (text area)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Row for "Search method" and "Reranking method"
        top_row_layout = QHBoxLayout()

        self.search_method_label = QLabel("搜索方法:")
        self.search_method_label.setStyleSheet("font-weight: bold; color: #495057;")
        self.search_method_combo = QComboBox()
        self.search_method_combo.addItem("混合搜索 (智能)")
        self.search_method_combo.addItem("语义搜索 (Embeddings)")
        self.search_method_combo.addItem("BM25 关键词")
        self.search_method_combo.addItem("BM25 前缀匹配")
        self.search_method_combo.addItem("精确文本搜索")

        # 设置工具提示
        self.search_method_combo.setItemData(0, "融合精确匹配、语义理解和关键词检索，智能排序（推荐）", Qt.ToolTipRole)
        self.search_method_combo.setItemData(1, "基于深度学习的语义理解，适合同义词和概念性查询", Qt.ToolTipRole)
        self.search_method_combo.setItemData(2, "经典关键词搜索，快速精准", Qt.ToolTipRole)
        self.search_method_combo.setItemData(3, "支持前缀匹配和负向排除（如：compar -comparison）", Qt.ToolTipRole)
        self.search_method_combo.setItemData(4, "精确短语匹配，支持引号", Qt.ToolTipRole)

        self.search_method_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #dee2e6;
                border-radius: 4px;
                background: white;
            }
            QComboBox:hover {
                border-color: #007bff;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.search_method_combo.currentIndexChanged.connect(self.update_rerank_combo_status)

        top_row_layout.addWidget(self.search_method_label)
        top_row_layout.addWidget(self.search_method_combo, 1)
        top_row_layout.addSpacing(20)

        self.rerank_label = QLabel("重排序:")
        self.rerank_label.setStyleSheet("font-weight: bold; color: #495057;")
        self.rerank_combo = QComboBox()
        self.rerank_combo.addItem("无重排序")
        self.rerank_combo.addItem("最小跨度评分")
        self.rerank_combo.addItem("精确文本匹配")
        self.rerank_combo.addItem("Embedding 重排序")
        self.rerank_combo.setEditable(False)
        self.rerank_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #dee2e6;
                border-radius: 4px;
                background: white;
            }
            QComboBox:hover {
                border-color: #007bff;
            }
        """)
        self.rerank_combo.currentIndexChanged.connect(self.search)

        top_row_layout.addWidget(self.rerank_label)
        top_row_layout.addWidget(self.rerank_combo, 1)
        top_row_layout.addSpacing(30)

        # 管理数据文件夹按钮
        self.manage_folders_button = QPushButton("管理数据文件夹")
        self.manage_folders_button.setToolTip("添加/删除/配置数据文件夹")
        self.manage_folders_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.manage_folders_button.clicked.connect(self.on_manage_folders)
        top_row_layout.addWidget(self.manage_folders_button)

        top_layout.addLayout(top_row_layout)

        # Search label/input
        self.query_label = QLabel("搜索查询:")
        self.query_label.setStyleSheet("font-weight: bold; color: #495057;")
        self.query_input = QLineEdit()
        self.query_input.setFont(QFont("Arial", self.font_size))
        self.query_input.setPlaceholderText("输入搜索关键词或短语...")
        self.query_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #dee2e6;
                border-radius: 4px;
                background: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #007bff;
                background: #f8f9fa;
            }
        """)
        # 启用输入法支持（修复 Rime 等输入法无法输入的问题）
        self.query_input.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.query_input.returnPressed.connect(self.search)
        top_layout.addWidget(self.query_label)
        top_layout.addWidget(self.query_input)

        # Navigation buttons
        button_layout = QHBoxLayout()

        # 结果导航按钮
        self.prev_button = QPushButton("◀ 上一个")
        self.next_button = QPushButton("下一个 ▶")
        self.prev_button.setToolTip("显示上一个搜索结果 (Alt+Left)")
        self.next_button.setToolTip("显示下一个搜索结果 (Alt+Right)")
        self.prev_button.clicked.connect(self.show_previous_chunk)
        self.next_button.clicked.connect(self.show_next_chunk)

        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addSpacing(20)

        # 字体大小按钮
        self.decrease_font_button = QPushButton("A-")
        self.increase_font_button = QPushButton("A+")
        self.decrease_font_button.setToolTip("减小字体")
        self.increase_font_button.setToolTip("增大字体")
        self.decrease_font_button.clicked.connect(self.decrease_font_size)
        self.increase_font_button.clicked.connect(self.increase_font_size)

        font_button_style = """
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """
        self.decrease_font_button.setStyleSheet(font_button_style)
        self.increase_font_button.setStyleSheet(font_button_style)

        button_layout.addWidget(self.decrease_font_button)
        button_layout.addWidget(self.increase_font_button)
        button_layout.addSpacing(20)

        # --- PDF 裁剪复选框 ---
        self.crop_pdf_view_checkbox = QCheckBox("裁剪 PDF 白边")
        self.crop_pdf_view_checkbox.setChecked(True)
        self.crop_pdf_view_checkbox.setToolTip("自动裁剪 PDF 页面的空白边距")
        self.crop_pdf_view_checkbox.toggled.connect(self.on_toggle_crop_pdf_view)
        button_layout.addWidget(self.crop_pdf_view_checkbox)
        button_layout.addStretch()  # 添加弹性空间，使按钮靠左对齐
        # -------------------------------------------------

        top_layout.addLayout(button_layout)

        # Results text area
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Arial", self.font_size))
        top_layout.addWidget(self.result_display)

        splitter.addWidget(top_widget)  # add top widget to splitter

        # Bottom widget (PDF viewer)
        self.graphics_view = ClickableGraphicsView()
        self.graphics_view.set_search_app(self)  # Set reference to SearchApp
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        splitter.addWidget(self.graphics_view)

        # Set the initial proportions (e.g., 30% for the left and 70% for the right)
        splitter.setSizes([30, 700])  # Proportions are in pixels but will scale proportionally

        # Let both splitter panes expand or shrink
        splitter.setStretchFactor(0, 1)  # top
        splitter.setStretchFactor(1, 1)  # bottom

        # Create a container layout to hold just the splitter
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(splitter)

        self.setCentralWidget(container)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Shortcuts
        QShortcut(QKeySequence(Qt.Key_PageUp), self, self.page_up)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, self.page_down)
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)

        QShortcut(QKeySequence("Alt+Left"), self, self.show_previous_chunk)
        QShortcut(QKeySequence("Alt+Right"), self, self.show_next_chunk)
        QShortcut(QKeySequence("Alt+Up"), self, self.page_up)       # ← added
        QShortcut(QKeySequence("Alt+Down"), self, self.page_down)   # ← added

        # PDF scrolling shortcuts
        QShortcut(QKeySequence("Ctrl+Left"), self, self.scroll_pdf_left)
        QShortcut(QKeySequence("Ctrl+Right"), self, self.scroll_pdf_right)
        QShortcut(QKeySequence("Ctrl+Up"), self, self.scroll_pdf_up)
        QShortcut(QKeySequence("Ctrl+Down"), self, self.scroll_pdf_down)

        # Set initial status for Reranking combo
        self.update_rerank_combo_status()

    def update_rerank_combo_status(self):
        current_method = self.search_method_combo.currentText()
        # Disable rerank for simple, embeddings, hybrid and substring methods
        if current_method in ("精确文本搜索", "语义搜索 (Embeddings)", "BM25 前缀匹配", "混合搜索 (智能)"):
            self.rerank_combo.setEnabled(False)
        else:
            self.rerank_combo.setEnabled(True)

    # -------------------------------------------------------------------------
    # PDF display and navigation with dynamic loading
    # -------------------------------------------------------------------------
    def display_pdf_page(self, pdf_path, page_number):
        """
        Initialize PDF viewer with the target page and surrounding pages.
        Sets up dynamic loading for continuous scrolling.
        """
        try:
            # Clear previous rendering state
            self.graphics_view.clear_rendered_pages()
            self.graphics_scene.clear()

            # Open document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Set PDF details
            self.graphics_view.set_pdf_details(pdf_path, page_number, total_pages)

            # Auto-fit width on first load if enabled
            if self.graphics_view.auto_fit_width:
                # Get the current page dimensions
                page = doc[page_number - 1]

                # Apply cropping if enabled to get actual content dimensions
                if self.crop_pdf_view_checkbox.isChecked():
                    text_blocks = page.get_text("blocks")
                    if text_blocks:
                        x_min = min(block[0] for block in text_blocks)
                        x_max = max(block[2] for block in text_blocks)
                        page_width = x_max - x_min
                    else:
                        page_width = page.rect.width
                else:
                    page_width = page.rect.width

                # Calculate available width
                view_width = self.graphics_view.viewport().width()
                margin = 20

                # Calculate scale factor to fit width
                base_dpi = 150
                zoom = base_dpi / 72
                rendered_width = page_width * zoom

                if rendered_width > 0:
                    target_scale = (view_width - margin) / rendered_width
                    self.scale_factor = max(0.1, min(5.0, target_scale))

            # Initial render: target page +/- 2 pages
            initial_pages_before = 2
            initial_pages_after = 2

            start_page = max(1, page_number - initial_pages_before)
            end_page = min(total_pages, page_number + initial_pages_after)

            # Render initial pages
            self.render_pages(doc, start_page, end_page, page_number)

            doc.close()

        except Exception as e:
            self.result_display.setText(f"Error displaying PDF: {e}")

    def render_pages(self, doc, start_page, end_page, target_page=None):
        """
        Render a range of PDF pages and add them to the scene.

        Args:
            doc: fitz.Document object
            start_page: First page to render (1-indexed)
            end_page: Last page to render (1-indexed)
            target_page: Page to scroll to after rendering (optional)
        """
        base_dpi = 150
        dpi = base_dpi * self.scale_factor
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        page_gap = 10 * zoom
        self.graphics_view.page_gap = page_gap

        # Calculate y_offset (if appending to existing content)
        if self.graphics_view.last_rendered_page and start_page > self.graphics_view.last_rendered_page:
            # Appending: start after last rendered page
            last_page_info = self.graphics_view.rendered_pages.get(self.graphics_view.last_rendered_page)
            if last_page_info:
                y_offset = last_page_info[1] + last_page_info[2] + page_gap
            else:
                y_offset = 0
        elif self.graphics_view.first_rendered_page and end_page < self.graphics_view.first_rendered_page:
            # Prepending: will be handled separately
            return self.prepend_pages(doc, start_page, end_page)
        else:
            # Initial render
            y_offset = 0

        target_page_y = None

        # Render each page in the range
        for current_page_num in range(start_page, end_page + 1):
            # Skip already rendered pages
            if current_page_num in self.graphics_view.rendered_pages:
                if target_page and current_page_num == target_page:
                    target_page_y = self.graphics_view.rendered_pages[current_page_num][1]
                continue

            page = doc[current_page_num - 1]

            # Apply cropping if enabled
            if self.crop_pdf_view_checkbox.isChecked():
                text_blocks = page.get_text("blocks")
                if text_blocks:
                    x_min = float('inf')
                    y_min = float('inf')
                    x_max = float('-inf')
                    y_max = float('-inf')

                    for block in text_blocks:
                        x0, y0, x1, y1 = block[:4]
                        x_min = min(x_min, x0)
                        y_min = min(y_min, y0)
                        x_max = max(x_max, x1)
                        y_max = max(y_max, y1)

                    crop_box = fitz.Rect(x_min, y_min, x_max, y_max)
                    media_box = page.mediabox

                    if (crop_box.x0 >= media_box.x0 and crop_box.y0 >= media_box.y0 and
                        crop_box.x1 <= media_box.x1 and crop_box.y1 <= media_box.y1):
                        page.set_cropbox(crop_box)

            # Record the Y position of the target page
            if target_page and current_page_num == target_page:
                target_page_y = y_offset

            # Render the page
            pix = page.get_pixmap(matrix=mat)
            qt_img = QPixmap()
            qt_img.loadFromData(pix.tobytes("ppm"))

            # Add page to scene at current y_offset
            pixmap_item = QGraphicsPixmapItem(qt_img)
            pixmap_item.setPos(0, y_offset)
            self.graphics_scene.addItem(pixmap_item)

            # Store page info
            self.graphics_view.rendered_pages[current_page_num] = (pixmap_item, y_offset, qt_img.height())

            # Highlight search terms on this page
            self.highlight_page_terms(page, y_offset, zoom)

            # Update y_offset for next page
            y_offset += qt_img.height() + page_gap

        # Update rendered page range
        if self.graphics_view.first_rendered_page is None or start_page < self.graphics_view.first_rendered_page:
            self.graphics_view.first_rendered_page = start_page
        if self.graphics_view.last_rendered_page is None or end_page > self.graphics_view.last_rendered_page:
            self.graphics_view.last_rendered_page = end_page

        # Update scene rect
        self.graphics_scene.setSceneRect(self.graphics_scene.itemsBoundingRect())

        # Scroll to target page if specified
        if target_page_y is not None:
            self.graphics_view.verticalScrollBar().setValue(int(target_page_y))

    def prepend_pages(self, doc, start_page, end_page):
        """
        Prepend pages before the currently rendered content.
        Maintains scroll position relative to previously rendered content.
        """
        base_dpi = 150
        dpi = base_dpi * self.scale_factor
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        page_gap = self.graphics_view.page_gap

        # Calculate total height of new pages
        new_content_height = 0
        pages_to_add = []

        for current_page_num in range(start_page, end_page + 1):
            if current_page_num in self.graphics_view.rendered_pages:
                continue

            page = doc[current_page_num - 1]

            # Apply cropping if enabled
            if self.crop_pdf_view_checkbox.isChecked():
                text_blocks = page.get_text("blocks")
                if text_blocks:
                    x_min = float('inf')
                    y_min = float('inf')
                    x_max = float('-inf')
                    y_max = float('-inf')

                    for block in text_blocks:
                        x0, y0, x1, y1 = block[:4]
                        x_min = min(x_min, x0)
                        y_min = min(y_min, y0)
                        x_max = max(x_max, x1)
                        y_max = max(y_max, y1)

                    crop_box = fitz.Rect(x_min, y_min, x_max, y_max)
                    media_box = page.mediabox

                    if (crop_box.x0 >= media_box.x0 and crop_box.y0 >= media_box.y0 and
                        crop_box.x1 <= media_box.x1 and crop_box.y1 <= media_box.y1):
                        page.set_cropbox(crop_box)

            # Render the page
            pix = page.get_pixmap(matrix=mat)
            qt_img = QPixmap()
            qt_img.loadFromData(pix.tobytes("ppm"))

            pages_to_add.append((current_page_num, page, qt_img))
            new_content_height += qt_img.height() + page_gap

        # Shift existing items down
        for page_num, (item, old_y, height) in self.graphics_view.rendered_pages.items():
            new_y = old_y + new_content_height
            item.setPos(0, new_y)
            self.graphics_view.rendered_pages[page_num] = (item, new_y, height)

        # Add new pages at the top
        y_offset = 0
        for current_page_num, page, qt_img in pages_to_add:
            pixmap_item = QGraphicsPixmapItem(qt_img)
            pixmap_item.setPos(0, y_offset)
            self.graphics_scene.addItem(pixmap_item)

            # Store page info
            self.graphics_view.rendered_pages[current_page_num] = (pixmap_item, y_offset, qt_img.height())

            # Highlight search terms
            self.highlight_page_terms(page, y_offset, zoom)

            y_offset += qt_img.height() + page_gap

        # Update rendered page range
        self.graphics_view.first_rendered_page = start_page

        # Update scene rect
        self.graphics_scene.setSceneRect(self.graphics_scene.itemsBoundingRect())

        # Adjust scroll position to maintain view
        vbar = self.graphics_view.verticalScrollBar()
        vbar.setValue(vbar.value() + int(new_content_height))

    def highlight_page_terms(self, page, y_offset, zoom):
        """Helper to highlight search terms on a page."""
        word_positions = page.get_text("words")
        for word in word_positions:
            raw_word = word[4].lower()
            raw_word = remove_accents(raw_word)
            raw_word = re.sub(r"[^\w]+", "", raw_word)

            if any(nt in raw_word for nt in self.query_terms):
                rect = QRectF(word[0] * zoom,
                             word[1] * zoom + y_offset,
                             (word[2] - word[0]) * zoom,
                             (word[3] - word[1]) * zoom)
                highlight = QGraphicsRectItem(rect)
                highlight.setBrush(QColor(255, 255, 0, 128))
                self.graphics_scene.addItem(highlight)

    def load_previous_pages(self):
        """Load pages before the currently rendered range."""
        if not self.graphics_view.current_pdf_path:
            return

        try:
            doc = fitz.open(self.graphics_view.current_pdf_path)

            # Load 3 pages before current range
            pages_to_load = 3
            start_page = max(1, self.graphics_view.first_rendered_page - pages_to_load)
            end_page = self.graphics_view.first_rendered_page - 1

            if start_page <= end_page:
                self.prepend_pages(doc, start_page, end_page)

            doc.close()

        except Exception as e:
            print(f"Error loading previous pages: {e}")

    def load_next_pages(self):
        """Load pages after the currently rendered range."""
        if not self.graphics_view.current_pdf_path:
            return

        try:
            doc = fitz.open(self.graphics_view.current_pdf_path)

            # Load 3 pages after current range
            pages_to_load = 3
            start_page = self.graphics_view.last_rendered_page + 1
            end_page = min(self.graphics_view.total_pages,
                          self.graphics_view.last_rendered_page + pages_to_load)

            if start_page <= end_page:
                self.render_pages(doc, start_page, end_page)

            doc.close()

        except Exception as e:
            print(f"Error loading next pages: {e}")

    def on_toggle_crop_pdf_view(self):
        """
        Called when the crop PDF view checkbox is toggled.
        Re-render the current PDF page to apply the new cropping setting.
        """
        if self.graphics_view.current_pdf_path:
            self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

    def page_up(self):
        if self.graphics_view.current_pdf_path and self.graphics_view.current_page > 1:
            self.graphics_view.current_page -= 1
            self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

    def page_down(self):
        if self.graphics_view.current_pdf_path and self.graphics_view.current_page < self.graphics_view.total_pages:
            self.graphics_view.current_page += 1
            self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

    def scroll_pdf_left(self):
        hbar = self.graphics_view.horizontalScrollBar()
        hbar.setValue(hbar.value() - 50)

    def scroll_pdf_right(self):
        hbar = self.graphics_view.horizontalScrollBar()
        hbar.setValue(hbar.value() + 50)

    def scroll_pdf_up(self):
        vbar = self.graphics_view.verticalScrollBar()
        vbar.setValue(vbar.value() - 50)

    def scroll_pdf_down(self):
        vbar = self.graphics_view.verticalScrollBar()
        vbar.setValue(vbar.value() + 50)

    # -------------------------------------------------------------------------
    # Searching (with new BM25 substring case)
    # -------------------------------------------------------------------------
    def search(self):
        global GLOBAL_BM25_MODEL, GLOBAL_CORPUS, GLOBAL_EMBED_MODEL, FASTEMBED_AVAILABLE

        # Always reset self.query_terms based on the *current* query
        raw_query = self.query_input.text().strip()
        self.query_terms = [remove_accents(w.lower()) for w in re.findall(r"\w+", raw_query, flags=re.IGNORECASE)]

        if not GLOBAL_CORPUS:
            self.result_display.setText("No corpus loaded.")
            return

        if not raw_query:
            self.result_display.setText("Please enter a search query.")
            return

        search_method = self.search_method_combo.currentText()
        method = self.rerank_combo.currentText()

        # ---------------------------------------------------------------------
        # CASE 0: "Hybrid Search (Smart)" - 智能混合搜索
        # ---------------------------------------------------------------------
        if search_method == "混合搜索 (智能)":
            if not self.search_engine:
                self.result_display.setText("Search engine not available.")
                return

            # 使用混合搜索
            try:
                hybrid_results = self.search_engine.search(
                    raw_query,
                    method="hybrid",
                    max_results=MAX_SEARCH_RESULTS
                )

                # hybrid_results 格式: [(idx, score, tags), ...]
                # 转换为标准格式并保存匹配标签
                self.results = []
                self.match_tags = {}  # 存储每个结果的匹配标签

                for idx, score, tags in hybrid_results:
                    self.results.append((idx, score))
                    self.match_tags[idx] = tags

                # 调试日志
                if self.results:
                    print("\n🎯 混合搜索结果:")
                    print(f"查询: '{raw_query}'")
                    print(f"总匹配文档数: {len(self.results)}")
                    print("\n前 5 个结果:")
                    for i, (idx, score) in enumerate(self.results[:5]):
                        doc = GLOBAL_CORPUS[idx]
                        text_preview = doc.get('text', '')[:80].replace('\n', ' ')
                        filename = doc.get('filename', 'unknown')
                        page = doc.get('page_number', '?')
                        tags = self.match_tags.get(idx, '')
                        print(f"  {i+1}. 分数: {score:.2f} | 标签: [{tags}] | "
                              f"{filename} p.{page}")
                        print(f"     预览: {text_preview}...")
                    print()

                self.current_result_index = 0

                if not self.results:
                    self.result_display.setText("No results found.")
                else:
                    self.show_current_chunk()
                self.status_bar.clearMessage()
                return

            except Exception as e:
                self.result_display.setText(f"Hybrid search error: {e}")
                import traceback
                traceback.print_exc()
                return

        # ---------------------------------------------------------------------
        # CASE 1: "Simple text search"
        # ---------------------------------------------------------------------
        if search_method == "精确文本搜索":
            quoted_phrases, unquoted_words = parse_simple_search_query(raw_query)

            quoted_phrases_norm = [remove_accents(p.lower()) for p in quoted_phrases]
            unquoted_words_norm = [remove_accents(w.lower()) for w in unquoted_words]

            matches = []
            for idx, doc in enumerate(GLOBAL_CORPUS):
                doc_text_norm = remove_accents(doc['text'].lower()) if 'text' in doc else ""
                # Must contain all quoted multi-word substrings
                if not all(phrase in doc_text_norm for phrase in quoted_phrases_norm):
                    continue
                # Must contain all unquoted words
                if not all(word in doc_text_norm for word in unquoted_words_norm):
                    continue
                matches.append(idx)

            self.results = [(doc_id, 1.0) for doc_id in matches]
            self.current_result_index = 0
            if not self.results:
                self.result_display.setText("No results found.")
            else:
                self.show_current_chunk()
            self.status_bar.clearMessage()
            return

        # ---------------------------------------------------------------------
        # CASE 2: "Embeddings search" - 使用优化的搜索引擎
        # ---------------------------------------------------------------------
        if search_method == "语义搜索 (Embeddings)":
            if not self.embeddings_present:
                self.result_display.setText("No .emb files found. Reverting to BM25 search.")
                self.search_method_combo.setCurrentText("BM25 关键词")
                return

            if not self.search_engine or not self.search_engine.embedding_searcher:
                self.result_display.setText("Embedding searcher not available. Reverting to BM25 search.")
                self.search_method_combo.setCurrentText("BM25 关键词")
                return

            # 使用优化的搜索引擎（查询向量归一化，长度惩罚 0.3，单次排序）
            self.results = self.search_engine.search(
                raw_query,
                method="embedding",
                max_results=MAX_SEARCH_RESULTS,
                length_penalty_exp=0.3  # ← 优化：从 0.5 降低到 0.3
            )

            # 调试日志：显示前5个结果的详细信息
            if self.results:
                print("\n🔍 优化后的 Embedding 搜索:")
                print(f"查询: '{raw_query}'")
                print(f"总匹配文档数: {len(self.results)}")
                print(f"长度惩罚: 0.3 (优化后，原为 0.5)")
                print("\n前 5 个结果:")
                for i, (idx, score) in enumerate(self.results[:5]):
                    doc = GLOBAL_CORPUS[idx]
                    text_preview = doc.get('text', '')[:100].replace('\n', ' ')
                    text_length = len(doc.get('text', ''))
                    filename = doc.get('filename', 'unknown')
                    page = doc.get('page_number', '?')
                    print(f"  {i+1}. 分数: {score:.4f} | 长度: {text_length:5d} | "
                          f"{filename} p.{page}")
                    print(f"     预览: {text_preview}...")
                print()

            self.current_result_index = 0

            if not self.results:
                self.result_display.setText("No results found.")
            else:
                self.show_current_chunk()
            self.status_bar.clearMessage()
            return

        # ---------------------------------------------------------------------
        # CASE 3: "BM25 substring"
        # ---------------------------------------------------------------------
        if search_method == "BM25 前缀匹配":
            # Parse positive & negative keywords
            raw_terms = raw_query.split()
            positive_keywords = []
            negative_keywords = []
            for term in raw_terms:
                norm_term = remove_accents(term.lower())
                if norm_term.startswith('-') and len(norm_term) > 1:
                    negative_keywords.append(norm_term[1:])
                elif not norm_term.startswith('-'):
                    positive_keywords.append(norm_term)
            if not positive_keywords:
                self.result_display.setText("Search requires at least one positive keyword.")
                return

            # Prepare corpus statistics
            N = len(GLOBAL_CORPUS)
            doc_term_freqs = []
            doc_lengths = []
            for doc in GLOBAL_CORPUS:	
                text = doc.get('text', '')
                norm_text = remove_accents(text.lower())
                terms = norm_text.split()
                doc_lengths.append(len(terms))
                doc_term_freqs.append(Counter(terms))
            avg_doc_len = sum(doc_lengths) / N if N > 0 else 0.0

            # Precompute document frequencies for each positive keyword
            dfs = {}
            for pos_kw in positive_keywords:
                dfs[pos_kw] = sum(
                    1 for freq in doc_term_freqs
                    if any(term.startswith(pos_kw) for term in freq)
                )

            # BM25 parameters
            k1 = 1.5
            b = 0.75

            results_with_flag = []
            # Evaluate each document
            for doc_id, doc in enumerate(GLOBAL_CORPUS):
                freqs = doc_term_freqs[doc_id]
                doc_len = doc_lengths[doc_id]

                # Exclude if any negative keyword matches
                if negative_keywords and any(
                    any(term.startswith(neg_kw) for term in freqs)
                    for neg_kw in negative_keywords
                ):
                    continue

                # Check presence of positive keywords
                contains_all = True
                found = []
                for pos_kw in positive_keywords:
                    if any(term.startswith(pos_kw) for term in freqs):
                        found.append(pos_kw)
                    else:
                        contains_all = False
                if not found:
                    continue  # need at least one match

                # Compute BM25‐style score with prefix TF/IDF
                bm25_score = 0.0
                for pos_kw in found:
                    tf = sum(cnt for term, cnt in freqs.items() if term.startswith(pos_kw))
                    df = dfs.get(pos_kw, 0)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    num = idf * tf * (k1 + 1)
                    den = tf + k1 * (1 - b + b * (doc_len / avg_doc_len if avg_doc_len > 0 else 1))
                    if den > 0:
                        bm25_score += num / den

                # Compute a proximity‐enhanced original_score
                count_score = sum(
                    cnt for term, cnt in freqs.items()
                    for pos_kw in found if term.startswith(pos_kw)
                )
                prox_score = 0.0
                if len(found) > 1:
                    text_norm = remove_accents(doc.get('text', '').lower())
                    positions = []
                    for pos_kw in found:
                        pattern = r'\b' + re.escape(pos_kw)
                        for m in re.finditer(pattern, text_norm):
                            positions.append(m.start())
                    if len(positions) >= 2:
                        positions.sort()
                        min_gap = min(
                            positions[i+1] - positions[i]
                            for i in range(len(positions)-1)
                        )
                        norm_len = max(len(text_norm), 1)
                        prox_score = max(0.0, 1.0 - (min_gap / norm_len)) * len(found)
                original_score = count_score + prox_score

                combined = 0.3 * original_score + 0.7 * bm25_score
                results_with_flag.append((contains_all, combined, doc_id))

            if not results_with_flag:
                self.result_display.setText("No matching documents found.")
                return

            # Sort by whether all keywords matched, then by score
            results_with_flag.sort(key=lambda x: (x[0], x[1]), reverse=True)
            # Store only (doc_id, score)
            self.results = [(doc_id, score) for (_, score, doc_id) in results_with_flag]
            self.results = self.results[:MAX_SEARCH_RESULTS]           # ← modified
            self.current_result_index = 0
            self.show_current_chunk()
            self.status_bar.clearMessage()
            return

        # ---------------------------------------------------------------------
        # CASE 4: "BM25"
        # ---------------------------------------------------------------------
        if GLOBAL_BM25_MODEL is None:
            self.result_display.setText("No BM25 model is available.")
            return

        tokenized_query = bm25s.tokenize(raw_query, stopwords="en")
        results, scores = GLOBAL_BM25_MODEL.retrieve(tokenized_query, k=len(GLOBAL_CORPUS))
        bm25_ranking = [(doc_idx, scores[0, i]) for i, doc_idx in enumerate(results[0])]
        bm25_ranking.sort(key=lambda x: x[1], reverse=True)
        truncated_ranking = bm25_ranking[:MAX_SEARCH_RESULTS]

        # Rerank if requested
        if method == "No reranking":
            final_ranking = truncated_ranking
        elif method == "Minimal span-based scoring":
            final_ranking = rerank_minimal_span(truncated_ranking, self.query_terms)
        elif method == "Exact text search":
            final_ranking = rerank_exact_text(truncated_ranking, raw_query)
        elif method == "Embeddings rerank":
            if not FASTEMBED_AVAILABLE:
                self.result_display.setText("Fastembed not installed. Cannot do embeddings rerank.")
                final_ranking = truncated_ranking
            else:
                final_ranking = self.rerank_with_embeddings(truncated_ranking, raw_query)
        else:
            final_ranking = truncated_ranking

        self.results = final_ranking
        self.current_result_index = 0

        if not self.results:
            self.result_display.setText("No results found.")
        else:
            self.show_current_chunk()

        self.status_bar.clearMessage()

    def rerank_with_embeddings(self, top_docs, query_phrase):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

        documents = [GLOBAL_CORPUS[doc_id]['text'] for (doc_id, _score) in top_docs]
        scores = list(encoder.rerank(query_phrase, documents))

        doc_scores = []
        for (doc_id, _bm25score), embed_score in zip(top_docs, scores):
            doc_scores.append((doc_id, embed_score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores

    def show_current_chunk(self):
        global GLOBAL_CORPUS
        if not self.results:
            self.result_display.setText("No results found.")
            return

        doc_id, score = self.results[self.current_result_index]
        chunk_data = GLOBAL_CORPUS[doc_id]

        # We'll highlight them in the chunk text (if any).
        text_to_display = chunk_data.get('text', "")
        highlighted_chunk = self.highlight_query_terms(text_to_display)

        # Get match tags if available (for hybrid search)
        match_tags_html = ""
        if hasattr(self, 'match_tags') and doc_id in self.match_tags:
            tags = self.match_tags[doc_id]
            # 为不同的匹配类型添加彩色标签
            tag_colors = {
                "精确匹配": "#28a745",  # 绿色 - 最高优先级
                "部分匹配": "#007bff",  # 蓝色
                "语义相关": "#6f42c1",  # 紫色
                "关键词": "#fd7e14"     # 橙色
            }
            tag_badges = []
            for tag in tags.split(','):
                tag = tag.strip()
                color = tag_colors.get(tag, "#6c757d")  # 默认灰色
                tag_badges.append(
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 2px 8px; border-radius: 3px; margin-right: 5px; '
                    f'font-size: 11px; font-weight: bold;">{tag}</span>'
                )
            match_tags_html = f"<b>匹配方式:</b> {''.join(tag_badges)}<br>"

        self.result_display.setHtml(
            f'<div style="font-family: Arial, sans-serif;">'
            f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">'
            f'<b style="color: #495057;">结果 {self.current_result_index + 1} / {len(self.results)}</b><br>'
            f'<b style="color: #495057;">文件:</b> <span style="color: #212529;">{os.path.basename(chunk_data.get("filename",""))}</span><br>'
            f'<b style="color: #495057;">页码:</b> <span style="color: #212529;">{chunk_data.get("page_number","")}</span><br>'
            f'{match_tags_html}'
            f'<b style="color: #495057;">相关度:</b> <span style="color: #007bff; font-weight: bold;">{score:.4f}</span>'
            f'</div>'
            f'<div style="padding: 10px; background-color: white; border-left: 3px solid #007bff;">'
            f'{highlighted_chunk}'
            f'</div>'
            f'</div>'
        )

        pdf_path = chunk_data.get('filename','')
        page_number = chunk_data.get('page_number', 1)
        if pdf_path and os.path.exists(pdf_path):
            self.display_pdf_page(pdf_path, page_number)
        else:
            self.result_display.append("<br><i>No PDF or page info available, or PDF not found.</i>")

    def highlight_query_terms(self, text):
        normalized_text = remove_accents(text)
        highlighted_text = normalized_text
        for term in self.query_terms:
            escaped_term = re.escape(term)
            # 使用更醒目的高亮颜色和样式
            highlighted_text = re.sub(
                rf'(?i)({escaped_term})',
                r'<span style="background-color: #ffeb3b; color: #000; font-weight: bold; '
                r'padding: 1px 2px; border-radius: 2px;">\1</span>',
                highlighted_text,
            )
        return highlighted_text

    # -------------------------------------------------------------------------
    # Zoom and font size
    # -------------------------------------------------------------------------
    def zoom_in(self):
        self.graphics_view.auto_fit_width = False  # Disable auto-fit when manually zooming
        self.scale_factor *= 1.2
        if self.graphics_view.current_pdf_path:
            self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

    def zoom_out(self):
        self.graphics_view.auto_fit_width = False  # Disable auto-fit when manually zooming
        self.scale_factor /= 1.2
        if self.graphics_view.current_pdf_path:
            self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

    def reset_zoom(self):
        self.graphics_view.auto_fit_width = True  # Re-enable auto-fit on reset
        self.auto_fit_pdf_width()

    def auto_fit_pdf_width(self):
        """Auto-fit PDF to the width of the graphics view."""
        if not self.graphics_view.current_pdf_path:
            return

        try:
            # Open document to get page dimensions
            doc = fitz.open(self.graphics_view.current_pdf_path)
            if len(doc) == 0:
                doc.close()
                return

            # Get the current page dimensions
            page = doc[self.graphics_view.current_page - 1]

            # Apply cropping if enabled to get actual content dimensions
            if self.crop_pdf_view_checkbox.isChecked():
                text_blocks = page.get_text("blocks")
                if text_blocks:
                    x_min = min(block[0] for block in text_blocks)
                    x_max = max(block[2] for block in text_blocks)
                    page_width = x_max - x_min
                else:
                    page_width = page.rect.width
            else:
                page_width = page.rect.width

            doc.close()

            # Calculate available width (accounting for scrollbar)
            view_width = self.graphics_view.viewport().width()
            margin = 20  # Small margin for aesthetics

            # Calculate scale factor to fit width
            base_dpi = 150
            zoom = base_dpi / 72
            rendered_width = page_width * zoom

            if rendered_width > 0:
                target_scale = (view_width - margin) / rendered_width
                self.scale_factor = max(0.1, min(5.0, target_scale))  # Clamp between 0.1x and 5x
            else:
                self.scale_factor = 1.0

            # Re-render the current page with new scale
            if self.graphics_view.current_pdf_path:
                self.display_pdf_page(self.graphics_view.current_pdf_path, self.graphics_view.current_page)

        except Exception as e:
            print(f"Error auto-fitting PDF width: {e}")
            self.scale_factor = 1.0

    def show_next_chunk(self):
        if not self.results:
            return
        self.current_result_index = (self.current_result_index + 1) % len(self.results)
        self.show_current_chunk()

    def show_previous_chunk(self):
        if not self.results:
            return
        self.current_result_index = (self.current_result_index - 1) % len(self.results)
        self.show_current_chunk()

    def increase_font_size(self):
        self.font_size += 1
        self.result_display.setFont(QFont("Arial", self.font_size))
        self.query_input.setFont(QFont("Arial", self.font_size))

    def decrease_font_size(self):
        if self.font_size > 1:
            self.font_size -= 1
            self.result_display.setFont(QFont("Arial", self.font_size))
            self.query_input.setFont(QFont("Arial", self.font_size))

    def on_manage_folders(self):
        """
        Opens the FoldersDialog to manage the folders. 
        If the user clicks OK, we update 'folders.ini' and reload the corpus.
        """
        global FOLDERS_DB
        global GLOBAL_EMBED_MODEL
        global FASTEMBED_AVAILABLE

        dialog = FoldersDialog(folders_list=FOLDERS_DB.copy(), parent=self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            save_folders_database(dialog.folders_list)
            FOLDERS_DB = dialog.folders_list

            # Reload the corpus
            GLOBAL_CORPUS.clear()
            errors, status = load_corpus_and_initialize_bm25(FOLDERS_DB)
            self.result_display.clear()
            for err in errors:
                self.result_display.append(err)
            self.result_display.append(status)

            # Check if embeddings are present
            self.embeddings_present = any(('embedding' in doc) for doc in GLOBAL_CORPUS)
            if self.embeddings_present:
                success = init_embedding_model(lambda msg: self.result_display.append(msg))
                if success:
                    # ← added: 重新初始化优化的搜索引擎
                    self.search_engine = SearchEngine(
                        corpus=GLOBAL_CORPUS,
                        bm25_model=GLOBAL_BM25_MODEL,
                        embed_model=GLOBAL_EMBED_MODEL
                    )
                    self.result_display.append("Folders updated. Corpus and embeddings loaded.")
                else:
                    self.search_engine = None
                    self.result_display.append("Folders updated. Embeddings found, but fastembed is not installed.")
            else:
                self.search_engine = None
                self.result_display.append("Folders updated.")
        else:
            # user canceled => do nothing
            pass


###############################################################################
# Program entry point
###############################################################################
if __name__ == "__main__":
    # 启用输入法支持（修复 Rime 等输入法问题）
    # 如果环境变量未设置，尝试常见的输入法模块
    if 'QT_IM_MODULE' not in os.environ:
        # 检测常见的输入法框架
        if os.path.exists('/usr/bin/ibus'):
            os.environ['QT_IM_MODULE'] = 'ibus'
        elif os.path.exists('/usr/bin/fcitx') or os.path.exists('/usr/bin/fcitx5'):
            os.environ['QT_IM_MODULE'] = 'fcitx'

    app = QApplication(sys.argv)
    # 确保应用程序支持输入法
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    window = SearchApp()
    window.resize(1000, 700)  # a bit taller, since it's top/bottom
    window.show()
    sys.exit(app.exec_())
