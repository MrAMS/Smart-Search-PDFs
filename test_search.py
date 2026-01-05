#!/usr/bin/env python3
"""
搜索引擎测试工具

用法：
    # 基本搜索
    python test_search.py "教学直播系统"

    # 指定长度惩罚参数
    python test_search.py "教学直播系统" --penalty 0.3

    # 显示详细调试信息
    python test_search.py "教学直播系统" --penalty 0.3 --debug

    # 对比不同惩罚参数
    python test_search.py "教学直播系统" --compare 0.0 0.3 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastembed import TextEmbedding
    from search_engine import EmbeddingSearcher, SearchEngine
    FASTEMBED_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 fastembed: pip install fastembed")
    sys.exit(1)


def load_corpus(data_folder):
    """
    加载语料库（JSON + EMB 文件）

    Args:
        data_folder: 包含 PDF/JSON/EMB 文件的文件夹路径

    Returns:
        包含 text 和 embedding 的文档列表
    """
    print(f"📂 加载数据: {data_folder}")

    if not os.path.isdir(data_folder):
        print(f"❌ 文件夹不存在: {data_folder}")
        sys.exit(1)

    corpus = []
    json_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.json')])

    if not json_files:
        print(f"❌ 未找到 JSON 文件")
        sys.exit(1)

    for json_file in json_files:
        json_path = os.path.join(data_folder, json_file)
        emb_path = json_path.replace('.json', '.emb')

        # 加载 JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                pages = json.load(f)
        except Exception as e:
            print(f"⚠️  加载 {json_file} 失败: {e}")
            continue

        # 加载 EMB（如果存在）
        if os.path.exists(emb_path):
            try:
                with open(emb_path, 'r', encoding='utf-8') as f:
                    emb_pages = json.load(f)

                # 合并 embedding 到对应的页面
                for i, page in enumerate(pages):
                    if i < len(emb_pages) and 'embedding' in emb_pages[i]:
                        page['embedding'] = emb_pages[i]['embedding']
            except Exception as e:
                print(f"⚠️  加载 {emb_path} 失败: {e}")

        corpus.extend(pages)

    # 统计
    total_docs = len(corpus)
    emb_docs = sum(1 for doc in corpus if 'embedding' in doc)

    print(f"✅ 加载完成: {total_docs} 个页面")
    print(f"📊 其中 {emb_docs} 个页面有 embedding ({emb_docs/total_docs*100:.1f}%)\n")

    return corpus


def format_filename(filename):
    """缩短文件名以便显示"""
    if len(filename) > 40:
        return filename[:37] + "..."
    return filename


def search_and_display(
    query,
    corpus,
    embed_model,
    penalty=0.3,
    top_k=10,
    debug=False
):
    """
    执行搜索并显示结果

    Args:
        query: 查询文本
        corpus: 文档语料库
        embed_model: Embedding 模型
        penalty: 长度惩罚参数
        top_k: 显示的结果数量
        debug: 是否显示调试信息
    """
    print(f"🔍 搜索: '{query}'")
    print(f"⚙️  长度惩罚: {penalty}")
    print(f"📝 返回前 {top_k} 个结果\n")

    # 创建搜索器
    searcher = EmbeddingSearcher(embed_model, enable_cache=True)

    # 执行搜索
    results = searcher.search(
        query,
        corpus,
        max_results=top_k,
        length_penalty_exp=penalty,
        return_details=True
    )

    if not results:
        print("❌ 未找到结果")
        return

    print("=" * 80)

    for i, (idx, final_score, cosine_sim, length) in enumerate(results, 1):
        doc = corpus[idx]
        filename = doc.get('filename', 'unknown')
        page = doc.get('page_number', '?')
        text = doc.get('text', '')

        # 计算惩罚因子
        if length > 0 and penalty > 0:
            penalty_factor = length ** penalty
        else:
            penalty_factor = 1.0

        # 显示结果
        print(f"\n{i}. {format_filename(filename)} - 第 {page} 页")
        print(f"   最终分数: {final_score:.4f} | 余弦相似度: {cosine_sim:.4f}")
        print(f"   文本长度: {length} 字 | 惩罚因子: {penalty_factor:.2f}")

        if debug:
            # 显示文本预览
            text_preview = text[:150].replace('\n', ' ')
            print(f"   预览: {text_preview}...")

            # 显示分数计算细节
            print(f"   计算: {cosine_sim:.4f} / {penalty_factor:.2f} = {final_score:.4f}")

    print("\n" + "=" * 80)

    # 显示缓存统计
    cache_stats = searcher.get_cache_stats()
    print(f"\n💾 缓存统计: 命中 {cache_stats['hits']} 次, "
          f"未命中 {cache_stats['misses']} 次, "
          f"命中率 {cache_stats['hit_rate']*100:.1f}%")


def compare_penalties(query, corpus, embed_model, penalties, top_k=5):
    """
    对比不同长度惩罚参数的效果

    Args:
        query: 查询文本
        corpus: 文档语料库
        embed_model: Embedding 模型
        penalties: 要对比的惩罚参数列表
        top_k: 每个参数显示的结果数
    """
    print(f"🔍 查询: '{query}'")
    print(f"⚙️  对比参数: {penalties}")
    print(f"📝 每个参数显示前 {top_k} 个结果\n")
    print("=" * 80)

    searcher = EmbeddingSearcher(embed_model, enable_cache=True)

    for penalty in penalties:
        print(f"\n📊 长度惩罚 = {penalty}")
        print("-" * 80)

        results = searcher.search(
            query,
            corpus,
            max_results=top_k,
            length_penalty_exp=penalty,
            return_details=True
        )

        for i, (idx, final_score, cosine_sim, length) in enumerate(results, 1):
            doc = corpus[idx]
            filename = format_filename(doc.get('filename', 'unknown'))
            page = doc.get('page_number', '?')

            print(f"{i}. {filename:40s} p.{page:3d} | "
                  f"分数: {final_score:.4f} | "
                  f"余弦: {cosine_sim:.4f} | "
                  f"长度: {length:4d}")

    print("\n" + "=" * 80)
    print("\n💡 建议：选择结果最相关的惩罚参数")
    print("   - 惩罚 0.0: 无惩罚，长文档可能排名靠前")
    print("   - 惩罚 0.2-0.4: 轻度惩罚（推荐）")
    print("   - 惩罚 0.5+: 中度到重度惩罚，长文档被压制")


def main():
    parser = argparse.ArgumentParser(
        description='测试 Embedding 搜索引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本搜索
  python test_search.py "教学直播系统"

  # 指定惩罚参数
  python test_search.py "教学直播系统" --penalty 0.3

  # 显示调试信息
  python test_search.py "教学直播系统" --penalty 0.3 --debug

  # 对比不同参数
  python test_search.py "教学直播系统" --compare 0.0 0.3 0.5
        """
    )

    parser.add_argument('query', help='搜索查询')
    parser.add_argument('--folder', default='/home/santiego/Downloads/分布式/pdfs',
                       help='数据文件夹路径')
    parser.add_argument('--penalty', type=float, default=0.3,
                       help='长度惩罚指数（默认 0.3）')
    parser.add_argument('--top', type=int, default=10,
                       help='显示前 N 个结果（默认 10）')
    parser.add_argument('--debug', action='store_true',
                       help='显示调试信息（文本预览、分数计算等）')
    parser.add_argument('--compare', nargs='+', type=float,
                       help='对比多个惩罚参数（例如：--compare 0.0 0.3 0.5）')

    args = parser.parse_args()

    # 加载数据
    corpus = load_corpus(args.folder)

    # 初始化模型
    print("🔧 初始化 embedding 模型...")
    cache_dir = os.path.expanduser("~/.cache/fastembed")
    model_name = "jinaai/jina-embeddings-v2-base-zh"

    embed_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    print("✅ 模型加载完成\n")

    # 执行搜索
    if args.compare:
        # 对比模式
        compare_penalties(args.query, corpus, embed_model, args.compare, top_k=args.top)
    else:
        # 普通搜索模式
        search_and_display(
            args.query,
            corpus,
            embed_model,
            penalty=args.penalty,
            top_k=args.top,
            debug=args.debug
        )


if __name__ == '__main__':
    main()
