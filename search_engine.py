#!/usr/bin/env python3
"""
搜索引擎模块 - 与 GUI 解耦的核心搜索逻辑

这个模块包含优化的搜索算法，可以独立测试和调优。

主要改进：
1. 查询向量归一化 - 修复性能问题和一致性
2. 可调长度惩罚 - 默认 0.3（而非 0.5）
3. 单次排序 - 避免重复排序
4. 查询缓存 - 提升重复查询性能
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class EmbeddingSearcher:
    """
    优化的 Embedding 搜索引擎

    关键优化：
    - 查询向量归一化（文档向量已在生成时归一化）
    - 可调长度惩罚参数
    - 单次遍历和排序
    - 查询向量缓存
    """

    def __init__(self, embed_model, enable_cache: bool = True):
        """
        初始化 Embedding 搜索器

        Args:
            embed_model: FastEmbed TextEmbedding 模型实例
            enable_cache: 是否启用查询向量缓存
        """
        self.embed_model = embed_model
        self.enable_cache = enable_cache
        self._query_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        向量归一化到单位长度

        Args:
            vec: 输入向量

        Returns:
            归一化后的向量（范数为 1.0）
        """
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        else:
            return vec

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        获取查询的 embedding 向量（带缓存）

        Args:
            query: 查询文本

        Returns:
            归一化的查询向量
        """
        # 检查缓存
        if self.enable_cache and query in self._query_cache:
            self._cache_hits += 1
            return self._query_cache[query]

        # 生成查询向量
        self._cache_misses += 1
        query_emb = list(self.embed_model.query_embed(query))[0]

        # 归一化（关键优化！）
        query_emb = self.normalize_vector(query_emb)

        # 缓存
        if self.enable_cache:
            # 限制缓存大小
            if len(self._query_cache) >= 100:
                # 简单策略：清空缓存
                self._query_cache.clear()
            self._query_cache[query] = query_emb

        return query_emb

    def search(
        self,
        query: str,
        corpus: List[Dict[str, Any]],
        max_results: int = 50,
        length_penalty_exp: float = 0.3,  # 默认 0.3（而非 0.5）
        return_details: bool = False
    ) -> List[Tuple[int, float, ...]]:
        """
        执行 embedding 搜索

        Args:
            query: 查询文本
            corpus: 文档语料库（包含 'embedding' 和 'text' 字段）
            max_results: 返回的最大结果数
            length_penalty_exp: 长度惩罚指数（0 = 无惩罚，越大惩罚越重）
            return_details: 是否返回详细信息（余弦相似度、文本长度等）

        Returns:
            如果 return_details=False:
                [(doc_idx, final_score), ...]
            如果 return_details=True:
                [(doc_idx, final_score, cosine_sim, text_length), ...]

        算法流程：
            1. 获取并归一化查询向量（带缓存）
            2. 单次遍历计算所有文档的最终分数
            3. 单次排序并返回 top-k 结果
        """
        # 获取归一化的查询向量
        query_emb = self.get_query_embedding(query)

        # 单次遍历计算最终分数
        doc_scores = []

        for idx, doc in enumerate(corpus):
            # 跳过没有 embedding 的文档
            if 'embedding' not in doc:
                continue

            # 跳过空文本
            text = doc.get('text', '')
            if not text.strip():
                continue

            # 计算余弦相似度
            # 文档向量已归一化，查询向量也已归一化
            # 所以：cosine_sim = dot(doc_emb, query_emb)
            doc_emb = np.array(doc['embedding'])
            cosine_sim = float(np.dot(doc_emb, query_emb))

            # 应用长度惩罚
            length = len(text)
            if length > 0 and length_penalty_exp > 0:
                penalty_factor = length ** length_penalty_exp
                final_score = cosine_sim / penalty_factor
            else:
                # 无惩罚
                final_score = cosine_sim

            # 记录结果
            if return_details:
                doc_scores.append((idx, final_score, cosine_sim, length))
            else:
                doc_scores.append((idx, final_score))

        # 单次排序（按最终分数降序）
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top-k
        return doc_scores[:max_results]

    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息

        Returns:
            {'hits': int, 'misses': int, 'size': int, 'hit_rate': float}
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._query_cache),
            'hit_rate': hit_rate
        }

    def clear_cache(self):
        """清空查询缓存"""
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class SearchEngine:
    """
    统一搜索引擎接口

    支持多种搜索方法：
    - embedding: 语义向量搜索
    - bm25: 关键词搜索（待实现）
    - hybrid: 混合搜索（待实现）
    """

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        bm25_model = None,
        embed_model = None
    ):
        """
        初始化搜索引擎

        Args:
            corpus: 文档语料库
            bm25_model: BM25 模型实例（可选）
            embed_model: FastEmbed 模型实例（可选）
        """
        self.corpus = corpus
        self.bm25_model = bm25_model
        self.embed_model = embed_model

        # 初始化 embedding 搜索器
        if embed_model is not None:
            self.embedding_searcher = EmbeddingSearcher(embed_model)
        else:
            self.embedding_searcher = None

    def search(
        self,
        query: str,
        method: str = "embedding",
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        统一搜索接口

        Args:
            query: 查询文本
            method: 搜索方法 ("embedding", "bm25", "hybrid")
            **kwargs: 方法特定的参数

        Returns:
            [(doc_idx, score), ...]
        """
        if method == "embedding":
            if self.embedding_searcher is None:
                raise ValueError("Embedding model not available")

            # 调用 embedding 搜索
            results = self.embedding_searcher.search(
                query,
                self.corpus,
                **kwargs
            )

            # 统一返回格式：(idx, score)
            return [(r[0], r[1]) for r in results]

        elif method == "bm25":
            if self.bm25_model is None:
                raise ValueError("BM25 model not available")
            # TODO: 实现 BM25 搜索
            raise NotImplementedError("BM25 search not yet implemented in SearchEngine")

        elif method == "hybrid":
            # TODO: 实现混合搜索
            raise NotImplementedError("Hybrid search not yet implemented")

        else:
            raise ValueError(f"Unknown search method: {method}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取搜索引擎统计信息

        Returns:
            包含各种统计数据的字典
        """
        stats = {
            'corpus_size': len(self.corpus),
            'has_bm25': self.bm25_model is not None,
            'has_embedding': self.embed_model is not None,
        }

        # 添加 embedding 搜索器的缓存统计
        if self.embedding_searcher:
            stats['embedding_cache'] = self.embedding_searcher.get_cache_stats()

        return stats


# 便捷函数
def compare_length_penalties(
    query: str,
    corpus: List[Dict[str, Any]],
    embed_model,
    penalties: List[float] = [0.0, 0.2, 0.3, 0.5, 0.7],
    top_k: int = 5
) -> Dict[float, List[Tuple[int, float, float, int]]]:
    """
    对比不同长度惩罚参数的搜索结果

    Args:
        query: 查询文本
        corpus: 文档语料库
        embed_model: Embedding 模型
        penalties: 要测试的惩罚参数列表
        top_k: 每个参数返回的结果数

    Returns:
        {penalty: [(idx, final_score, cosine_sim, length), ...], ...}
    """
    searcher = EmbeddingSearcher(embed_model, enable_cache=True)
    results = {}

    for penalty in penalties:
        results[penalty] = searcher.search(
            query,
            corpus,
            max_results=top_k,
            length_penalty_exp=penalty,
            return_details=True
        )

    return results
