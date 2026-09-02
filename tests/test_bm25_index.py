"""BM25 关键词检索：分词、排序、零分过滤与截断。

注意：BM25Okapi 的 IDF 对「出现在 ≥ 半数文档」的词会归零甚至为负，
`search` 的 `if s > 0` 过滤会把这些命中全部丢掉——真实语料（数百 chunk）不受影响，
但测试语料必须让查询词足够稀有（≤ 半数文档包含），这是小语料下的重要边界事实。
"""
from tests.conftest import mkdoc

_FILLERS = [
    mkdoc("财务报销须在费用发生后一个月内提交发票"),
    mkdoc("考勤：工作日上下班需要打卡，迟到按制度处理"),
    mkdoc("信息安全：不得将内部文档外发至个人邮箱"),
    mkdoc("差旅住宿标准：一线城市每晚限额三百元"),
    mkdoc("会议室使用：需提前在系统预约，会后清理白板"),
    mkdoc("办公用品领用：每月第一个工作日集中发放"),
]


def test_tokenize_strips_whitespace_and_drops_empty():
    from app.rag.bm25_index import BM25Index

    tokens = BM25Index.tokenize("  年假 请提前 3 天 申请。 ")
    assert all(t == t.strip() and t for t in tokens)


def test_build_and_search_ranks_relevant_doc_first():
    from app.rag.bm25_index import BM25Index

    target = mkdoc("员工请年假须提前三个工作日在 OA 系统申请")
    other = mkdoc("财务报销须在费用发生后一个月内提交发票")
    index = BM25Index(_FILLERS[:4] + [target, other])
    hits = index.search("年假 提前 申请", top_k=5)
    assert hits, "查询词在少数文档中出现时应有正分命中"
    assert hits[0][0] is target


def test_search_orders_by_score_descending():
    from app.rag.bm25_index import BM25Index

    strong = mkdoc("年假制度：年假申请与天数说明，年假不休可跨一年再休")
    weak = mkdoc("关于年假顺延的一条补充说明")
    index = BM25Index(_FILLERS + [strong, weak])
    hits = index.search("年假", top_k=5)
    assert len(hits) == 2
    assert hits[0][0] is strong
    assert hits[0][1] >= hits[1][1]


def test_zero_score_hits_are_excluded():
    from app.rag.bm25_index import BM25Index

    index = BM25Index(_FILLERS + [mkdoc("员工请年假须提前申请")])
    assert index.search("量子计算 芯片", top_k=5) == []


def test_top_k_limits_results():
    from app.rag.bm25_index import BM25Index

    targets = [mkdoc(f"年假条款第{i}条：年假说明{i}") for i in range(4)]
    fillers = [
        mkdoc(f"制度类别{i}号：日常行政事务规范说明条目{i}")
        for i in range(20)
    ]
    index = BM25Index(fillers + targets)
    assert len(index.search("年假", top_k=3)) == 3


def test_empty_index_returns_no_hits():
    from app.rag.bm25_index import BM25Index

    index = BM25Index()
    assert index.search("年假", top_k=5) == []
