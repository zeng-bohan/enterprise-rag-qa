"""后端契约测试（工单 05）。

为什么单独成目录、单独打标记
--------------------------
`tests/` 根下的那套离线测试是刻意不碰真后端的（无 PG / 无 Redis / 不下载模型），
这个设计本身没错——它保证了任何人 clone 下来都能秒跑。但它留下了一个致命的盲区：
PGvectorStore、ChromaStore、PGKBRegistry、BGEEmbeddings 四个类在那里
**一次都没有被引用过**，于是 psycopg3 下写成 `?` 的占位符能一路活到 main 分支，
CI 全绿。

本目录补的就是这个盲区：两个后端跑**同一套断言**，谁不满足契约谁红。

跑法
----
    pytest tests -m contract            # 需要 Docker 起 postgres（docker compose up -d）
    pytest tests -m "not contract"      # 离线那套，不需要任何外部服务

CI 里两个 job 分开跑，保证没有 Docker 的贡献者仍然能跑离线套件。
"""
