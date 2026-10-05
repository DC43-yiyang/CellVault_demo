# CellVault 外部系统 Benchmark MVP 计划

> 执行日期：2026-09-23。冻结 CellVault 功能面；本轮只增加公平对照适配器、
> 统一测量协议和机器可读结果。下面保留原始设计依据。

## MVP-B0：冻结任务与证据协议

- [ ] `B0-01` 定义同一数据、同一 cohort、同一 feature axis 和同一结果终点的 JSON workload manifest。
- [ ] `B0-02` 统一记录 import、open、query、matrix access、compute、finalize、wall time、peak RSS、进程 I/O 和 artifact bytes。
- [ ] `B0-03` 用 cell IDs、feature IDs、矩阵和聚合结果指纹做跨系统正确性校验。
- [ ] `B0-04` 将一次性导入成本与重复查询成本分开，记录版本、线程和缓存声明。

## MVP-B1：AnnData 标准生态基线

- [ ] `B1-01` 实现 eager、backed 和可用时的 `read_lazy` 适配器。
- [ ] `B1-02` 所有路径计时到矩阵真正可用，而不是停在惰性对象创建。
- [ ] `B1-03` 对同一 workload 输出统一结果协议并通过参考指纹。

## MVP-B2：TileDB-SOMA 核心对照

- [ ] `B2-01` 将同一 H5AD 本地导入 SOMA Experiment，并单独报告导入时间与存储大小。
- [ ] `B2-02` 实现公开 API 的 direct cohort 工作流。
- [ ] `B2-03` 实现明确标注为 benchmark adapter 的 shared-union 工作流，不冒充原生自动联合调度。
- [ ] `B2-04` 验证选择、fan-out、重叠 cohort、聚合和混合结果与统一参考一致。

## MVP-B3：AnnSQL 核心 SQL 对照

- [ ] `B3-01` 本地构建 AnnSQL 数据库，单独报告导入时间、峰值内存和存储大小。
- [ ] `B3-02` 使用公开 SQL/API 实现可支持的查询、物化和聚合终点。
- [ ] `B3-03` 对无法合理扩展或不支持的任务输出结构化 capability status，不静默跳过。
- [ ] `B3-04` 验证所有已支持结果与统一参考一致。

## MVP-B4：Scarf 工作流对照

- [ ] `B4-01` 在隔离依赖环境中将同一输入转换为 Scarf Zarr，并记录导入成本。
- [ ] `B4-02` 使用 Scarf 公共接口实现其实际支持的选择性读取和多 subset 工作流。
- [ ] `B4-03` 将 Scarf 原生分析与访问层对照分开，并为不支持任务记录明确边界。

## MVP-B5：五类统一工作负载

- [ ] `B5-01` 单 cohort 查询与完整矩阵提取。
- [ ] `B5-02` 多 cohort fan-out。
- [ ] `B5-03` 重叠 cohort／共享对照。
- [ ] `B5-04` 多层级 `sum`／`mean`／`count_nonzero` 聚合。
- [ ] `B5-05` 聚合＋局部物化混合任务。

## MVP-B6：正式运行与报告

- [ ] `B6-01` 在本地同一数据和线程设置下随机方法顺序、独立进程重复测量。
- [ ] `B6-02` 保存 raw runs、median/IQR、正确性结果、能力矩阵和失败原因。
- [ ] `B6-03` 保留现有 H5AD/Zarr/CellVault 内部消融，并与外部系统结果分层解释。
- [ ] `B6-04` 运行测试、编译、JSON 解析和补丁检查；更新综合报告。

---

## 原始设计依据

**当然要比，而且 TileDB-SOMA 和 AnnSQL 应该进入核心外部 benchmark，不能只放在讨论里。** 我上一条列出了工作负载和内部消融，却漏掉了“与哪些现有系统比较”这一维，这是我计划表述不完整。

**“功能开发可以冻结”的意思是：转入竞品适配与正式测量，不是只用现有内部结果就可以写论文。**

## 一、正式对照应当包括哪些？

按你冲击 Genome Biology 的目标，我建议固定以下四类外部对照：

| 对照                            | 在 benchmark 中的位置 | 重点比较什么                              |
| ----------------------------- | ---------------- | ----------------------------------- |
| **TileDB-SOMA**               | **核心系统对照**       | 元数据筛选、选择性矩阵访问、多 cohort 提取、聚合输入与资源占用 |
| **AnnSQL**                    | **核心 SQL 路线对照**  | SQL 查询、分组聚合、查询结果物化，以及数据库导入与存储成本     |
| **AnnData：eager／backed／lazy** | **标准生态基线**       | 合理使用现有 AnnData 时，CellVault 是否仍有增益   |
| **Scarf**                     | **重要工作流对照**      | 内存受限、重复子群分析和多模态工作流；按实际支持的重叠任务比较     |

这几个对照各自回答不同的问题：

**TileDB-SOMA：为什么不直接使用已有的单细胞数组数据库？** 它已经支持按 observation／feature 条件选择数据、访问关联矩阵，以及与 AnnData 互操作，与你的核心任务直接相关。([TileDB-SOMA][1])

**AnnSQL：为什么需要另一种 SQL 驱动的单细胞系统？** 它已经使用 DuckDB 组织 AnnData 数据，支持 SQL 查询、聚合和 AnnData 等格式输出，不能绕开。([GitHub][2])

**AnnData：优势是否只是来自对照没有合理使用现有能力？** 正式基线不能只有完整加载和 subset 保存重载，还应覆盖 `backed` 和适用的 `read_lazy` 路径；后者已经支持包括 `obs`、`var` 在内的惰性读取。([anndata][3])

**Scarf：你的工作流价值，相对于已有的内存高效分析系统是什么？** 它的当前文档已经描述 Zarr 存储、分块执行和多个细胞／特征选择共存。因此应进入比较计划，但文档能力必须与实际测试版本对应。([Scarf][4])

**不要求每个工具参与每个任务，但不能因为接口不同就排除它。比较的是能否完成同一研究任务，不是有没有和 CellVault 同名的函数。**

## 二、尤其是 TileDB-SOMA，应该怎么比？

我建议直接使用现有任务，不需要为了它再设计一套新应用：

| 任务                    | 各系统必须完成的相同输出             | 主要问题               |
| --------------------- | ------------------------ | ------------------ |
| **单 cohort 查询与提取**    | 相同细胞、相同基因及相同矩阵值          | 单次选择性访问，谁更有效？      |
| **多 lineage fan-out** | 多个指定群体的局部矩阵              | 多群体同时需要数据时，代价如何变化？ |
| **重叠 cohort／共享对照**    | 每个任务各自正确的成员与输出           | 重叠访问能否被有效复用？       |
| **多层级聚合**             | 相同分组的 sum／mean／count 等结果 | 不保留细胞级矩阵时，谁更有效？    |
| **聚合＋局部物化混合任务**       | 相同聚合结果和局部分析输入            | 在相同资源条件下，任务如何完成？   |

其中，**多层级聚合、重叠任务和混合执行，最能检验你当前提出的执行层贡献；单 cohort 访问则防止只挑对 CellVault 有利的场景。**

### 最关键的一点：不能只给 SOMA 写一个低效循环

我建议将 SOMA 对照分为两种实现，明确标注：

**第一种：基于公开接口的直接工作流。**
各 cohort 使用对应查询，完成各自的数据提取或聚合。

**第二种：合理优化的共享访问适配脚本。**
在任务语义允许时，合并所需细胞／特征范围，读取后分发到多个任务，或在同一批输入上计算多个聚合。

第二种是**我们为 benchmark 编写的优化适配层**，不能冒充 SOMA 原生自动提供的联合调度能力；但也不能禁止对照采用这种合理优化。

这样才能区分：

> **CellVault 相比独立查询循环的收益，和 CellVault 相比合理优化的其他后端的收益。**

假如 SOMA 加一个简单共享适配脚本就达到相近性能，这也是重要结果：它意味着共享执行思想具有通用价值，而 CellVault 的优势可能更多体现在自动组织任务、使用成本或某些工作负载上，不能直接宣称底层访问全面更快。

**现在没有 SOMA 的实测结果，不能预设 CellVault 会赢；正式 benchmark 就是要回答这个问题。**

## 三、外部比较和内部消融，要同时保留

你报告里的 H5AD independent／shared、Zarr independent／shared 和 CellVault joint，已经是有价值的**机制归因实验**。它们说明共享扫描的收益，以及通用执行器自身的额外成本。

但它们不能替代外部工具比较：

| 证据层           | 回答的问题                                                     |
| ------------- | --------------------------------------------------------- |
| **外部工具比较**    | 用户为什么选择 CellVault，而不是 TileDB-SOMA、AnnSQL、AnnData 或 Scarf？ |
| **内部消融与轻量控制** | CellVault 的收益具体来自哪个机制，而不是某个存储格式或不合理基线？                    |

**这两层都需要。** 我上一条只把第二层和应用任务展开了，没有明确写出第一层，不应该让你理解成“竞品不用比”。

## 四、现在应该补什么代码？

**补 benchmark adapters，不是继续扩张 CellVault 本体。**

适配器只负责把同一份任务定义翻译成各工具能执行的操作，并返回相同约定的结果。没有必要先给 CellVault 开发一个完整的 TileDB 后端。

正式测量时，我会固定三条原则：

**相同任务终点。** 需要矩阵时，计时必须到矩阵真正可用，而不是只创建一个惰性查询对象；只需要聚合时，允许对照直接聚合，不强迫它先生成完整 AnnData。涉及下游分析时，访问层比较使用相同算法和参数；Scarf 原生分析流程另列，避免把算法差异算成存储优势。

**相同资源与数据来源。** 首轮全部使用同一机器上的本地数据，不能把远程 SOMA 查询与本地 CellVault 混比。允许各工具使用合理的原生存储布局，但明确记录版本、线程、压缩和缓存条件。

**导入成本与重复使用成本分别报告。** 不能只算其他工具的转换开销、不算 CellVault 的；也不能让 AnnData 每个任务都重新加载，而 CellVault 一直复用已打开的数据。

---

**所以，我现在给你的明确执行顺序是：冻结 CellVault → 建立 AnnData、TileDB-SOMA、AnnSQL 的核心适配器 → 跑相同任务的外部比较 → 加入 Scarf 的对应工作流比较，同时保留现有内部消融。**

**TileDB-SOMA 不是可有可无的补充，而是这轮正式 benchmark 最应该优先加入的外部对照之一。**

[1]: https://tiledbsoma.readthedocs.io/en/latest/python-tiledbsoma-experiment.html "tiledbsoma.Experiment — TileDB-SOMA-Py documentation"
[2]: https://github.com/ArpiarSaundersLab/annsql "GitHub - ArpiarSaundersLab/annsql: The AnnSQL package enables SQL based queries on AnnData objects. · GitHub"
[3]: https://anndata.readthedocs.io/en/stable/generated/anndata.io.read_h5ad.html?utm_source=chatgpt.com "anndata.io.read_h5ad - Read the Docs"
[4]: https://scarf.readthedocs.io/ "Scarf — Scarf documentation"
