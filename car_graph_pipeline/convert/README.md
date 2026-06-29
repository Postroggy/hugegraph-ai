# 格式转换

本模块把 car-graph 的 raw 抽取结果转换为更适合 HugeGraph 导入的顶点和边文件。

## 为什么单独放转换层

抽取流程输出的是 raw 图谱事实；如果需要实体消歧，应先运行
`car_graph_pipeline.disambiguation`，再把消歧后的 `merged_entities.json` 和
`merged_relations.json` 交给本模块转换：

```text
extracted_entities/extracted_relations
-> disambiguation
-> merged_entities/merged_relations
-> convert
```

原始抽取输出形如：

```json
{
  "entities": [],
  "relations": []
}
```

HugeGraph 导入需要带 label、id、端点和 properties 的 vertex/edge 结构。
转换逻辑单独放在这里，可以避免把抽取质量控制和图存储序列化混在一起。

## 文件

- `to_hugegraph_v2.py`：较新的车型作用域转换器，适配点边带
  `vehicle_model` 的 schema。
- `to_hugegraph.py`：旧版无车型作用域转换器，保留作参考。

## 注意

这些转换脚本仍保留了原 `/Users/lzj/proj/car_graph/car_graph_pipeline`
工作区中的部分路径和假设。将它们作为 HugeGraph AI 下的生产 CLI 使用前，
需要先参数化输入/输出路径，并与最终采用的车型作用域 schema 对齐。
