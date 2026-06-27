# 汽车手册图谱抽取 few-shot 压缩参考

这些示例只作为抽取风格和 coverage 参照，不是当前文档事实。当前文档的 `vehicle_brand` / `vehicle_model` 最终以脚本传入值为准。

## 示例1：操作步骤类

输入片段：
“要完全开启天窗，向 OPEN 侧按压开关②。要完全关闭天窗，向 CLOSE 侧按压开关①。要向上倾斜天窗，在天窗全关闭状态下，向上按压开关③。车辆行驶或天窗正在关闭时，切勿允许任何人站立或把身体任何部位伸出开口处。”

应该抽：
- VehicleSystem：车身开闭系统
- Component：天窗、天窗开关
- Function：天窗开闭和倾斜控制
- Operation：操作天窗，steps 保留 OPEN/CLOSE/上按的方向和顺序，warnings 保留“切勿伸出身体”
- 关系：天窗 BELONGS_TO 车身开闭系统；天窗开关 ACTIVATES 天窗开闭和倾斜控制；Function OPERATED_BY Operation；Operation OPERATES_ON 天窗

Review 卡点：
- 不能只抽“天窗”，漏掉开关、功能和操作。
- 警告不能丢。
- 操作步骤不能合并成含糊一句。

## 示例2：规格/表格类

输入片段：
“冷态轮胎气压：前轮 240 kPa，后轮 240 kPa；满载时前轮 260 kPa，后轮 280 kPa。”

应该抽：
- Component：轮胎
- MaintenanceItem：检查轮胎气压
- Specification：前轮冷态胎压 240 kPa、后轮冷态胎压 240 kPa、满载前轮胎压 260 kPa、满载后轮胎压 280 kPa
- 关系：MaintenanceItem MAINT_HAS_SPEC 各胎压规格；轮胎 HAS_SPEC 各胎压规格

Review 卡点：
- 每个条件不同的数值应拆成独立 Specification。
- `condition_note` 必须保留“冷态/满载/前轮/后轮”等条件。
- 不要把多个数值压成一个泛泛的 `value_text`。

## 示例3：警告灯/状态/故障类

输入片段：
“如果制动系统警告灯在行驶中点亮，表示制动系统可能存在故障。请在安全地点停车，并联系授权维修站。继续驾驶可能导致制动性能下降。”

应该抽：
- Component：制动系统警告灯
- VehicleSystem：制动系统
- Status：制动系统警告灯点亮
- Fault：制动系统故障、制动性能下降风险
- Operation：安全停车并联系授权维修站
- 关系：制动系统警告灯 BELONGS_TO 制动系统；制动系统警告灯 HAS_STATUS 警告灯点亮；Status CAUSED_BY Fault；Fault AFFECTS 制动系统；Status RESOLVED_BY Operation 或 Fault FAULT_RESOLVED_BY Operation

Review 卡点：
- “警告灯”是 Component，“灯亮”是 Status，根因/风险是 Fault。
- 故障处理步骤要进入 Operation。

## 示例4：菜单/按钮路径类

输入片段：
“在中控屏选择：设置 > 车辆 > 灯光 > 伴我回家照明，可设置照明持续时间。”

应该抽：
- Component：中控屏
- VehicleSystem：车辆设置系统、灯光系统
- Function：伴我回家照明持续时间设置
- Operation：设置伴我回家照明时间，steps 保留完整菜单路径
- 关系：中控屏 ACTIVATES Function；Function OPERATED_BY Operation；Operation OPERATES_ON 中控屏；Function 可关联灯光系统

Review 卡点：
- 菜单路径不能丢层级。
- 不要把“设置 > 车辆 > 灯光”误抽成多个无意义零散组件。

## 示例5：油液/材料/保养类

输入片段：
“更换发动机机油时，请使用 SAE 0W-20 机油。请勿混用不同规格机油，否则可能损坏发动机。”

应该抽：
- VehicleSystem：发动机系统
- Component：发动机
- MaintenanceItem：更换发动机机油
- Material：SAE 0W-20 机油
- Specification：发动机机油规格 SAE 0W-20
- Fault：发动机损坏风险
- 关系：MaintenanceItem MAINT_REQUIRES Material；MaintenanceItem MAINT_HAS_SPEC Specification；Component HAS_SPEC Specification；Fault AFFECTS 发动机系统

Review 卡点：
- 机油是 Material，不是 Component。
- 保养动作是 MaintenanceItem，不能只抽物料。
- 警告风险要保留。

## 通用约束

- 不要在每个 chunk 重复输出 VehicleBrand、VehicleModel、HAS_MODEL。
- 不要为每个 Component/Function 强行输出 VehicleModel 的全局承接边；这些后处理会按车型属性补。
- 目录、页码索引、纯跳转表不要逐行造点造边。
- 每个 chunk 只输出对问答召回有价值的局部事实；长表格/长列表优先抽有含义的状态、故障、操作、规格，不要机械展开所有行。
