# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Prompt templates for LLM-based evaluation metrics.

All prompts include few-shot examples derived from RAGAS and
GraphRAG-Benchmark (ICLR'26) reference implementations.

Two languages are supported:
- ``en`` (default): English prompts matching the original RAGAS / GraphRAG-Bench
  wording.
- ``zh``: Chinese prompts localized for Chinese automotive-manual evaluation.

Use :func:`get_prompt` to select the correct template for the current
``language`` setting.
"""

from typing import Dict

# ============================================================================
# Statement Decomposition (shared by Faithfulness, Answer Correctness)
# Reference: RAGAS StatementGeneratorPrompt + GraphRAG-Bench
# ============================================================================

STATEMENT_DECOMPOSE_PROMPT = """\
Given a question and an answer, break down each sentence in the answer into \
one or more fully understandable atomic statements. Ensure that no pronouns \
are used in any statement. Each statement should be a standalone factual claim \
that can be independently verified.

Example:
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to \
be one of the greatest and most influential physicists of all time. He was \
best known for developing the theory of relativity, he also made important \
contributions to the development of the theory of quantum mechanics.

Output:
{{
  "statements": [
    "Albert Einstein was a German-born theoretical physicist.",
    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
    "Albert Einstein was best known for developing the theory of relativity.",
    "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."
  ]
}}

Now do the same for:
Question: {question}
Answer: {answer}

Output format: Return a JSON object with a single key "statements" \
containing a list of strings, each being an atomic statement.
Return one statement for each atomic claim, preserve the statement text, and \
return no additional keys or text.
"""

_STATEMENT_DECOMPOSE_PROMPT_ZH = """\
给定一个问题和一个回答，请将回答中的每个句子拆分为一个或多个完整可理解的\
原子陈述。每个陈述必须是独立的、可被单独验证的事实性主张，且不能包含代词。

示例：
问题：阿尔伯特·爱因斯坦是谁，他最著名的是什么？
回答：他是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大、\
最具影响力的物理学家之一。他因提出相对论而闻名，还对量子力学的发展做出了重要贡献。

输出：
{{
  "statements": [
    "阿尔伯特·爱因斯坦是一位出生于德国的理论物理学家。",
    "阿尔伯特·爱因斯坦被广泛认为是有史以来最伟大、最具影响力的物理学家之一。",
    "阿尔伯特·爱因斯坦因提出相对论而闻名。",
    "阿尔伯特·爱因斯坦还对量子力学的发展做出了重要贡献。"
  ]
}}

现在请对以下内容做同样处理：
问题：{question}
回答：{answer}

输出格式：返回一个 JSON 对象，包含唯一的键 "statements"，其值为字符串列表，\
每个字符串是一个原子陈述。
每个原子主张都必须对应一个陈述，保留陈述原文，不得返回其他键或文本。
"""

# ============================================================================
# Faithfulness: NLI Statement Verification
# Reference: RAGAS NLIStatementPrompt + GraphRAG-Bench faithfulness
# ============================================================================

NLI_STATEMENT_PROMPT = """\
Your task is to judge the faithfulness of a series of statements based on \
a given context. For each statement return verdict "Yes" if it can be \
directly inferred from the context, or "No" if it cannot be directly inferred.

Example 1:
Context: John is a student at XYZ University. He is pursuing a degree in \
Computer Science. He is enrolled in several courses this semester, including \
Data Structures, Algorithms, and Database Management. John is a diligent \
student and spends a significant amount of time studying and completing \
assignments. He often stays late in the library to work on his projects.

Statements:
1. John is majoring in Biology.
2. John is taking a course on Artificial Intelligence.
3. John is a dedicated student.
4. John has a part-time job.

Output:
{{
  "verdicts": [
    {{"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science.", "verdict": "No"}},
    {{"statement": "John is taking a course on Artificial Intelligence.", "reason": "AI is not mentioned in the course list.", "verdict": "No"}},
    {{"statement": "John is a dedicated student.", "reason": "The context states he spends significant time studying and stays late at the library.", "verdict": "Yes"}},
    {{"statement": "John has a part-time job.", "reason": "No information about a part-time job in the context.", "verdict": "No"}}
  ]
}}

Example 2:
Context: Photosynthesis is a process used by plants, algae, and certain \
bacteria to convert light energy into chemical energy.

Statements:
1. Albert Einstein was a genius.

Output:
{{
  "verdicts": [
    {{"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated.", "verdict": "No"}}
  ]
}}

Now evaluate:
Context:
{context}

Statements:
{statements}

Output format: Return a JSON object with a single key "verdicts" \
containing a list of objects, each with "statement" (str), \
"reason" (str), and "verdict" ("Yes" or "No") keys.
Return exactly one verdict object for each numbered statement, preserve the \
statement text, and return no additional keys or text.
"""

_NLI_STATEMENT_PROMPT_ZH = """\
你的任务是根据给定的上下文，判断一系列陈述是否忠实于上下文。对于每个陈述，\
如果它能从上下文中直接推断出来，请返回 verdict 为 "Yes"；如果不能直接从上下文中\
推断出来，请返回 verdict 为 "No"。

示例 1：
上下文：约翰是 XYZ 大学的学生，正在攻读计算机科学学位。本学期他选修了多门课程，\
包括数据结构、算法和数据库管理。约翰是一名勤奋的学生，花费大量时间学习和完成作业。\
他经常待在图书馆里熬夜做项目。

陈述：
1. 约翰主修生物学。
2. 约翰正在修一门人工智能课程。
3. 约翰是一名用功的学生。
4. 约翰有一份兼职工作。

输出：
{{
  "verdicts": [
    {{"statement": "约翰主修生物学。", "reason": "上下文中明确说明约翰的专业是计算机科学。", "verdict": "No"}},
    {{"statement": "约翰正在修一门人工智能课程。", "reason": "课程列表中没有提到人工智能。", "verdict": "No"}},
    {{"statement": "约翰是一名用功的学生。", "reason": "上下文提到他花大量时间学习并经常在图书馆待到很晚。", "verdict": "Yes"}},
    {{"statement": "约翰有一份兼职工作。", "reason": "上下文中没有关于兼职工作的信息。", "verdict": "No"}}
  ]
}}

示例 2：
上下文：光合作用是植物、藻类和某些细菌将光能转化为化学能的过程。

陈述：
1. 阿尔伯特·爱因斯坦是一位天才。

输出：
{{
  "verdicts": [
    {{"statement": "阿尔伯特·爱因斯坦是一位天才。", "reason": "上下文与陈述无关。", "verdict": "No"}}
  ]
}}

现在请评估：
上下文：
{context}

陈述：
{statements}

输出格式：返回一个 JSON 对象，包含唯一的键 "verdicts"，其值为对象列表，\
每个对象包含 "statement"（字符串）、"reason"（字符串）和 "verdict"（"Yes" 或 "No"）。
必须为每条编号陈述恰好返回一个 verdict 对象，保留陈述原文，不得返回其他键或文本。
"""

# ============================================================================
# Answer Correctness: TP / FP / FN Classification
# Reference: RAGAS CorrectnessClassifier + GraphRAG-Bench answer_accuracy
# ============================================================================

CORRECTNESS_CLASSIFY_PROMPT = """\
Given a ground truth and answer statements, analyze each statement and \
classify them in one of the following categories:
- TP (true positive): statements present in answer that are also directly \
supported by one or more statements in ground truth.
- FP (false positive): statements present in the answer but not directly \
supported by any statement in ground truth.
- FN (false negative): statements found in the ground truth but not present \
in answer.

Each statement can only belong to one of the categories. Provide a reason \
for each classification.

Completeness rules (MUST follow):
- Every candidate statement must appear in exactly one of TP or FP. Do not \
skip any candidate statement.
- Every reference statement must appear in exactly one of TP or FN. Do not \
skip any reference statement.

Example 1:
Question: What powers the sun and what is its primary function?
Candidate Answer Statements:
1. The sun is powered by nuclear fission, similar to nuclear reactors on Earth.
2. The primary function of the sun is to provide light to the solar system.

Reference Answer Statements:
1. The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.
2. This fusion process releases a tremendous amount of energy.
3. The energy provides heat and light, essential for life on Earth.
4. The sun's light plays a critical role in Earth's climate system.
5. Sunlight helps drive weather and ocean currents.

Output:
{{
  "tp": [{{"statement": "The primary function of the sun is to provide light to the solar system.", "reason": "Supported by ground truth mentioning the sun providing light."}}],
  "fp": [{{"statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.", "reason": "Incorrect - ground truth states nuclear fusion, not fission."}}],
  "fn": [
    {{"statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.", "reason": "Not mentioned in answer."}},
    {{"statement": "This fusion process releases a tremendous amount of energy.", "reason": "Not mentioned in answer."}},
    {{"statement": "The energy provides heat and light, essential for life on Earth.", "reason": "Only light is mentioned in answer."}},
    {{"statement": "The sun's light plays a critical role in Earth's climate system.", "reason": "Not mentioned in answer."}},
    {{"statement": "Sunlight helps drive weather and ocean currents.", "reason": "Not mentioned in answer."}}
  ]
}}

Example 2:
Question: What is the boiling point of water?
Candidate Answer Statements:
1. The boiling point of water is 100 degrees Celsius at sea level.

Reference Answer Statements:
1. The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.
2. The boiling point of water can change with altitude.

Output:
{{
  "tp": [{{"statement": "The boiling point of water is 100 degrees Celsius at sea level", "reason": "Directly supported by ground truth."}}],
  "fp": [],
  "fn": [{{"statement": "The boiling point of water can change with altitude.", "reason": "Not mentioned in the answer."}}]
}}

Now classify:
Question: {question}
Candidate Answer Statements:
{candidate_statements}

Reference Answer Statements:
{reference_statements}

Output format: Return a JSON object with keys "tp", "fp", "fn", each \
containing a list of objects with "statement" and "reason" fields.
Every candidate/reference statement must appear exactly once in the required \
buckets, and the response must contain no additional keys or text.
"""

_CORRECTNESS_CLASSIFY_PROMPT_ZH = """\
给定标准答案和候选答案中的若干陈述，请对每个陈述进行分析，并将其归入以下类别之一：
- TP（真正例）：候选答案中出现，并且能被标准答案中的陈述直接支持的陈述。
- FP（假正例）：候选答案中出现，但不能被标准答案中的任何陈述直接支持的陈述。
- FN（假反例）：标准答案中有，但候选答案中未出现的陈述。

每个陈述只能属于一个类别，并请注明分类理由。

完备性要求（必须遵守）：
- 每条候选答案陈述必须恰好出现在 TP 或 FP 中的一个。不得遗漏任何候选答案陈述。
- 每条标准答案陈述必须恰好出现在 TP 或 FN 中的一个。不得遗漏任何标准答案陈述。

示例 1：
问题：太阳的能量来源是什么，它的主要功能是什么？
候选答案陈述：
1. 太阳的能量来源是核裂变，类似于地球上的核反应堆。
2. 太阳的主要功能是为太阳系提供光。

标准答案陈述：
1. 太阳的能量来源是核聚变，氢原子聚变形成氦。
2. 这一聚变过程释放出巨大的能量。
3. 这些能量提供热和光，对地球上的生命至关重要。
4. 太阳的光在地球气候系统中起着关键作用。
5. 阳光有助于驱动天气和洋流。

输出：
{{
  "tp": [{{"statement": "太阳的主要功能是为太阳系提供光。", "reason": "标准答案中提到太阳提供光。"}}],
  "fp": [{{"statement": "太阳的能量来源是核裂变，类似于地球上的核反应堆。", "reason": "错误——标准答案指出是核聚变，而非核裂变。"}}],
  "fn": [
    {{"statement": "太阳的能量来源是核聚变，氢原子聚变形成氦。", "reason": "候选答案未提及。"}},
    {{"statement": "这一聚变过程释放出巨大的能量。", "reason": "候选答案未提及。"}},
    {{"statement": "这些能量提供热和光，对地球上的生命至关重要。", "reason": "候选答案只提到了光。"}},
    {{"statement": "太阳的光在地球气候系统中起着关键作用。", "reason": "候选答案未提及。"}},
    {{"statement": "阳光有助于驱动天气和洋流。", "reason": "候选答案未提及。"}}
  ]
}}

示例 2：
问题：水的沸点是多少？
候选答案陈述：
1. 在标准大气压下，水的沸点是 100 摄氏度。

标准答案陈述：
1. 在标准大气压下，水的沸点是 100 摄氏度（212 华氏度）。
2. 水的沸点会随海拔变化。

输出：
{{
  "tp": [{{"statement": "在标准大气压下，水的沸点是 100 摄氏度。", "reason": "被标准答案直接支持。"}}],
  "fp": [],
  "fn": [{{"statement": "水的沸点会随海拔变化。", "reason": "候选答案未提及。"}}]
}}

现在请分类：
问题：{question}
候选答案陈述：
{candidate_statements}

标准答案陈述：
{reference_statements}

输出格式：返回一个 JSON 对象，包含键 "tp"、"fp"、"fn"，每个键对应的值为\
包含 "statement" 和 "reason" 字段的对象列表。
每条候选/标准答案陈述必须恰好出现在规定的分类中一次，且不得返回其他键或文本。
"""

# ============================================================================
# Context Precision: Per-context relevance binary judgment
# Reference: RAGAS ContextPrecisionPrompt
# ============================================================================

CONTEXT_PRECISION_PROMPT = """\
Given a question and a ground truth answer, determine whether the following \
context passage is useful for correctly answering the question.

Example 1:
Question: What can you tell me about Albert Einstein?
Ground Truth: Albert Einstein, born on 14 March 1879, was a German-born \
theoretical physicist, widely held to be one of the greatest scientists of \
all time. He received the 1921 Nobel Prize in Physics.
Context: Albert Einstein (14 March 1879 - 18 April 1955) was a German-born \
theoretical physicist, widely held to be one of the greatest and most \
influential scientists of all time. Best known for developing the theory of \
relativity, he also made important contributions to quantum mechanics.

Output: {{"verdict": "Yes"}}

Example 2:
Question: What is the tallest mountain in the world?
Ground Truth: Mount Everest is the tallest mountain in the world.
Context: The Andes is the longest continental mountain range in the world, \
located in South America. It features many of the highest peaks in the \
Western Hemisphere.

Output: {{"verdict": "No"}}

Now evaluate:
Question: {question}
Ground Truth Answer: {ground_truth}
Context Passage: {context}

Output format: Return a JSON object with a single key "verdict" \
containing "Yes" or "No".
Return no explanation, additional keys, or additional text.
"""

_CONTEXT_PRECISION_PROMPT_ZH = """\
给定一个问题和对应的标准答案，请判断下面的上下文段落是否有助于正确回答该问题。

示例 1：
问题：你能告诉我关于阿尔伯特·爱因斯坦的什么信息？
标准答案：阿尔伯特·爱因斯坦，1879 年 3 月 14 日出生，是一位出生于德国的理论物理学家，\
被广泛认为是有史以来最伟大的科学家之一。他获得了 1921 年的诺贝尔物理学奖。
上下文：阿尔伯特·爱因斯坦（1879 年 3 月 14 日—1955 年 4 月 18 日）是一位出生于德国的理论物理学家，\
被广泛认为是有史以来最伟大、最具影响力的科学家之一。他因提出相对论而闻名，\
还对量子力学做出了重要贡献。

输出：{{"verdict": "Yes"}}

示例 2：
问题：世界上最高的山是什么？
标准答案：珠穆朗玛峰是世界上最高的山。
上下文：安第斯山脉是世界上最长的陆地山脉，位于南美洲。它拥有西半球许多最高的山峰。

输出：{{"verdict": "No"}}

现在请评估：
问题：{question}
标准答案：{ground_truth}
上下文段落：{context}

输出格式：返回一个 JSON 对象，包含唯一的键 "verdict"，其值为 "Yes" 或 "No"。
不得返回解释、其他键或其他文本。
"""

# ============================================================================
# Context Relevancy: Per-context graded relevance score (0-2)
# Reference: GraphRAG-Benchmark context_relevance.py
# ============================================================================

CONTEXT_RELEVANCE_PROMPT = """\
### Instructions
You are a world class expert designed to evaluate the relevance score of a \
Context in order to answer the Question.
Your task is to determine if the Context contains proper information to \
answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.

Scoring rules:
0. If the context does not contain any relevant information to answer the \
question, score 0.
1. If the context partially contains relevant information to answer the \
question, score 1.
2. If the context fully contains relevant information to answer the question, \
score 2.

Output format:
You must output strictly in JSON format with a single key "score".
No explanation, no additional text.
The score must be exactly one of 0, 1, or 2.

Example:
Question: What is the capital of France?
Context: Paris is the capital of France.
Output:
{{ "score": 2 }}

Now evaluate the following:
Question: {question}
Context: {context}
"""

_CONTEXT_RELEVANCE_PROMPT_ZH = """\
### 指令
你是一位顶尖专家，负责评估“上下文”对回答“问题”的相关性得分。
你的任务是判断上下文是否包含回答该问题的恰当信息。
请不要依赖你对该问题的先验知识，仅使用上下文和问题中明确写出的内容。

评分规则：
0. 如果上下文不包含任何回答问题的相关信息，得分为 0。
1. 如果上下文包含部分回答问题的相关信息，得分为 1。
2. 如果上下文包含完整回答问题的相关信息，得分为 2。

输出格式：
你必须严格以 JSON 格式输出，只包含一个键 "score"。
不要解释，不要附加任何其他文本。
得分必须严格为 0、1 或 2 之一。

示例：
问题：法国的首都是哪里？
上下文：巴黎是法国的首都。
输出：
{{ "score": 2 }}

现在请评估以下内容：
问题：{question}
上下文：{context}
"""

# ============================================================================
# Evidence Recall: Gold evidence support verification
# Reference: GraphRAG-Bench evidence_recall.py
# ============================================================================

EVIDENCE_RECALL_PROMPT = """\
### Task
You are given a list of evidences and a Context. For each evidence, determine \
whether it can be attributed to the Context.

Respond ONLY with a JSON object containing a "classifications" list. Each \
item should include:
- "statement": the exact evidence string
- "reason": a brief explanation (1 sentence)
- "attributed": 1 if the evidence can be attributed to the Context, otherwise 0
Return exactly one classification for each evidence item, preserve each \
statement verbatim, and return no additional keys or text.

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Evidence: ["Einstein received the Nobel Prize", "He was born in Germany"]

Output:
{{
  "classifications": [
    {{
      "statement": "Einstein received the Nobel Prize",
      "reason": "Matches context about Nobel Prize for physics in 1921.",
      "attributed": 1
    }},
    {{
      "statement": "He was born in Germany",
      "reason": "Birth information not present in context.",
      "attributed": 0
    }}
  ]
}}

### Actual Input
Context: "{context}"
Evidence: {evidence}
Question: "{question}" (for reference only)

### Your Response:
"""

_EVIDENCE_RECALL_PROMPT_ZH = """\
### 任务
给定一组证据和一个上下文，请判断每条证据是否可以从该上下文中得到归因。

请只返回一个 JSON 对象，其中包含 "classifications" 列表。每个条目包括：
- "statement"：证据的原文
- "reason"：简要说明（一句话）
- "attributed"：如果证据可以从上下文中得到归因则为 1，否则为 0
必须为每条证据恰好返回一个分类，保留每条证据原文，不得返回其他键或文本。

### 示例
输入：
上下文："爱因斯坦于 1921 年获得了诺贝尔物理学奖。"
证据：["爱因斯坦获得了诺贝尔奖", "他出生于德国"]

输出：
{{
  "classifications": [
    {{
      "statement": "爱因斯坦获得了诺贝尔奖",
      "reason": "与上下文中关于 1921 年获得诺贝尔物理学奖的信息一致。",
      "attributed": 1
    }},
    {{
      "statement": "他出生于德国",
      "reason": "上下文中没有关于出生地的信息。",
      "attributed": 0
    }}
  ]
}}

### 实际输入
上下文："{context}"
证据：{evidence}
问题："{question}"（仅供参考）

### 你的回答：
"""


# ============================================================================
# Coverage Score: reference-fact coverage (GraphRAG-Benchmark coverage_score)
# ============================================================================

COVERAGE_FACT_EXTRACT_PROMPT = """\
You are given a question and a reference answer. Break down the reference answer \
into a list of distinct, independently verifiable factual statements (facts). \
Each fact should be a standalone claim that can be checked on its own.

Example:
Question: What causes seasons?
Reference Answer: "Seasonal changes result from Earth's axial tilt. This tilt \
causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

Now do the same for:
Question: {question}
Reference Answer: {reference}

Output format: Return a JSON object with a single key "facts" containing a list \
of strings, each being an independently verifiable factual statement.
Return no additional keys or text.
"""

_COVERAGE_FACT_EXTRACT_PROMPT_ZH = """\
给定一个问题和一个参考答案，请将参考答案拆分为一系列独立的、可单独验证的\
事实性陈述（facts）。每个事实必须是可独立核查的完整主张。

示例：
问题：季节更替是由什么引起的？
参考答案："季节变化由地球自转轴倾斜造成。这种倾斜导致不同半球接收到的阳光不同。"

输出：
{{
  "facts": [
    "季节变化由地球自转轴倾斜造成",
    "自转轴倾斜导致不同半球接收到不同的阳光"
  ]
}}

现在请对以下内容做同样处理：
问题：{question}
参考答案：{reference}

输出格式：返回一个 JSON 对象，包含唯一的键 "facts"，其值为字符串列表，\
每个字符串是一个可独立验证的事实性陈述。
不得返回其他键或文本。
"""

COVERAGE_CHECK_PROMPT = """\
For each factual statement from the reference, decide whether it is covered — \
i.e. can be inferred or is directly supported — by the response. \
Respond ONLY with a JSON object containing a "classifications" list. Each item \
must have:
- "statement": the exact fact from the reference
- "reason": a brief explanation of why the fact is or is not covered
- "attributed": 1 if the fact is covered by the response, 0 otherwise

Example:
Response: "Seasons are caused by Earth's tilted axis."
Reference Facts: ["Seasonal changes result from Earth's axial tilt", \
"The axial tilt causes different hemispheres to receive varying sunlight"]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "reason": "The response attributes seasons to the tilted axis.", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "reason": "This detail is not stated in the response.", "attributed": 0}}
  ]
}}

Now do the same for:
Question: {question}
Response: {response}
Reference Facts: {facts}

Output format: Return a JSON object with a single key "classifications".
Return exactly one classification for each reference fact, preserve each \
statement verbatim, and return no additional keys or text.
"""

_COVERAGE_CHECK_PROMPT_ZH = """\
对于参考答案中的每条事实性陈述，判断它是否被回答所覆盖（即能由回答推断出或\
被回答直接支持）。请只返回一个包含 "classifications" 列表的 JSON 对象，\
列表中每一项包含：
- "statement"：参考答案中的原事实
- "reason"：说明该事实是否被回答覆盖的简要理由
- "attributed"：若该事实被回答覆盖则为 1，否则为 0

示例：
回答："季节是由地球倾斜的自转轴造成的。"
参考事实：["季节变化由地球自转轴倾斜造成", "自转轴倾斜导致不同半球接收到不同的阳光"]

输出：
{{
  "classifications": [
    {{"statement": "季节变化由地球自转轴倾斜造成", "reason": "回答说明季节由倾斜的自转轴造成。", "attributed": 1}},
    {{"statement": "自转轴倾斜导致不同半球接收到不同的阳光", "reason": "回答未提及不同半球接收阳光的细节。", "attributed": 0}}
  ]
}}

现在请对以下内容做同样处理：
问题：{question}
回答：{response}
参考事实：{facts}

输出格式：返回一个 JSON 对象，包含唯一的键 "classifications"。
必须为每条参考事实恰好返回一个分类，保留事实原文，不得返回其他键或文本。
"""


# ============================================================================
# Entity Semantic Match (Graph Extraction — LLM-based)
# Judges whether each candidate vertex semantically matches any gold vertex.
# Reference: car33 评分规则.md §4.1 (entity normalization rules)
# ============================================================================

ENTITY_SEMANTIC_MATCH_PROMPT = """\
Your task is to judge whether candidate entities (from an automated KG extractor)
semantically match gold entities (from human annotation).

For each candidate entity, determine if it is semantically equivalent to any
gold entity of the SAME type. The gold entity list is the reference standard.

Matching rules:
- Entity TYPE (label) must match exactly. Component ≠ Function, Status ≠ Specification.
- Entity NAME allows: synonym normalization, abbreviation expansion, phrasing variation.
  Example: "制动液" matches "制动液检查/更换" (same core concept, different granularity).
- Each gold entity can be matched at most once.
- Each candidate entity can be matched at most once.
- If two candidate entities match the same gold entity, the first one wins.
- Special case: A warning-light Component in the gold that is expressed as
  Status in the candidate may still match if the semantic signal is identical
  (e.g., gold "Status(ABS故障警告灯)" ↔ candidate "Status(ABS系统故障指示)").

Return a JSON object with:
- "matches": list of objects with "candidate_index" and "gold_index" (0-indexed)
- "reasoning": brief explanation (1-2 sentences)
Indices must refer to the lists in the actual input, and each index may appear \
at most once. Return no additional keys or text.
The "reasoning" value is required even when no matches are found. Return no \
additional keys or text.

IMPORTANT: Do NOT match by position order. Candidate [0] does NOT necessarily match \
gold [0]. You MUST compare semantics across all pairs, regardless of index.

Example:
Gold entities (indexed):
[0] {{"label": "Status", "name": "ABS故障警告灯"}}
[1] {{"label": "Component", "name": "制动液"}}
[2] {{"label": "Specification", "name": "最高车速_205km/h"}}
[3] {{"label": "Component", "name": "轮胎"}}

Candidate entities (indexed):
[0] {{"label": "Component", "name": "轮胎气压"}}
[1] {{"label": "Status", "name": "ABS系统故障指示灯"}}
[2] {{"label": "Component", "name": "制动液检查/更换"}}
[3] {{"label": "Specification", "name": "发动机排量_1.5T"}}

Expected JSON:
{{
  "matches": [{{"candidate_index": 1, "gold_index": 0}}, {{"candidate_index": 2, "gold_index": 1}}],
  "reasoning": "Candidate[1] ABS系统故障指示灯 → gold[0] ABS故障警告灯 (semantic equivalent); \
candidate[2] 制动液检查/更换 → gold[1] 制动液 (core concept match). \
Candidate[0] 轮胎气压 does NOT match gold[3] 轮胎 — type is Component but 气压 adds unsupported detail making it a different entity. \
Candidate[3] 发动机排量_1.5T does NOT match gold[2] 最高车速_205km/h — completely different specifications."
}}

Now evaluate the following:

Gold entities:
{gold_entities}

Candidate entities:
{candidate_entities}

Return only the JSON object.
"""

_ENTITY_SEMANTIC_MATCH_PROMPT_ZH = """\
你的任务是判断候选实体（由自动 KG 抽取器生成）是否与标准答案实体（人工标注）在语义上等价。

针对每个候选实体，判断它是否与同类型的某个标准答案实体语义等价。标准答案实体列表是参考基准。

匹配规则：
- 实体类型（label）必须一致。Component ≠ Function，Status ≠ Specification。
- 实体名称允许：同义词归一、简称展开、表述变化。
  例如："制动液" 与 "制动液检查/更换" 可匹配（核心概念相同，粒度不同）。
- 每个标准答案实体最多被匹配一次。
- 每个候选实体最多被匹配一次。
- 如果两个候选实体匹配同一个标准答案实体，取第一个。
- 特殊情况：标准答案中的告警灯 Component 在候选答案中以 Status 表达，若语义信号一致可匹配。

返回 JSON 对象，包含：
- "matches"：包含 "candidate_index" 和 "gold_index" 的对象列表（从0开始索引）
- "reasoning"：简要说明（1-2 句中文）
索引必须对应实际输入列表，且每个索引最多出现一次。不得返回其他键或文本。
即使没有匹配项也必须填写非空的 "reasoning"，不得返回其他键或文本。

重要提示：不要按索引顺序匹配。候选 [0] 不一定匹配标准答案 [0]。必须跨所有组合进行语义比对，索引顺序不代表匹配关系。

示例：
标准答案实体（带索引）：
[0] {{"label": "Status", "name": "ABS故障警告灯"}}
[1] {{"label": "Component", "name": "制动液"}}
[2] {{"label": "Specification", "name": "最高车速_205km/h"}}
[3] {{"label": "Component", "name": "轮胎"}}

候选实体（带索引）：
[0] {{"label": "Component", "name": "轮胎气压"}}
[1] {{"label": "Status", "name": "ABS系统故障指示灯"}}
[2] {{"label": "Component", "name": "制动液检查/更换"}}
[3] {{"label": "Specification", "name": "发动机排量_1.5T"}}

期望输出：
{{
  "matches": [{{"candidate_index": 1, "gold_index": 0}}, {{"candidate_index": 2, "gold_index": 1}}],
  "reasoning": "候选[1] ABS系统故障指示灯→标准答案[0] ABS故障警告灯（语义等价）；候选[2] 制动液检查/更换→标准答案[1] 制动液（核心概念匹配）。候选[0] 轮胎气压与标准答案[3] 轮胎不匹配——虽然类型同为 Component，但「气压」引入了额外细节，属于不同实体。候选[3] 发动机排量_1.5T与标准答案[2] 最高车速_205km/h不匹配——规格含义完全不同。"
}}

现在请评估：

标准答案实体：
{gold_entities}

候选实体：
{candidate_entities}

只返回 JSON 对象。
"""


# ============================================================================
# Triple Semantic Match (Graph Extraction — LLM-based)
# Judges whether each candidate triple (edge) semantically matches any gold triple.
# Reference: car33 评分规则.md §4.2 (relation normalization rules)
# ============================================================================

TRIPLE_SEMANTIC_MATCH_PROMPT = """\
Your task is to judge whether candidate triples (from an automated KG extractor)
semantically match gold triples (from human annotation).

Each triple is expressed as: [source_entity_name] --relation_type--> [target_entity_name].

A triple match requires ALL of the following:
1. Source entity: semantically equivalent (same rules as entity matching)
2. Relation type: semantically equivalent. Allow synonym relations if clearly
   expressing the same relationship (e.g., HAS_STATUS ≈ SYSTEM_HAS_STATUS).
3. Target entity: semantically equivalent
4. Direction: must be identical (source→target, not reversed)

Matching rules:
- Each gold triple can be matched at most once.
- Each candidate triple can be matched at most once.
- If the relation type differs but the semantic meaning is identical
  (e.g., "HAS_STATUS" for a warning-light status carrier vs "HAS_COMPONENT"
  for a physical part), judge based on whether the factual claim is the same.
- Partial matches (e.g., source OK but relation wrong) are NOT counted as matches.

Return a JSON object with:
- "matches": list of objects with "candidate_index" and "gold_index" (0-indexed)
- "reasoning": brief explanation (1-2 sentences)
Indices must refer to the lists in the actual input, and each index may appear \
at most once. Return no additional keys or text.
The "reasoning" value is required even when no matches are found. Return no \
additional keys or text.

IMPORTANT: Do NOT match by position order. Candidate [0] does NOT necessarily match \
gold [0]. You MUST compare semantics across all pairs, regardless of index.

Example:
Gold triples:
[0] [制动液] --HAS_SPEC--> [容量:1L]
[1] [组合仪表] --HAS_STATUS--> [ABS故障警告灯点亮]
[2] [发动机] --HAS_COMPONENT--> [火花塞]

Candidate triples:
[0] [组合仪表显示屏] --HAS_STATUS--> [ABS故障指示灯点亮]
[1] [制动液] --HAS_COMPONENT--> [制动系统]
[2] [发动机总成] --HAS_COMPONENT--> [火花塞]

Expected JSON:
{{
  "matches": [{{"candidate_index": 0, "gold_index": 1}}, {{"candidate_index": 2, "gold_index": 2}}],
  "reasoning": "Candidate[0] 组合仪表显示屏--HAS_STATUS-->ABS故障指示灯点亮 matches gold[1] (source and target semantically equivalent, same relation). \
Candidate[2] 发动机总成--HAS_COMPONENT-->火花塞 matches gold[2] (发动机总成≈发动机, exact relation and target). \
Candidate[1] 制动液--HAS_COMPONENT-->制动系统 does NOT match gold[0] 制动液--HAS_SPEC-->容量:1L — relation and target are completely different factual claims."
}}

Now evaluate:

Gold triples:
{gold_triples}

Candidate triples:
{candidate_triples}

Return only the JSON object.
"""

_TRIPLE_SEMANTIC_MATCH_PROMPT_ZH = """\
你的任务是判断候选三元组（自动 KG 抽取结果）是否与标准答案三元组（人工标注）语义等价。

每个三元组表示为：[源实体名] --关系类型--> [目标实体名]。

一个三元组匹配必须同时满足以下全部条件：
1. 源实体：语义等价（与实体匹配规则相同）
2. 关系类型：语义等价。允许等价关系映射（如 HAS_STATUS ≈ SYSTEM_HAS_STATUS）。
3. 目标实体：语义等价
4. 方向：必须一致（源→目标，不可反向）

匹配规则：
- 每个标准答案三元组最多被匹配一次。
- 每个候选三元组最多被匹配一次。
- 关系类型不同但语义完全一致时，以事实声明是否相同为准。
- 部分匹配（如源实体匹配但关系错误）不算命中。

返回 JSON 对象，包含：
- "matches"：包含 "candidate_index" 和 "gold_index" 的对象列表（从0开始索引）
- "reasoning"：简要说明（1-2 句中文）
索引必须对应实际输入列表，且每个索引最多出现一次。不得返回其他键或文本。
即使没有匹配项也必须填写非空的 "reasoning"，不得返回其他键或文本。

重要提示：不要按索引顺序匹配。候选 [0] 不一定匹配标准答案 [0]。必须跨所有组合进行语义比对，索引顺序不代表匹配关系。

示例：
标准答案三元组：
[0] [制动液] --HAS_SPEC--> [容量:1L]
[1] [组合仪表] --HAS_STATUS--> [ABS故障警告灯点亮]
[2] [发动机] --HAS_COMPONENT--> [火花塞]

候选三元组：
[0] [组合仪表显示屏] --HAS_STATUS--> [ABS故障指示灯点亮]
[1] [制动液] --HAS_COMPONENT--> [制动系统]
[2] [发动机总成] --HAS_COMPONENT--> [火花塞]

期望输出：
{{
  "matches": [{{"candidate_index": 0, "gold_index": 1}}, {{"candidate_index": 2, "gold_index": 2}}],
  "reasoning": "候选[0] 组合仪表显示屏→ABS故障指示灯点亮与标准答案[1]语义等价（源实体和目标均等价，关系一致）；候选[2] 发动机总成→火花塞与标准答案[2]匹配（发动机总成≈发动机，关系和目标完全一致）；候选[1] 制动液--HAS_COMPONENT-->制动系统与标准答案[0] 制动液--HAS_SPEC-->容量:1L不匹配——关系类型和目标实体完全不同，事实声明不同。"
}}

现在请评估：

标准答案三元组：
{gold_triples}

候选三元组：
{candidate_triples}

只返回 JSON 对象。
"""


# ============================================================================
# Extraction Faithfulness (Graph Extraction — LLM-based, no GT required)
# Judges whether each candidate vertex/edge has textual support in the input.
# Reference: deepeval FaithfulnessMetric + ragas NLIStatementPrompt
# ============================================================================

EXTRACTION_FAITHFULNESS_PROMPT = """\
Your task is to judge whether each item in a knowledge-graph extraction result
is faithfully supported by the original input text.

For each vertex (entity) or edge (triple), determine if the factual claim it
makes can be directly or reasonably inferred from the input text.

Rules:
- verdict = "Yes": The item's factual content is clearly stated in or can be
  directly inferred from the input text.
- verdict = "No": The item's factual content is NOT supported by the input text
  (hallucination, over-extrapolation, or contradiction).
- If the input text mentions a concept but the item adds unsupported detail,
  verdict = "No".
- If the input text is empty or contains no relevant information for the item,
  verdict = "No".

Return a JSON object with:
- "verdicts": list of {{"idx": <int>, "verdict": "Yes" or "No", "reason": "<brief>"}}
Return exactly one verdict for every extraction item, using the displayed \
0-based index exactly once. Return no additional keys or text.

Example:
Input text:
"The vehicle uses DOT 4 brake fluid. The brake fluid reservoir is located in the engine compartment. Replace brake fluid every 2 years or 30,000 km."

Extraction items:
[0] {{"type": "vertex", "label": "Component", "name": "制动液"}}
[1] {{"type": "vertex", "label": "Specification", "name": "制动液更换周期:2年"}}
[2] {{"type": "edge", "label": "HAS_SPEC", "source": "制动液", "target": "制动液型号:DOT5"}}
[3] {{"type": "vertex", "label": "Component", "name": "发动机机油"}}

Expected JSON:
{{
  "verdicts": [
    {{"idx": 0, "verdict": "Yes", "reason": "Text mentions 'DOT 4 brake fluid', supporting the Component 制动液."}},
    {{"idx": 1, "verdict": "Yes", "reason": "Text states 'Replace brake fluid every 2 years', supporting the 2-year cycle."}},
    {{"idx": 2, "verdict": "No", "reason": "Text specifies DOT 4, but item claims DOT 5 — contradicts the source."}},
    {{"idx": 3, "verdict": "No", "reason": "Text never mentions engine oil — this is a hallucination."}}
  ]
}}

Now evaluate:

Input text:
{input_text}

Extraction items:
{items}

Return only the JSON object.
"""

_EXTRACTION_FAITHFULNESS_PROMPT_ZH = """\
你的任务是判断知识图谱抽取结果中的每一项是否有原始输入文本作为依据。

对每个顶点（实体）或边（三元组），判断它所声称的事实是否可以从输入文本中直接或合理推断出来。

规则：
- verdict = "Yes"：该项的事实内容在输入文本中有明确陈述或可直接推断。
- verdict = "No"：该项的事实内容在输入文本中没有依据（幻觉、过度推断或矛盾）。
- 若输入文本提到了某个概念但该项添加了无依据的细节，verdict = "No"。
- 若输入文本为空或不含该项相关信息，verdict = "No"。

返回 JSON 对象，包含：
- "verdicts"：[{{"idx": <编号>, "verdict": "Yes" 或 "No", "reason": "<简要原因>"}}, ...] 列表
必须为每个抽取项恰好返回一个 verdict，使用展示的从 0 开始索引且每个索引只出现一次，\
不得返回其他键或文本。

示例：
输入文本：
"本车使用 DOT 4 制动液。制动液储液罐位于发动机舱内。每 2 年或 30,000 公里更换制动液。"

抽取项：
[0] {{"type": "vertex", "label": "Component", "name": "制动液"}}
[1] {{"type": "vertex", "label": "Specification", "name": "制动液更换周期:2年"}}
[2] {{"type": "edge", "label": "HAS_SPEC", "source": "制动液", "target": "制动液型号:DOT5"}}
[3] {{"type": "vertex", "label": "Component", "name": "发动机机油"}}

期望输出：
{{
  "verdicts": [
    {{"idx": 0, "verdict": "Yes", "reason": "文中提到'DOT 4 制动液'，支持 Component 制动液。"}},
    {{"idx": 1, "verdict": "Yes", "reason": "文中说'每2年更换制动液'，支持2年更换周期。"}},
    {{"idx": 2, "verdict": "No", "reason": "文中的是DOT 4，该项声称DOT 5，与原文矛盾。"}},
    {{"idx": 3, "verdict": "No", "reason": "文中从未提及发动机机油，属于幻觉。"}}
  ]
}}

现在请评估：

输入文本：
{input_text}

抽取项：
{items}

只返回 JSON 对象。
"""


# ============================================================================
# Prompt selection helper
# ============================================================================

_PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "STATEMENT_DECOMPOSE_PROMPT": {
        "en": STATEMENT_DECOMPOSE_PROMPT,
        "zh": _STATEMENT_DECOMPOSE_PROMPT_ZH,
    },
    "NLI_STATEMENT_PROMPT": {
        "en": NLI_STATEMENT_PROMPT,
        "zh": _NLI_STATEMENT_PROMPT_ZH,
    },
    "CORRECTNESS_CLASSIFY_PROMPT": {
        "en": CORRECTNESS_CLASSIFY_PROMPT,
        "zh": _CORRECTNESS_CLASSIFY_PROMPT_ZH,
    },
    "CONTEXT_PRECISION_PROMPT": {
        "en": CONTEXT_PRECISION_PROMPT,
        "zh": _CONTEXT_PRECISION_PROMPT_ZH,
    },
    "CONTEXT_RELEVANCE_PROMPT": {
        "en": CONTEXT_RELEVANCE_PROMPT,
        "zh": _CONTEXT_RELEVANCE_PROMPT_ZH,
    },
    "EVIDENCE_RECALL_PROMPT": {
        "en": EVIDENCE_RECALL_PROMPT,
        "zh": _EVIDENCE_RECALL_PROMPT_ZH,
    },
    "COVERAGE_FACT_EXTRACT_PROMPT": {
        "en": COVERAGE_FACT_EXTRACT_PROMPT,
        "zh": _COVERAGE_FACT_EXTRACT_PROMPT_ZH,
    },
    "COVERAGE_CHECK_PROMPT": {
        "en": COVERAGE_CHECK_PROMPT,
        "zh": _COVERAGE_CHECK_PROMPT_ZH,
    },
    "ENTITY_SEMANTIC_MATCH_PROMPT": {
        "en": ENTITY_SEMANTIC_MATCH_PROMPT,
        "zh": _ENTITY_SEMANTIC_MATCH_PROMPT_ZH,
    },
    "TRIPLE_SEMANTIC_MATCH_PROMPT": {
        "en": TRIPLE_SEMANTIC_MATCH_PROMPT,
        "zh": _TRIPLE_SEMANTIC_MATCH_PROMPT_ZH,
    },
    "EXTRACTION_FAITHFULNESS_PROMPT": {
        "en": EXTRACTION_FAITHFULNESS_PROMPT,
        "zh": _EXTRACTION_FAITHFULNESS_PROMPT_ZH,
    },
}


def get_prompt(name: str, language: str = "en") -> str:
    """Return the prompt template identified by *name* for the given *language*.

    Supported names match the historical module-level constants:
    ``STATEMENT_DECOMPOSE_PROMPT``, ``NLI_STATEMENT_PROMPT``,
    ``CORRECTNESS_CLASSIFY_PROMPT``, ``CONTEXT_PRECISION_PROMPT``,
    ``CONTEXT_RELEVANCE_PROMPT``, ``EVIDENCE_RECALL_PROMPT``,
    ``COVERAGE_FACT_EXTRACT_PROMPT``, ``COVERAGE_CHECK_PROMPT``,
    ``ENTITY_SEMANTIC_MATCH_PROMPT``, ``TRIPLE_SEMANTIC_MATCH_PROMPT``,
    ``EXTRACTION_FAITHFULNESS_PROMPT``.

    Args:
        name: Prompt constant name.
        language: ``"en"`` (default) or ``"zh"``.

    Returns:
        The prompt template string. Falls back to the English template if the
        requested language is unknown.
    """
    variants = _PROMPT_REGISTRY.get(name)
    if variants is None:
        raise KeyError(f"Unknown LLM-Judge prompt: {name}")
    return variants.get(language, variants["en"])
