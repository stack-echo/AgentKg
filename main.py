import os
import asyncio
import aiohttp
import redis
import hashlib
import pickle
import uuid
import time
import json
import psutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from io import BytesIO

from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
from neo4j import GraphDatabase
from contextlib import asynccontextmanager
from docx import Document
import pdfplumber

from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
import requests
from cachetools import TTLCache

load_dotenv()

# ===========================================
# 全局变量和配置
# ===========================================

processing_status = {}
cache_manager = None

# 初始化 Redis 缓存
try:
    cache_manager = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    cache_manager.ping()
    print("✅ Redis缓存已连接")
except Exception as e:
    cache_manager = None
    print(f"⚠️ Redis未连接，缓存功能禁用: {e}")


# ===========================================
# 缓存工具函数
# ===========================================

def get_cache_key(prefix: str, content: str) -> str:
    """生成缓存键"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}:{content_hash}"


async def get_cached_result(cache_key: str):
    """获取缓存结果"""
    if not cache_manager:
        return None
    try:
        cached = cache_manager.get(cache_key)
        if cached:
            return pickle.loads(cached)
    except Exception as e:
        print(f"缓存读取失败: {e}")
    return None


async def set_cached_result(cache_key: str, result, expire_seconds: int = 3600):
    """设置缓存结果"""
    if not cache_manager:
        return
    try:
        cache_manager.setex(cache_key, expire_seconds, pickle.dumps(result))
    except Exception as e:
        print(f"缓存写入失败: {e}")


# ===========================================
# 结果保存函数
# ===========================================

RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)


def save_analysis_result(task_id: str, file_name: str, result_data: dict, processing_time: float):
    """保存分析结果到文件"""
    try:
        result_file = RESULTS_DIR / f"{task_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "task_id": task_id,
                "file_name": file_name,
                "processed_at": datetime.now().isoformat(),
                "processing_time": processing_time,
                "result": result_data
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 分析结果已保存: {result_file}")
    except Exception as e:
        print(f"⚠️ 保存分析结果失败: {e}")


# ===========================================
# 统一的 LLM 调用函数
# ===========================================

async def call_deepseek_api(prompt: str, task_type: str = "general", temperature: float = 0.1) -> Dict:
    """
    统一的 DeepSeek API 调用，带缓存
    """
    cache_key = get_cache_key(task_type, prompt)

    # 尝试从缓存获取
    cached_result = await get_cached_result(cache_key)
    if cached_result:
        print(f"🎯 使用 {task_type} 缓存结果")
        return cached_result

    # API 调用
    connector = aiohttp.TCPConnector(
        limit=10,
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    timeout = aiohttp.ClientTimeout(total=60, connect=10)

    async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json"
            }
    ) as session:
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"You are a {task_type} expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }

        try:
            async with session.post("https://api.deepseek.com/v1/chat/completions", json=data) as response:
                if response.status == 200:
                    result_data = await response.json()
                    content = result_data["choices"][0]["message"]["content"].strip()

                    # 解析 JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    result = json.loads(content) if content else {}

                    # 缓存结果
                    await set_cached_result(cache_key, result)
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"API错误 {response.status}: {error_text}")

        except Exception as e:
            print(f"{task_type} API调用错误: {e}")
            return {"error": str(e)}


# ===========================================
# 外部知识库类
# ===========================================

class ExternalKnowledge:
    def __init__(self, cache_ttl=3600):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.api_url = "https://www.wikidata.org/w/api.php"

    def query_wikidata_entity(self, entity_name, lang='zh'):
        cache_key = f"entity_{entity_name}_{lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = """
        SELECT ?item ?itemLabel ?itemDescription ?location ?instanceOf
        WHERE {
          ?item rdfs:label "%s"@%s .
          OPTIONAL { ?item wdt:P625 ?location . }
          OPTIONAL { ?item wdt:P31 ?instanceOf . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "%s,[en]" . }
        }
        LIMIT 1
        """ % (entity_name, lang, lang)
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(SPARQL_JSON)
        try:
            results = self.sparql.query().convert()
            if results['results']['bindings']:
                binding = results['results']['bindings'][0]
                result = {
                    'qid': binding['item']['value'].split('/')[-1],
                    'label': binding['itemLabel']['value'],
                    'description': binding.get('itemDescription', {}).get('value', ''),
                    'location': binding.get('location', {}).get('value', ''),
                    'instance_of': binding.get('instanceOf', {}).get('value', '').split('/')[-1]
                }
                self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Wikidata查询错误 {entity_name}: {str(e)}")
        return None

    def get_candidates(self, entity_name, lang='zh'):
        cache_key = f"candidates_{entity_name}_{lang}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            'action': 'wbsearchentities',
            'search': entity_name,
            'language': lang,
            'format': 'json',
            'limit': 5
        }
        try:
            response = requests.get(self.api_url, params=params).json()
            candidates = [item['id'] for item in response.get('search', [])]
            self.cache[cache_key] = candidates
            return candidates
        except Exception as e:
            print(f"Wikidata搜索错误 {entity_name}: {str(e)}")
            return [entity_name]

    def compute_similarity(self, candidate_qid, context, lang='zh'):
        cache_key = f"entity_{candidate_qid}_{lang}"
        if cache_key in self.cache:
            candidate_info = self.cache[cache_key]
        else:
            query = """
            SELECT ?itemLabel ?itemDescription
            WHERE {
              wd:%s rdfs:label ?itemLabel .
              OPTIONAL { wd:%s schema:description ?itemDescription . }
              SERVICE wikibase:label { bd:serviceParam wikibase:language "%s,[en]" . }
            }
            LIMIT 1
            """ % (candidate_qid, candidate_qid, lang)
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(SPARQL_JSON)
            try:
                results = self.sparql.query().convert()
                if results['results']['bindings']:
                    binding = results['results']['bindings'][0]
                    candidate_info = {
                        'qid': candidate_qid,
                        'label': binding['itemLabel']['value'],
                        'description': binding.get('itemDescription', {}).get('value', '')
                    }
                    self.cache[cache_key] = candidate_info
                else:
                    return 0.0
            except Exception as e:
                print(f"相似度查询错误 {candidate_qid}: {str(e)}")
                return 0.0

        description = candidate_info.get('description', '')
        context_words = set(context.lower().split())
        desc_words = set(description.lower().split())
        common_words = len(context_words.intersection(desc_words))
        return common_words / max(len(desc_words), 1)


# ===========================================
# 实体消歧类
# ===========================================

class EntityDisambiguation:
    def __init__(self):
        self.knowledge = ExternalKnowledge()

    def disambiguate(self, entity, context):
        try:
            candidates = self.knowledge.get_candidates(entity, 'zh')
            if len(candidates) == 1:
                return candidates[0]
            scores = [(c, self.knowledge.compute_similarity(c, context, 'zh')) for c in candidates]
            best_candidate = max(scores, key=lambda x: x[1], default=(entity, 0))[0]
            return best_candidate if best_candidate else entity
        except Exception as e:
            print(f"消歧错误 {entity}: {str(e)}")
            return entity


# ===========================================
# 多智能体调度器
# ===========================================

class MultiAgentScheduler:
    def __init__(self):
        self.agent_pool = {}

    def register_agent(self, agent):
        self.agent_pool[agent.name] = agent

    def schedule_task(self, task):
        try:
            task_type = task.get('type')
            agent = self.agent_pool.get(task_type)
            if not agent:
                raise ValueError(f"No agent for {task_type}")
            return agent.execute(task)
        except Exception as e:
            print(f"调度错误 {task.get('type')}: {str(e)}")
            return {"result": [], "error": f"Scheduling failed: {str(e)}"}


# ===========================================
# 智能体基类
# ===========================================

# 全局 OpenAI 客户端（单例模式）
_deepseek_client = None


def get_deepseek_client():
    """获取全局 DeepSeek 客户端"""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
    return _deepseek_client


class CustomAgent:
    def __init__(self, name, task_type):
        self.name = name
        self.task_type = task_type
        # 使用全局客户端，避免重复创建
        self.deepseek_client = get_deepseek_client()

    def execute(self, task):
        """统一的任务执行入口"""
        try:
            # 根据任务类型选择执行方法
            if self.task_type == "ner":
                prompt = self._build_ner_prompt(task)
            elif self.task_type == "relation":
                prompt = self._build_relation_prompt(task)
            elif self.task_type == "event":
                prompt = self._build_event_prompt(task)
            elif self.task_type == "preprocess":
                return self._execute_preprocess(task)
            elif self.task_type == "graph":
                return self._execute_graph_build(task)
            elif self.task_type == "semantic":
                return self._execute_semantic_enhancement(task)
            elif self.task_type == "standardize":
                return self._execute_entity_standardization(task)
            elif self.task_type == "complete_relations":
                return self._execute_relation_completion(task)
            elif self.task_type == "extract_document":
                return self._execute_document_extraction(task)
            else:
                return {"result": [], "error": "Unknown task type"}

            # 调用 DeepSeek API
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"You are a {self.task_type} agent. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()

            # 清理响应
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content) if content else {"result": [], "error": "Empty response"}
            return result

        except json.JSONDecodeError as e:
            print(f"{self.task_type} JSON解析错误: {str(e)}")
            return {"result": [], "error": f"Failed to parse response: {str(e)}"}
        except Exception as e:
            print(f"{self.task_type} 错误: {str(e)}")
            return {"result": [], "error": str(e)}

    def _build_ner_prompt(self, task):
        text = task.get('text', '')
        return f"""你是知识图谱构建专家。从文本中精确提取5类实体:人物、机构、地点、项目、概念。

文本: ```{text}```

提取规则:
1. **人物(PERSON)** - 必须是具体的人名
2. **机构(ORGANIZATION)** - 组织、公司、政府部门
3. **地点(LOCATION)** - 地理位置、设施
4. **项目(PROJECT)** - 计划、项目、行动
5. **概念(CONCEPT)** - 抽象概念、理论、协议、物品

关键要求:
- 保持原貌:不要改变实体名称,使用文中的完整形式
- 提取所有:每个符合条件的实体都要提取
- 避免泛化:绝不使用"人物"、"组织"等泛化词汇

输出格式(纯JSON,无其他内容):
{{
  "entities": [
    {{"word": "李维·克罗宁", "label": "PERSON"}},
    {{"word": "星耀联邦", "label": "ORGANIZATION"}}
  ]
}}"""

    def _build_relation_prompt(self, task):
        entities = task.get('entities', [])
        text = task.get('text', '')
        entities_str = ', '.join([e['word'] for e in entities])

        return f"""# 知识图谱关系抽取任务

    ## 第一步：理解文本
    文本内容：
    {text}

    ## 第二步：识别实体
    可用实体列表（主语和宾语必须从这里选择）：
    {entities_str}

    ## 第三步：提取完整关系三元组

    请按以下步骤操作：
    1. 找出文本中哪些实体之间存在关系
    2. 确定它们之间的动作或关系（必须是完整的动词短语，例如"提出"、"发布"、"属于"）
    3. 组织成 主语-谓语-宾语 格式

    ## 关键要求：
    ✓ 谓语必须是完整的动词或动词短语（2-4个字）
    ✓ 从原文中提取，保持语义完整
    ✓ 示例：
      - "国务院提出十四五规划" → predicate: "提出" ✓
      - "公司发布新产品" → predicate: "发布" ✓  
      - "员工属于某部门" → predicate: "属于" ✓

    ✗ 避免使用单字动词：
      - predicate: "提" ✗
      - predicate: "发" ✗
      - predicate: "属" ✗

    ## 输出格式
    直接输出JSON数组，不要任何解释：
    [
      {{"subject": "实体A", "predicate": "完整动词短语", "object": "实体B"}}
    ]"""

    def _build_event_prompt(self, task):
        text = task.get('text', '')
        return f"""你是事件抽取专家。从文本中识别关键事件，每个事件包含：时间、地点、参与者、动作。

    文本: ```{text}```

    抽取规则:
    1. 识别明确发生的事件（签约、会议、交易、协议等）
    2. 包含时间信息的优先
    3. 每个事件用一句话描述

    返回格式(纯JSON):
    {{
      "events": [
        "2020年9月1日，浙江申友四达与嘉善县第三幼儿园签订租赁合同",
        "合同约定租赁期限为5年",
        "租赁面积为3779.70平方米"
      ]
    }}"""

    def _execute_preprocess(self, task):
        try:
            text = task['text']
            cleaned_text = ' '.join(text.split())
            return {
                "cleaned_text": cleaned_text,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text)
            }
        except Exception as e:
            return {"cleaned_text": task.get('text', ''), "error": str(e)}

    def _execute_document_extraction(self, task):
        """提取文档文本"""
        try:
            file_content = task['file_content']
            file_name = task['file_name']
            extension = os.path.splitext(file_name)[1].lower()

            if extension == '.txt':
                for encoding in ['utf-8', 'gbk', 'utf-16']:
                    try:
                        text = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("无法解码文本文件")

            elif extension == '.docx':
                doc = Document(BytesIO(file_content))
                text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

            elif extension == '.pdf':
                text = ''
                with pdfplumber.open(BytesIO(file_content)) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ''
                text = text.strip()

            else:
                raise ValueError(f"不支持的文件类型: {extension}")

            return {"extracted_text": text}
        except Exception as e:
            return {"extracted_text": "", "error": str(e)}

    def _execute_graph_build(self, task):
        """构建知识图谱"""
        try:
            entities = task.get('entities', [])
            relations = task.get('relations', [])
            enhanced_entities = task.get('enhanced_entities', {})
            disambiguated_entities = task.get('disambiguated_entities', [])

            print(f"🔨 开始构建图谱:")
            print(f"  - 输入实体: {len(entities)}")
            print(f"  - 输入关系: {len(relations)}")

            nodes = []
            edges = []

            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
            )

            with driver.session() as session:
                # 创建实体节点
                for e in entities:
                    entity_name = e['word']
                    entity_label = e['label']
                    original_name = e.get('original', entity_name)

                    disambig = next((d for d in disambiguated_entities
                                     if d.get('original', d.get('standardized', '')) == entity_name), None)
                    qid = disambig['disambiguated'] if disambig else None

                    enhanced = enhanced_entities.get(original_name, {}) if isinstance(enhanced_entities, dict) else {}
                    description = enhanced.get('enhanced_description', '')
                    instance_of = enhanced.get('instance_of', '')

                    try:
                        session.run(
                            """
                            MERGE (n:Entity {name: $name})
                            ON CREATE SET n.qid = $qid, n.label = $entity_label, 
                                         n.description = $description, n.instance_of = $instance_of,
                                         n.original_name = $original_name, n.created_at = datetime()
                            ON MATCH SET n.qid = COALESCE(n.qid, $qid), 
                                        n.label = $entity_label,
                                        n.description = COALESCE(n.description, $description),
                                        n.updated_at = datetime()
                            """,
                            name=entity_name, qid=qid, entity_label=entity_label,
                            description=description, instance_of=instance_of,
                            original_name=original_name
                        )
                        nodes.append({"id": qid or entity_name, "label": entity_name, "type": entity_label})
                    except Exception as e:
                        print(f"⚠️ 创建节点失败 {entity_name}: {str(e)}")

                print(f"✓ 创建了 {len(nodes)} 个节点")

                # 创建关系
                # 创建关系
                for rel in relations:
                    if len(rel) == 3:
                        s, r, t = rel

                        # 验证关系类型不为空
                        if not r or not r.strip():
                            print(f"⚠️ 跳过空关系: {s} -> ??? -> {t}")
                            continue

                        # 验证源和目标实体是否存在
                        entity_names = [e['word'] for e in entities]
                        if s not in entity_names:
                            print(f"⚠️ 源实体不存在: {s}")
                            continue
                        if t not in entity_names:
                            print(f"⚠️ 目标实体不存在: {t}")
                            continue

                        is_inferred = task.get('completed_relations') and rel in task.get('completed_relations', [])

                        try:
                            result = session.run(
                                """
                                MATCH (a:Entity {name: $s})
                                MATCH (b:Entity {name: $t})
                                MERGE (a)-[rel:REL {type: $r}]->(b)
                                ON CREATE SET rel.inferred = $inferred, rel.created_at = datetime()
                                ON MATCH SET rel.updated_at = datetime()
                                RETURN count(rel) as cnt
                                """,
                                s=s, t=t, r=r, inferred=is_inferred
                            )

                            record = result.single()
                            if record and record['cnt'] > 0:
                                edges.append({"source": s, "target": t, "type": r, "inferred": is_inferred})
                            else:
                                print(f"⚠️ 关系创建失败: {s} -> {r} -> {t}")

                        except Exception as e:
                            print(f"⚠️ 创建关系异常 {s}-{r}->{t}: {str(e)}")

                print(f"✓ 创建了 {len(edges)} 条关系")

            driver.close()

            result_msg = f"成功构建图谱: {len(nodes)} 节点, {len(edges)} 关系"
            print(f"✅ {result_msg}")

            return {
                "graph": {"nodes": nodes, "edges": edges},
                "message": result_msg
            }

        except Exception as e:
            error_msg = f"图谱构建异常: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "graph": {"nodes": [], "edges": []},
                "message": error_msg,
                "error": str(e)
            }

    def _execute_semantic_enhancement(self, task):
        """语义增强"""
        try:
            entities = task.get('entities', [])
            enhanced_entities = []
            knowledge = ExternalKnowledge()

            for e in entities:
                entity_name = e['word']
                wikidata_info = knowledge.query_wikidata_entity(entity_name, 'zh')
                if wikidata_info:
                    enhanced_entities.append({
                        'original': entity_name,
                        'qid': wikidata_info['qid'],
                        'enhanced_description': wikidata_info['description'],
                        'instance_of': wikidata_info['instance_of'],
                        'location': wikidata_info['location']
                    })
                else:
                    enhanced_entities.append({'original': entity_name, 'enhanced': False})

            return {
                "enhanced": len(enhanced_entities) > 0,
                "details": enhanced_entities,
                "message": f"增强了 {len(enhanced_entities)} 个实体"
            }
        except Exception as e:
            return {"enhanced": False, "details": [], "error": str(e)}

    def _execute_entity_standardization(self, task):
        """实体标准化"""
        try:
            entities = task.get('entities', [])
            if not entities:
                return {"standardized_mapping": {}, "standardized_entities": []}

            entity_list = [e['word'] for e in entities]
            prompt = f"""你是一位实体消歧与知识表示领域的专家。你的任务是对知识图谱中的实体名称进行标准化,以确保命名的一致性。

    以下是从知识图谱中提取的一组实体名称,其中部分实体可能指代同一真实世界概念,但表述方式不同。请对这些实体进行归一化,识别出指向同一概念的实体变体,并为每组变体指定一个统一的标准名称。

    实体列表:{entity_list}

    规则:
    1. 只归并确定指向同一概念的实体
    2. 标准名称应该是最完整、最规范的形式
    3. 考虑缩写、简称、全称的对应关系

    输出格式示例:
    {{
      "标准名称1": ["变体名称1", "变体名称2"],
      "标准名称2": ["变体名称3", "变体名称4"]
    }}

    如果没有需要标准化的实体,返回空对象: {{}}

    只输出JSON,不要任何其他内容。"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            standardized = json.loads(content) if content else {}

            # 创建映射:所有变体 -> 标准名
            variant_to_standard = {}
            for standard_name, variants in standardized.items():
                for variant in variants:
                    variant_to_standard[variant] = standard_name
                variant_to_standard[standard_name] = standard_name

            # 标准化实体
            standardized_entities = []
            seen_standards = set()

            for entity in entities:
                original_word = entity['word']
                standard_name = variant_to_standard.get(original_word, original_word)

                if standard_name not in seen_standards:
                    standardized_entities.append({
                        'word': standard_name,
                        'label': entity['label'],
                        'original': original_word,
                        'variants': standardized.get(standard_name, [])
                    })
                    seen_standards.add(standard_name)

            validated_entities = self._validate_entities(standardized_entities)

            print(f"✓ 标准化: {len(entities)} -> {len(validated_entities)} 个实体")
            print(f"✓ 合并映射: {standardized}")

            return {
                "standardized_mapping": standardized,
                "standardized_entities": validated_entities,
                "entity_mapping": variant_to_standard
            }

        except Exception as e:
            print(f"实体标准化错误: {str(e)}")
            return {"standardized_mapping": {}, "standardized_entities": entities, "error": str(e)}

    def _validate_entities(self, entities):
        """验证和过滤实体"""
        generic_terms = {'人物', '组织', '机构', '地点', '家族', '公司',
                         '政府', '项目', '计划', '概念', '理论', '协议'}
        validated = []

        for entity in entities:
            word = entity.get('word', '')
            original = entity.get('original', word)

            # 检查是否过度泛化
            if word in generic_terms:
                print(f"⚠️ 检测到过度泛化: {word} (原名: {original})")
                entity['word'] = original  # 恢复原名

            # 检查是否过短
            if len(word.strip()) < 2:
                print(f"⚠️ 跳过过短实体: {word}")
                continue

            validated.append(entity)

        print(f"✓ 验证后保留 {len(validated)} 个实体")
        return validated

    def _execute_relation_completion(self, task):
        """关系补全"""
        try:
            entities = task.get('entities', [])
            existing_relations = task.get('relations', [])

            if len(entities) < 2:
                return {"completed_relations": []}

            mid = len(entities) // 2
            community1 = [e['word'] for e in entities[:mid]][:5]
            community2 = [e['word'] for e in entities[mid:]][:5]

            if not community1 or not community2:
                return {"completed_relations": []}

            relations_text = '\n'.join([f"{r[0]} - {r[1]} - {r[2]}" for r in existing_relations[:10]])

            prompt = f"""你是一位知识表示与推理方面的专家。你的任务是在知识图谱中推断出彼此未直接连接的实体之间的可能关系。

    社区1实体列表:{community1}
    社区2实体列表:{community2}

    以下是部分涉及这些实体的已有三元组关系:
    {relations_text}

    请根据上述信息,推理出2-3条在社区1与社区2之间可能存在的关系。

    要求:
    (1) 仅包含高度可信、语义明确的关系
    (2) 谓语必须简洁明了,限制为最多2个词,推荐使用1个词
    (3) 禁止主宾实体相同,不得出现自指关系
    (4) 所选谓语应能准确描述两实体之间的具体联系

    输出格式:
    [
      {{"subject": "社区1中的实体", "predicate": "推理关系", "object": "社区2中的实体"}}
    ]

    只输出JSON数组,不要任何其他内容。"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Return only valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            completed_relations = json.loads(content) if content else []

            formatted_relations = []
            for rel in completed_relations:
                if isinstance(rel, dict) and all(k in rel for k in ['subject', 'predicate', 'object']):
                    # 确保subject和object不相同
                    if rel['subject'] != rel['object']:
                        formatted_relations.append([rel['subject'], rel['predicate'], rel['object']])

            return {"completed_relations": formatted_relations}

        except Exception as e:
            return {"completed_relations": [], "error": str(e)}


# ===========================================
# 具体智能体实现
# ===========================================

class PreprocessorAgent(CustomAgent):
    def __init__(self):
        super().__init__("Preprocessor", "preprocess")


class NERAgent(CustomAgent):
    def __init__(self):
        super().__init__("NER", "ner")


class RelationExtractionAgent(CustomAgent):
    def __init__(self):
        super().__init__("RelationExtraction", "relation")


class EventExtractionAgent(CustomAgent):
    def __init__(self):
        super().__init__("EventExtraction", "event")


class EntityStandardizationAgent(CustomAgent):
    def __init__(self):
        super().__init__("EntityStandardization", "standardize")


class RelationCompletionAgent(CustomAgent):
    def __init__(self):
        super().__init__("RelationCompletion", "complete_relations")


class GraphBuilderAgent(CustomAgent):
    def __init__(self):
        super().__init__("GraphBuilder", "graph")


class SemanticEnhancementAgent(CustomAgent):
    def __init__(self):
        super().__init__("SemanticEnhancement", "semantic")


class DocumentExtractorAgent(CustomAgent):
    def __init__(self):
        super().__init__("DocumentExtractor", "extract_document")


# ===========================================
# 异步处理函数
# ===========================================

async def async_process_document_task(file_content: bytes, file_name: str, task_id: str):
    """异步文档处理后台任务"""
    try:
        start_time = time.time()
        processing_status[task_id] = {
            "status": "processing",
            "progress": 5,
            "message": "开始处理...",
            "created_at": time.time()
        }

        # 1. 文档提取 - 直接实例化
        processing_status[task_id].update({"progress": 10, "message": "提取文档内容..."})
        extractor = DocumentExtractorAgent()
        extract_task = {"type": "extract_document", "file_content": file_content, "file_name": file_name}
        extracted = extractor.execute(extract_task)
        text = extracted.get("extracted_text", "")

        if not text:
            raise Exception(extracted.get("error", "文档提取失败"))

        # 2. 预处理
        processing_status[task_id].update({"progress": 20, "message": "预处理文本..."})
        cleaned_text = ' '.join(text.split())

        # 3. 并行执行 NER 和事件抽取
        processing_status[task_id].update({"progress": 30, "message": "并行提取实体和事件..."})
        entities_task = call_deepseek_api(
            f"""从文本中提取实体。文本: ```{cleaned_text}```
返回JSON: {{"entities": [{{"word": "实体", "label": "类型"}}]}}""", "ner"
        )
        events_task = call_deepseek_api(
            f"""从文本中提取事件。文本: ```{cleaned_text}```
返回JSON: {{"events": ["事件1"]}}""", "event"
        )

        entities_result, events_result = await asyncio.gather(
            entities_task, events_task, return_exceptions=True
        )

        if isinstance(events_result, Exception):
            print(f"⚠️ 事件抽取失败: {str(events_result)}")
            events_result = {"events": []}
        else:
            print(f"✓ 事件抽取成功: {events_result}")

        print(f"✓ 实体数量: {len(entities_result.get('entities', []))}")
        print(f"✓ 事件数量: {len(events_result.get('events', []))}")
        print(f"✓ 事件内容: {events_result.get('events', [])[:3]}")  # 显示前3个事件_execute_graph_build

        if isinstance(entities_result, Exception):
            entities_result = {"entities": []}
        if isinstance(events_result, Exception):
            events_result = {"events": []}

        # 4. 实体标准化（提前到关系抽取之前）
        processing_status[task_id].update({"progress": 50, "message": "实体标准化..."})
        standardization_agent = EntityStandardizationAgent()
        standardize_task = {"type": "standardize", "entities": entities_result.get("entities", [])}
        standardized = standardization_agent.execute(standardize_task)
        standardized_entities = standardized.get("standardized_entities", entities_result.get("entities", []))

        # 5. 关系抽取（使用标准化后的实体）
        processing_status[task_id].update({"progress": 60, "message": "提取关系..."})
        if standardized_entities:
            entities_str = ', '.join([e['word'] for e in standardized_entities])
            relations_result = await call_deepseek_api(
                f"""从文本中提取实体间的关系。

        文本: ```{cleaned_text}```

        **可用实体（必须精确使用）**: 
        {entities_str}

        **规则**:
        1. 主语和宾语必须从上述实体列表中精确选择
        2. 谓语使用简短动词（1-2字）
        3. 只提取文本中明确的关系

        返回JSON: {{"relations": [["实体1", "动词", "实体2"]]}}""", "relation"
            )
        else:
            relations_result = {"relations": []}

        print(f"✓ 关系数量: {len(relations_result.get('relations', []))}")

        # 6. 关系补全 - 直接实例化
        processing_status[task_id].update({"progress": 70, "message": "关系补全..."})
        completion_agent = RelationCompletionAgent()
        complete_task = {
            "type": "complete_relations",
            "entities": standardized_entities,
            "relations": relations_result.get("relations", [])
        }
        completed = completion_agent.execute(complete_task)
        all_relations = relations_result.get("relations", []) + completed.get("completed_relations", [])

        # 7. 语义增强 - 直接实例化
        processing_status[task_id].update({"progress": 80, "message": "语义增强..."})
        semantic_agent = SemanticEnhancementAgent()
        semantic_task = {"type": "semantic", "entities": standardized_entities}
        enhanced = semantic_agent.execute(semantic_task)

        # 8. 实体消歧
        disambiguated = []
        for e in standardized_entities:
            disambig_entity = disambiguator.disambiguate(e.get('original', e['word']), text)
            disambiguated.append({
                "original": e.get('original', e['word']),
                "standardized": e['word'],
                "disambiguated": disambig_entity,
                "label": e['label']
            })

        # 8.5 关系实体名称修正
        processing_status[task_id].update({"progress": 85, "message": "修正关系..."})

        entity_names = {e['word'] for e in standardized_entities}
        entity_name_list = [e['word'] for e in standardized_entities]

        corrected_relations = []
        skipped_relations = []

        for rel in all_relations:
            if len(rel) != 3:
                continue

            s, r, t = rel
            original_s, original_t = s, t
            s_found, t_found = False, False

            # 精确匹配检查
            if s in entity_names:
                s_found = True
            if t in entity_names:
                t_found = True

            # 如果源实体未找到,尝试模糊匹配
            if not s_found:
                for entity in entity_name_list:
                    # 方法1: 包含关系
                    if s in entity or entity in s:
                        s = entity
                        s_found = True
                        print(f"✓ 修正源实体: '{original_s}' -> '{entity}'")
                        break
                    # 方法2: 去除"基于"等前缀后匹配
                    s_clean = s.replace('基于', '').replace('的', '').replace('与', '').replace('协作', '').strip()
                    if len(s_clean) >= 3 and s_clean in entity:
                        s = entity
                        s_found = True
                        print(f"✓ 修正源实体: '{original_s}' -> '{entity}'")
                        break

            # 如果目标实体未找到，尝试模糊匹配
            if not t_found:
                for entity in entity_name_list:
                    if t in entity or entity in t:
                        t = entity
                        t_found = True
                        print(f"✓ 修正目标实体: '{original_t}' -> '{entity}'")
                        break
                    t_clean = t.replace('基于', '').replace('的', '').replace('与', '')
                    e_clean = entity.replace('基于', '').replace('的', '').replace('与', '')
                    if t_clean and e_clean and (t_clean in e_clean or e_clean in t_clean):
                        t = entity
                        t_found = True
                        print(f"✓ 修正目标实体: '{original_t}' -> '{entity}'")
                        break

            # 只保留两端实体都找到的关系
            if s_found and t_found:
                corrected_relations.append([s, r, t])
            else:
                skipped_relations.append(rel)
                if not s_found:
                    print(f"⚠️ 无法修正源实体: {original_s}")
                if not t_found:
                    print(f"⚠️ 无法修正目标实体: {original_t}")

        print(f"✅ 关系修正完成: {len(all_relations)} -> {len(corrected_relations)} (跳过 {len(skipped_relations)})")
        all_relations = corrected_relations

        # 9. 图构建 - 直接实例化
        processing_status[task_id].update({"progress": 90, "message": "构建知识图谱..."})
        enhanced_details = enhanced.get("details", [])
        enhanced_map = {item['original']: item for item in enhanced_details}

        print(f"📊 准备构建图谱:")
        print(f"  - 实体数: {len(standardized_entities)}")
        print(f"  - 关系数: {len(all_relations)}")
        print(f"  - 增强实体数: {len(enhanced_map)}")

        graph_agent = GraphBuilderAgent()
        graph_task = {
            "type": "graph",
            "entities": standardized_entities,
            "relations": all_relations,
            "enhanced_entities": enhanced_map,
            "disambiguated_entities": disambiguated,
            "completed_relations": completed.get("completed_relations", [])
        }

        try:
            graph = graph_agent.execute(graph_task)
            print(f"✅ 图谱构建结果: {graph.get('message', 'Unknown')}")

            if 'error' in graph:
                print(f"⚠️ 图谱构建警告: {graph.get('error')}")

        except Exception as e:
            print(f"❌ 图谱构建失败: {str(e)}")
            import traceback
            traceback.print_exc()
            graph = {
                "graph": {"nodes": [], "edges": []},
                "message": f"图谱构建失败: {str(e)}"
            }
        graph = graph_agent.execute(graph_task)

        processing_time = time.time() - start_time

        # 构建完整结果
        result_data = {
            "status": "success",
            "preprocessed": {
                "cleaned_text": cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text)
            },
            "entities": entities_result,
            "entity_standardization": standardized,
            "relations": relations_result,
            "relation_completion": completed,
            "all_relations": {"relations": all_relations},
            "events": events_result,
            "semantic_enhancement": enhanced,
            "graph": graph,
            "disambiguated_entities": disambiguated
        }

        processing_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "处理完成",
            "processing_time": processing_time,
            "created_at": processing_status[task_id]["created_at"],
            "result": result_data
        }

        # 添加额外日志
        print(f"✅ 任务完成:")
        print(f"  - 任务ID: {task_id}")
        print(f"  - 节点数: {len(graph.get('graph', {}).get('nodes', []))}")
        print(f"  - 边数: {len(graph.get('graph', {}).get('edges', []))}")
        print(f"  - 事件数: {len(events_result.get('events', []))}")

        save_analysis_result(task_id, file_name, result_data, processing_time)

    except Exception as e:
        print(f"❌ 异步处理失败: {str(e)}")
        processing_status[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"处理失败: {str(e)}",
            "error": str(e),
            "created_at": processing_status.get(task_id, {}).get("created_at", time.time())
        }


# 更好的编码检测
def detect_encoding(content):
    """检测文件编码"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5']

    for encoding in encodings:
        try:
            content.decode(encoding)
            return encoding
        except (UnicodeDecodeError, AttributeError):
            continue

    return 'utf-8'  # 默认返回 UTF-8

# ===========================================
# 初始化全局对象
# ===========================================

scheduler = MultiAgentScheduler()
disambiguator = EntityDisambiguation()


# ===========================================
# FastAPI应用配置
# ===========================================

async def cleanup_old_tasks():
    """清理24小时前的任务状态"""
    while True:
        try:
            current_time = time.time()
            tasks_to_remove = [
                task_id for task_id, status in processing_status.items()
                if current_time - status.get("created_at", current_time) > 24 * 3600
            ]
            for task_id in tasks_to_remove:
                del processing_status[task_id]
            if tasks_to_remove:
                print(f"🧹 清理了 {len(tasks_to_remove)} 个旧任务")
        except Exception as e:
            print(f"清理任务时出错: {e}")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app):
    print("🚀 启动知识图谱系统...")

    # 数据库优化
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )
        with driver.session() as session:
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
                "CREATE INDEX entity_qid IF NOT EXISTS FOR (n:Entity) ON (n.qid)",
                "CREATE INDEX entity_label IF NOT EXISTS FOR (n:Entity) ON (n.label)",
            ]
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    print(f"索引创建警告: {e}")
        driver.close()
        print("✅ 数据库索引优化完成")
    except Exception as e:
        print(f"⚠️ 数据库优化失败: {e}")

    # 注册所有智能体（确保按正确顺序）
    agents = [
        DocumentExtractorAgent(),  # 文档提取（第一个）
        PreprocessorAgent(),  # 预处理
        NERAgent(),  # 实体识别
        RelationExtractionAgent(),  # 关系抽取
        EventExtractionAgent(),  # 事件抽取
        EntityStandardizationAgent(),  # 实体标准化
        RelationCompletionAgent(),  # 关系补全
        SemanticEnhancementAgent(),  # 语义增强
        GraphBuilderAgent(),  # 图构建
    ]

    for agent in agents:
        scheduler.register_agent(agent)
        print(f"  ✓ 注册 {agent.name}")

    print("✅ 所有智能体注册成功")

    # 启动清理任务
    cleanup_task = asyncio.create_task(cleanup_old_tasks())
    print("✅ 后台清理任务启动")

    yield

    cleanup_task.cancel()
    print("🛑 应用关闭")


# 创建FastAPI应用
app = FastAPI(
    title="知识图谱管理系统",
    description="智能文档分析与知识图谱构建平台",
    version="2.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 性能监控中间件
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    if process_time > 2.0:
        print(f"⚠️ 慢请求: {request.url.path} 耗时 {process_time:.2f}s")

    response.headers["X-Process-Time"] = str(process_time)
    return response


# ===========================================
# API路由
# ===========================================

@app.post("/process_document_async")
async def process_document_async(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """异步文档处理端点"""
    try:
        task_id = str(uuid.uuid4())
        file_content = await file.read()

        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.txt', '.docx', '.pdf']:
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        background_tasks.add_task(async_process_document_task, file_content, file.filename, task_id)

        return {"task_id": task_id, "status": "started", "message": "文档处理已开始"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing_status/{task_id}")
async def get_processing_status(task_id: str):
    """获取处理状态"""
    status = processing_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务未找到")
    return status


@app.get("/get_graph")
async def get_graph():
    """获取图谱数据"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )

        with driver.session() as session:
            # 获取所有节点(包括孤立节点)
            nodes_result = session.run("""
                MATCH (n:Entity)
                RETURN n.name as name, n.label as label, n.qid as qid,
                       n.description as description, n.instance_of as instance_of
                LIMIT 1000
            """)

            nodes_data = []
            node_types = {}
            for record in nodes_result:
                name = record["name"]
                label = record["label"]
                nodes_data.append({
                    "id": name,
                    "label": name,
                    "type": label or "Unknown",
                    "qid": record["qid"],
                    "description": record["description"]
                })
                node_types[name] = label or "Unknown"

            # 获取所有关系
            edges_result = session.run("""
                MATCH (n:Entity)-[r:REL]->(m:Entity)
                RETURN n.name as source, r.type as relation, m.name as target,
                       coalesce(r.inferred, false) as inferred
                LIMIT 1000
            """)

            edges = []
            for record in edges_result:
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["relation"],
                    "inferred": record["inferred"]
                })

            print(f"📊 /get_graph 返回: {len(nodes_data)} 节点, {len(edges)} 边")
            return {"nodes": nodes_data, "edges": edges}

        driver.close()
    except Exception as e:
        print(f"❌ 获取图谱失败: {str(e)}")
        return {"nodes": [], "edges": [], "error": str(e)}


@app.post("/clear_graph")
async def clear_graph():
    """清空图谱和所有分析结果"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()

        if RESULTS_DIR.exists():
            for file in RESULTS_DIR.glob("*.json"):
                file.unlink()

        if cache_manager:
            try:
                cache_manager.flushdb()
            except Exception as e:
                print(f"⚠️ Redis清空失败: {e}")

        return {"status": "success", "message": "图谱和所有分析结果已清空"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export_graph")
async def export_graph():
    """导出图谱和分析结果"""
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )

        with driver.session() as session:
            nodes_result = session.run("""
                MATCH (n:Entity)
                RETURN n.name as name, n.qid as qid, n.label as label,
                       n.description as description, n.instance_of as instance_of,
                       n.original_name as original_name
            """)
            nodes = [dict(record) for record in nodes_result]

            edges_result = session.run("""
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                RETURN a.name as source, b.name as target, 
                       r.type as relation_type, r.inferred as inferred
            """)
            edges = [dict(record) for record in edges_result]

        driver.close()

        latest_analysis = None
        if RESULTS_DIR.exists():
            files = sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    latest_analysis = json.load(f)

        export_data = {
            "version": "2.0",
            "export_time": datetime.now().isoformat(),
            "graph": {"nodes": nodes, "edges": edges},
            "statistics": {"node_count": len(nodes), "edge_count": len(edges)},
            "latest_analysis": latest_analysis
        }

        return JSONResponse(
            content=export_data,
            headers={"Content-Disposition": f"attachment; filename=kg_export_{int(time.time())}.json"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import_graph")
async def import_graph(file: UploadFile = File(...)):
    """导入图谱和分析结果"""
    try:
        content = await file.read()
        graph_data = json.loads(content.decode('utf-8'))

        # 兼容不同的JSON格式
        if "graph" in graph_data:
            # 格式1: {"graph": {"nodes": [...], "edges": [...]}}
            nodes = graph_data["graph"].get("nodes", [])
            edges = graph_data["graph"].get("edges", [])
        elif "nodes" in graph_data and "edges" in graph_data:
            # 格式2: {"nodes": [...], "edges": [...]}
            nodes = graph_data["nodes"]
            edges = graph_data["edges"]
        else:
            raise ValueError("无效的JSON格式,缺少nodes或edges字段")

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )

        nodes_imported = 0
        edges_imported = 0

        with driver.session() as session:
            # 导入节点
            for node in nodes:
                try:
                    # 兼容不同的节点字段名
                    node_name = node.get('name') or node.get('id') or node.get('label')
                    if not node_name:
                        print(f"⚠️ 跳过无效节点: {node}")
                        continue

                    session.run("""
                        MERGE (n:Entity {name: $name})
                        SET n.qid = $qid, 
                            n.label = $label,
                            n.description = $description, 
                            n.instance_of = $instance_of,
                            n.original_name = $original_name, 
                            n.imported_at = datetime()
                    """,
                                name=node_name,
                                qid=node.get('qid'),
                                label=node.get('label') or node.get('type'),
                                description=node.get('description', ''),
                                instance_of=node.get('instance_of', ''),
                                original_name=node.get('original_name', node_name)
                                )
                    nodes_imported += 1
                except Exception as e:
                    print(f"⚠️ 导入节点失败 {node.get('name')}: {str(e)}")

            # 导入边
            for edge in edges:
                try:
                    # 兼容不同的边字段名
                    source = edge.get("source") or edge.get("from")
                    target = edge.get("target") or edge.get("to")
                    rel_type = edge.get("type") or edge.get("relation") or edge.get("relation_type")

                    if not source or not target or not rel_type:
                        print(f"⚠️ 跳过无效边: {edge}")
                        continue

                    session.run("""
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:REL {type: $relation_type}]->(b)
                        SET r.inferred = $inferred, r.imported_at = datetime()
                    """,
                                source=source,
                                target=target,
                                relation_type=rel_type,
                                inferred=edge.get("inferred", False)
                                )
                    edges_imported += 1
                except Exception as e:
                    print(f"⚠️ 导入边失败 {edge.get('source')}->{edge.get('target')}: {str(e)}")

        driver.close()

        return {
            "status": "success",
            "message": f"成功导入 {nodes_imported} 个节点和 {edges_imported} 条关系",
            "nodes_imported": nodes_imported,
            "edges_imported": edges_imported
        }

    except Exception as e:
        print(f"❌ 导入错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


@app.get("/get_latest_analysis")
async def get_latest_analysis():
    """获取最新的分析结果"""
    try:
        if not RESULTS_DIR.exists():
            return {"status": "no_results", "message": "暂无分析结果"}

        files = sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not files:
            return {"status": "no_results", "message": "暂无分析结果"}

        with open(files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {"status": "success", "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_analysis_history")
async def get_analysis_history(limit: int = 10):
    """获取历史分析记录列表"""
    try:
        if not RESULTS_DIR.exists():
            return {"status": "no_results", "history": []}

        files = sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

        history = []
        for file_path in files[:limit]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history.append({
                        "task_id": data.get("task_id"),
                        "file_name": data.get("file_name"),
                        "processed_at": data.get("processed_at"),
                        "processing_time": data.get("processing_time")
                    })
            except:
                continue

        return {"status": "success", "history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_analysis_by_id/{task_id}")
async def get_analysis_by_id(task_id: str):
    """根据任务ID获取分析结果"""
    try:
        result_file = RESULTS_DIR / f"{task_id}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="分析结果不存在")

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    redis_status = "connected" if cache_manager else "disconnected"

    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )
        with driver.session() as session:
            session.run("RETURN 1")
        neo4j_status = "connected"
        driver.close()
    except:
        neo4j_status = "error"

    return {
        "status": "healthy",
        "services": {"redis": redis_status, "neo4j": neo4j_status},
        "timestamp": time.time()
    }


@app.post("/import_analysis_csv")
async def import_analysis_csv(
        nodes_file: UploadFile = File(...),
        edges_file: UploadFile = File(None)
):
    """导入CSV格式的分析结果"""
    try:
        # 读取节点CSV
        nodes_content = await nodes_file.read()

        detected_encoding = detect_encoding(nodes_content)
        nodes_text = nodes_content.decode(detected_encoding)


        if not nodes_text:
            raise HTTPException(status_code=400, detail="无法解码节点CSV文件，请检查文件编码")

        # 解析节点CSV
        nodes_lines = nodes_text.strip().split('\n')
        if len(nodes_lines) < 2:
            raise HTTPException(status_code=400, detail="节点CSV文件格式错误或为空")

        # 简单CSV解析函数
        def parse_csv_line(line):
            parts = []
            current = ""
            in_quotes = False
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current.strip().strip('"'))
                    current = ""
                else:
                    current += char
            parts.append(current.strip().strip('"'))
            return parts

        # 跳过表头，解析节点
        nodes = []
        for i, line in enumerate(nodes_lines[1:], start=2):
            if not line.strip():
                continue

            try:
                parts = parse_csv_line(line)

                if len(parts) < 2:
                    print(f"⚠️ 跳过第{i}行：字段不足")
                    continue

                node_name = parts[0].strip()
                node_label = parts[1].strip() if len(parts) > 1 else ""
                node_type = parts[2].strip() if len(parts) > 2 else "Unknown"
                node_description = parts[3].strip() if len(parts) > 3 else ""

                # 验证节点名称
                if not node_name:
                    print(f"⚠️ 跳过第{i}行：节点名称为空")
                    continue

                nodes.append({
                    "name": node_name,
                    "label": node_label or node_name,
                    "type": node_type,
                    "description": node_description
                })
            except Exception as e:
                print(f"⚠️ 解析第{i}行失败: {str(e)}")
                continue

        if not nodes:
            raise HTTPException(status_code=400, detail="未能解析出任何有效节点")

        # 解析边CSV（如果提供）
        edges = []
        if edges_file:
            edges_content = await edges_file.read()

            # 尝试多种编码
            edges_text = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
                try:
                    edges_text = edges_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if not edges_text:
                print("⚠️ 无法解码边CSV文件，跳过边的导入")
            else:
                edges_lines = edges_text.strip().split('\n')

                for i, line in enumerate(edges_lines[1:], start=2):
                    if not line.strip():
                        continue

                    try:
                        parts = parse_csv_line(line)

                        if len(parts) < 3:
                            print(f"⚠️ 跳过边第{i}行：字段不足")
                            continue

                        source = parts[0].strip()
                        rel_type = parts[1].strip()
                        target = parts[2].strip()
                        inferred = parts[3].strip().lower() == '是' if len(parts) > 3 else False

                        # 验证必需字段
                        if not source or not target or not rel_type:
                            print(f"⚠️ 跳过边第{i}行：源节点、目标节点或关系类型为空")
                            continue

                        edges.append({
                            "source": source,
                            "type": rel_type,
                            "target": target,
                            "inferred": inferred
                        })
                    except Exception as e:
                        print(f"⚠️ 解析边第{i}行失败: {str(e)}")
                        continue

        # 导入到Neo4j
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678"))
        )

        nodes_imported = 0
        edges_imported = 0

        with driver.session() as session:
            # 创建节点
            for node in nodes:
                try:
                    session.run("""
                        MERGE (n:Entity {name: $name})
                        SET n.label = $label, n.type = $type,
                            n.description = $description, n.imported_at = datetime()
                    """, name=node["name"], label=node["label"],
                                type=node["type"], description=node["description"])
                    nodes_imported += 1
                except Exception as e:
                    print(f"⚠️ 导入节点失败 {node['name']}: {str(e)}")

            # 创建边
            for edge in edges:
                try:
                    session.run("""
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:REL {type: $rel_type}]->(b)
                        SET r.inferred = $inferred, r.imported_at = datetime()
                    """, source=edge["source"], target=edge["target"],
                                rel_type=edge["type"], inferred=edge["inferred"])
                    edges_imported += 1
                except Exception as e:
                    print(f"⚠️ 导入边失败 {edge['source']}->{edge['target']}: {str(e)}")

        driver.close()

        return {
            "status": "success",
            "message": f"成功导入 {nodes_imported} 个节点和 {edges_imported} 条关系",
            "nodes_imported": nodes_imported,
            "edges_imported": edges_imported,
            "nodes_total": len(nodes),
            "edges_total": len(edges)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ CSV导入错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


# 挂载静态文件
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "../static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ===========================================
# 主函数
# ===========================================

if __name__ == "__main__":
    print("🌐 启动知识图谱管理系统...")
    print("📊 功能特性:")
    print("   - 异步文档处理")
    print("   - Redis缓存加速")
    print("   - 实体标准化与验证")
    print("   - 关系推理补全")
    print()
    print("🔗 访问地址:")
    print("   - 前端界面: http://localhost:8000")
    print("   - API文档: http://localhost:8000/docs")
    print("   - 健康检查: http://localhost:8000/health")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, log_level="info")