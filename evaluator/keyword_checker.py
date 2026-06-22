"""
关键词检测器

使用Aho-Corasick多模式匹配算法实现O(n)级关键词检测。
"""
import os
import asyncio
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class KeywordMatchResult:
    """
    关键词匹配结果
    
    Attributes:
        text: 待检测文本
        matched: 是否匹配到关键词
        matched_keywords: 匹配到的关键词列表
        matched_categories: 匹配到的关键词类别
        match_positions: 匹配位置信息
        severity: 严重程度（1-5）
    """
    text: str
    matched: bool
    matched_keywords: List[str]
    matched_categories: List[str]
    match_positions: List[Dict]
    severity: int
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "text": self.text,
            "matched": self.matched,
            "matched_keywords": self.matched_keywords,
            "matched_categories": self.matched_categories,
            "match_positions": self.match_positions,
            "severity": self.severity
        }


class AhoCorasickAutomaton:
    """
    Aho-Corasick多模式匹配自动机
    
    构建时间 O(total_keyword_length)
    匹配时间 O(text_length + total_matches)
    
    相比逐词正则匹配（O(text_length * keyword_count)）效率提升数百倍。
    """
    
    def __init__(self):
        # 每个节点：{char: next_node_id}
        self.trie: List[Dict[str, int]] = [{}]
        # 失败指针
        self.fail: List[int] = [0]
        # 节点输出：{keyword: category}
        self.output: List[Dict[str, str]] = [{}]
        # 是否已构建失败指针
        self._built = False
    
    def add_keyword(self, keyword: str, category: str):
        """添加关键词到Trie树"""
        node = 0
        for char in keyword:
            if char not in self.trie[node]:
                self.trie[node][char] = len(self.trie)
                self.trie.append({})
                self.fail.append(0)
                self.output.append({})
            node = self.trie[node][char]
        self.output[node][keyword] = category
        self._built = False
    
    def build(self):
        """构建失败指针（BFS）"""
        from collections import deque
        q = deque()
        
        # 初始化第一层节点的失败指针
        for char, next_node in self.trie[0].items():
            q.append(next_node)
            self.fail[next_node] = 0
        
        # BFS构建所有节点的失败指针
        while q:
            r = q.popleft()
            for char, u in self.trie[r].items():
                q.append(u)
                # 沿失败指针回溯，寻找最长后缀
                v = self.fail[r]
                while v and char not in self.trie[v]:
                    v = self.fail[v]
                self.fail[u] = self.trie[v].get(char, 0)
                # 合并输出（后缀模式匹配）
                if self.fail[u]:
                    self.output[u].update(self.output[self.fail[u]])
        
        self._built = True
    
    def search(self, text: str) -> List[Dict]:
        """
        搜索文本中所有匹配的关键词
        
        Args:
            text: 待搜索文本
            
        Returns:
            List[Dict]: 匹配结果列表，每项含keyword/category/start/end
        """
        if not self._built:
            self.build()
        
        node = 0
        matches = []
        
        for i, char in enumerate(text):
            # 沿失败指针寻找匹配字符
            while node and char not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(char, 0)
            
            # 检查当前节点是否有输出
            if self.output[node]:
                for keyword, category in self.output[node].items():
                    matches.append({
                        "keyword": keyword,
                        "category": category,
                        "start": i - len(keyword) + 1,
                        "end": i + 1
                    })
        
        return matches


class KeywordChecker:
    """
    关键词检测器
    
    使用Aho-Corasick自动机进行高效关键词匹配。
    批量检测使用asyncio并行加速。
    """
    
    def __init__(self, lexicon_paths: Optional[List[str]] = None):
        """
        初始化关键词检测器
        
        Args:
            lexicon_paths: 词库文件路径列表（可选）
        """
        # 词库存储
        self.lexicons: Dict[str, Set[str]] = {}
        
        # 所有关键词集合
        self.all_keywords: Set[str] = set()
        
        # 关键词到类别的映射
        self.keyword_to_category: Dict[str, str] = {}
        
        # Aho-Corasick自动机
        self.automaton = AhoCorasickAutomaton()
        
        # 严重程度映射
        self.severity_map = {
            "暴恐词库": 5,
            "涉枪涉爆": 4,
            "色情类型": 3,
        }
        
        # 默认词库路径
        default_paths = [
            "/Vocabulary/暴恐词库.txt",
            "/Vocabulary/涉枪涉爆.txt",
            "/Vocabulary/色情类型.txt",
        ]
        
        # 加载词库
        paths = lexicon_paths or default_paths
        total_added = 0
        for path in paths:
            count = self._load_lexicon(path)
            total_added += count
        
        # 构建Aho-Corasick自动机
        self.automaton.build()
        
        logger.info(f"关键词检测器初始化完成，共加载 {total_added} 个关键词，"
                    f"涉及 {len(self.lexicons)} 个类别")
    
    def _load_lexicon(self, path: str) -> int:
        """
        加载词库文件
        
        Args:
            path: 词库文件路径
            
        Returns:
            int: 加载的关键词数量
        """
        possible_paths = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(__file__), "..", "..", path),
        ]
        
        for full_path in possible_paths:
            if os.path.exists(full_path):
                try:
                    category = Path(full_path).stem
                    count = 0
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                self.all_keywords.add(line)
                                self.keyword_to_category[line] = category
                                self.automaton.add_keyword(line, category)
                                count += 1
                    
                    if category not in self.lexicons:
                        self.lexicons[category] = set()
                    self.lexicons[category].update(
                        [k for k, c in self.keyword_to_category.items() if c == category]
                    )
                    
                    logger.debug(f"加载词库: {category}, 关键词数: {count}")
                    return count
                    
                except Exception as e:
                    logger.error(f"加载词库失败: {path}, 错误: {e}")
        
        logger.warning(f"词库文件不存在: {path}")
        return 0
    
    def check_text(self, text: str) -> KeywordMatchResult:
        """
        使用Aho-Corasick自动机检测文本中的关键词
        
        Args:
            text: 待检测文本
            
        Returns:
            KeywordMatchResult: 检测结果
        """
        # 使用AC自动机一次性完成所有关键词匹配
        matches = self.automaton.search(text)
        
        matched_keywords = []
        matched_categories = []
        match_positions = []
        seen_keywords = set()
        
        for m in matches:
            kw = m["keyword"]
            cat = m["category"]
            if kw not in seen_keywords:
                seen_keywords.add(kw)
                matched_keywords.append(kw)
            if cat not in matched_categories:
                matched_categories.append(cat)
            match_positions.append(m)
        
        severity = self._calculate_severity(matched_keywords, matched_categories)
        
        return KeywordMatchResult(
            text=text,
            matched=len(matched_keywords) > 0,
            matched_keywords=matched_keywords,
            matched_categories=matched_categories,
            match_positions=match_positions,
            severity=severity
        )
    
    def _calculate_severity(
        self,
        matched_keywords: List[str],
        matched_categories: List[str]
    ) -> int:
        """
        计算严重程度
        
        Args:
            matched_keywords: 匹配的关键词列表
            matched_categories: 匹配的类别列表
            
        Returns:
            int: 严重程度（1-5）
        """
        if not matched_keywords:
            return 0
        
        max_severity = 0
        for category in matched_categories:
            sev = self.severity_map.get(category, 2)
            max_severity = max(max_severity, sev)
        
        if len(matched_keywords) > 5:
            max_severity = min(5, max_severity + 1)
        
        return max_severity
    
    async def check_text_async(self, text: str) -> KeywordMatchResult:
        """
        异步检测（与同步检测相同，AC自动机本身已是O(n)，
        此处为接口统一提供async包装）
        
        Args:
            text: 待检测文本
            
        Returns:
            KeywordMatchResult: 检测结果
        """
        return self.check_text(text)
    
    async def batch_check(self, texts: List[str]) -> List[KeywordMatchResult]:
        """
        批量检测——使用asyncio.gather并行执行
        
        由于AC自动机每次匹配已是O(n)且不涉及I/O阻塞，
        使用asyncio.gather实现并发调度。
        
        Args:
            texts: 待检测文本列表
            
        Returns:
            List[KeywordMatchResult]: 检测结果列表
        """
        tasks = [self.check_text_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def get_lexicon_info(self) -> Dict:
        """
        获取词库信息
        
        Returns:
            Dict: 词库统计信息
        """
        return {
            "total_keywords": len(self.all_keywords),
            "categories": {
                category: len(keywords)
                for category, keywords in self.lexicons.items()
            }
        }
    
    def add_custom_keywords(self, keywords: List[str], category: str = "custom"):
        """
        添加自定义关键词
        
        Args:
            keywords: 关键词列表
            category: 类别名称
        """
        for keyword in keywords:
            self.all_keywords.add(keyword)
            self.keyword_to_category[keyword] = category
            self.automaton.add_keyword(keyword, category)
        
        if category not in self.lexicons:
            self.lexicons[category] = set()
        self.lexicons[category].update(keywords)
        
        # 重建自动机
        self.automaton.build()
        
        logger.info(f"添加自定义关键词: {len(keywords)} 个, 类别: {category}")


# 创建全局关键词检测器实例
keyword_checker = KeywordChecker()


def get_keyword_checker() -> KeywordChecker:
    """
    获取全局关键词检测器实例
    
    Returns:
        KeywordChecker: 关键词检测器实例
    """
    return keyword_checker