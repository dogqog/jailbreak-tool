"""
JailBench词库加载器

加载JailBench_seed.csv格式的敏感问题库，每个问题已经是完整的可用句式。
"""
import os
import csv
import random
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class JailBenchQuery:
    """
    JailBench问题数据结构
    
    Attributes:
        index: 问题索引
        query: 完整的越狱提示词
        primary_category: 一级领域分类
        secondary_category: 二级领域分类
    """
    index: int
    query: str
    primary_category: str
    secondary_category: str
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "index": self.index,
            "query": self.query,
            "primary_category": self.primary_category,
            "secondary_category": self.secondary_category
        }


class JailBenchLoader:
    """
    JailBench词库加载器
    
    加载CSV格式的敏感问题库，提供按类别查询和随机选择功能。
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化JailBench加载器
        
        Args:
            csv_path: CSV文件路径（可选，默认使用Vocabulary/JailBench_seed.csv）
        """
        # 问题存储
        self.queries: List[JailBenchQuery] = []
        
        # 按一级领域分类存储
        self.queries_by_primary: Dict[str, List[JailBenchQuery]] = {}
        
        # 按二级领域分类存储
        self.queries_by_secondary: Dict[str, List[JailBenchQuery]] = {}
        
        # 所有一级领域类别
        self.primary_categories: Set[str] = set()
        
        # 所有二级领域类别
        self.secondary_categories: Set[str] = set()
        
        # 默认CSV路径
        default_path = "Vocabulary/JailBench_seed.csv"
        
        # 加载CSV
        path = csv_path or default_path
        self._load_csv(path)
        
        logger.info(
            f"JailBench词库加载完成，共 {len(self.queries)} 条问题，"
            f"一级领域 {len(self.primary_categories)} 个，"
            f"二级领域 {len(self.secondary_categories)} 个"
        )
    
    def _load_csv(self, path: str) -> int:
        """
        加载CSV文件
        
        Args:
            path: CSV文件路径
            
        Returns:
            int: 加载的问题数量
        """
        possible_paths = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(__file__), "..", "..", path),
        ]
        
        for full_path in possible_paths:
            if os.path.exists(full_path):
                try:
                    count = 0
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        
                        for row in reader:
                            # 解析CSV行
                            index = int(row.get('', 0))
                            query = row.get('question_zh', '').strip()
                            primary_cat = row.get('一级领域', '').strip()
                            secondary_cat = row.get('二级领域', '').strip()
                            
                            if query:
                                # 创建问题对象
                                jb_query = JailBenchQuery(
                                    index=index,
                                    query=query,
                                    primary_category=primary_cat,
                                    secondary_category=secondary_cat
                                )
                                
                                # 添加到总列表
                                self.queries.append(jb_query)
                                
                                # 添加到一级领域分类
                                if primary_cat not in self.queries_by_primary:
                                    self.queries_by_primary[primary_cat] = []
                                self.queries_by_primary[primary_cat].append(jb_query)
                                self.primary_categories.add(primary_cat)
                                
                                # 添加到二级领域分类
                                if secondary_cat not in self.queries_by_secondary:
                                    self.queries_by_secondary[secondary_cat] = []
                                self.queries_by_secondary[secondary_cat].append(jb_query)
                                self.secondary_categories.add(secondary_cat)
                                
                                count += 1
                    
                    logger.debug(f"加载JailBench CSV: {full_path}, 问题数: {count}")
                    return count
                    
                except Exception as e:
                    logger.error(f"加载JailBench CSV失败: {path}, 错误: {e}")
                    return 0
        
        logger.warning(f"JailBench CSV文件不存在: {path}")
        return 0
    
    def get_random_query(self) -> JailBenchQuery:
        """
        随机获取一条问题
        
        Returns:
            JailBenchQuery: 随机选择的问题
        """
        if not self.queries:
            raise ValueError("词库为空，无法获取问题")
        
        return random.choice(self.queries)
    
    def get_random_queries(self, count: int = 1) -> List[JailBenchQuery]:
        """
        随机获取多条问题
        
        Args:
            count: 获取数量
            
        Returns:
            List[JailBenchQuery]: 随机选择的问题列表
        """
        if not self.queries:
            raise ValueError("词库为空，无法获取问题")
        
        return random.sample(self.queries, min(count, len(self.queries)))
    
    def get_queries_by_primary_category(self, category: str) -> List[JailBenchQuery]:
        """
        按一级领域获取问题
        
        Args:
            category: 一级领域类别
            
        Returns:
            List[JailBenchQuery]: 该类别下的所有问题
        """
        return self.queries_by_primary.get(category, [])
    
    def get_queries_by_secondary_category(self, category: str) -> List[JailBenchQuery]:
        """
        按二级领域获取问题
        
        Args:
            category: 二级领域类别
            
        Returns:
            List[JailBenchQuery]: 该类别下的所有问题
        """
        return self.queries_by_secondary.get(category, [])
    
    def get_random_query_by_category(
        self,
        primary_category: Optional[str] = None,
        secondary_category: Optional[str] = None
    ) -> JailBenchQuery:
        """
        按类别随机获取一条问题
        
        Args:
            primary_category: 一级领域类别（可选）
            secondary_category: 二级领域类别（可选）
            
        Returns:
            JailBenchQuery: 随机选择的问题
        """
        # 如果指定了二级领域，优先使用
        if secondary_category and secondary_category in self.queries_by_secondary:
            return random.choice(self.queries_by_secondary[secondary_category])
        
        # 如果指定了一级领域
        if primary_category and primary_category in self.queries_by_primary:
            return random.choice(self.queries_by_primary[primary_category])
        
        # 否则随机选择
        return self.get_random_query()
    
    def get_statistics(self) -> Dict:
        """
        获取词库统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "total_queries": len(self.queries),
            "primary_categories": {
                cat: len(queries)
                for cat, queries in self.queries_by_primary.items()
            },
            "secondary_categories": {
                cat: len(queries)
                for cat, queries in self.queries_by_secondary.items()
            }
        }


# 创建全局JailBench加载器实例
jailbench_loader = JailBenchLoader()


def get_jailbench_loader() -> JailBenchLoader:
    """
    获取全局JailBench加载器实例
    
    Returns:
        JailBenchLoader: JailBench加载器实例
    """
    return jailbench_loader