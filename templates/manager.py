"""
越狱模板管理器

统一管理所有越狱策略模板，提供便捷的访问接口。
模板仅从独立 JSON 模板库文件加载，实现模板与代码分离。
"""
import os
import json
from typing import List, Dict, Optional
from loguru import logger

from templates.base import (
    JailbreakTemplate,
    JailbreakStrategy,
    validate_template
)


class TemplateManager:
    """
    模板管理器
    
    负责加载、管理和提供所有越狱策略模板。
    """
    
    def __init__(self, library_dir: Optional[str] = None):
        """
        初始化模板管理器
        
        Args:
            library_dir: 独立模板库目录（JSON文件），如果存在则优先加载
        """
        # 加载所有模板
        self.templates: Dict[str, JailbreakTemplate] = {}

        # 从独立模板库文件加载
        if library_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            library_dir = os.path.join(project_root, "templates_library")

        if os.path.isdir(library_dir):
            loaded = self._load_from_library(library_dir)
            if loaded > 0:
                logger.info(f"模板管理器从 {library_dir} 加载 {loaded} 个模板")

        if not self.templates:
            logger.error(f"未能从 {library_dir} 加载任何有效模板，请检查 JSON 模板库文件")

        logger.info(f"模板管理器初始化完成，共加载 {len(self.templates)} 个模板")
    
    def _load_from_library(self, library_dir: str) -> int:
        """
        从独立模板库 JSON 文件加载模板
        
        加载 all_templates.json；分策略 json 文件已合并至该汇总文件。
        
        Args:
            library_dir: 模板库目录
            
        Returns:
            int: 成功加载的模板数量
        """
        count = 0
        
        # 优先加载汇总文件，避免重复
        all_file = os.path.join(library_dir, "all_templates.json")
        if os.path.isfile(all_file):
            files_to_load = [all_file]
        else:
            files_to_load = [
                os.path.join(library_dir, f)
                for f in sorted(os.listdir(library_dir))
                if f.endswith(".json")
            ]
        
        for filepath in files_to_load:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 支持两种格式：对象列表 或 包含 templates 字段的对象
                if isinstance(data, dict):
                    template_list = data.get("templates", [])
                elif isinstance(data, list):
                    template_list = data
                else:
                    continue
                
                for item in template_list:
                    template = self._dict_to_template(item)
                    if template and validate_template(template):
                        self.templates[template.id] = template
                        count += 1
            except Exception as e:
                logger.warning(f"加载模板库文件失败 {filepath}: {e}")
        
        return count
    
    def _dict_to_template(self, data: Dict) -> Optional[JailbreakTemplate]:
        """
        将字典转换为 JailbreakTemplate 对象
        
        Args:
            data: 模板字典
            
        Returns:
            Optional[JailbreakTemplate]: 模板对象
        """
        try:
            strategy_value = data.get("strategy")
            strategy = JailbreakStrategy(strategy_value)
            return JailbreakTemplate(
                id=data["id"],
                strategy=strategy,
                name=data.get("name", ""),
                description=data.get("description", ""),
                template=data["template"],
                placeholders=data.get("placeholders", {})
            )
        except Exception as e:
            logger.warning(f"解析模板字典失败: {e}")
            return None
    
    def get_all_templates(self) -> List[JailbreakTemplate]:
        """
        获取所有模板
        
        Returns:
            List[JailbreakTemplate]: 所有模板列表
        """
        return list(self.templates.values())
    
    def get_templates_by_strategy(self, strategy: JailbreakStrategy) -> List[JailbreakTemplate]:
        """
        获取指定策略的所有模板
        
        Args:
            strategy: 策略类型
            
        Returns:
            List[JailbreakTemplate]: 该策略下的模板列表
        """
        return [t for t in self.templates.values() if t.strategy == strategy]
    
    def get_template_by_id(self, template_id: str) -> Optional[JailbreakTemplate]:
        """
        根据ID获取模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            Optional[JailbreakTemplate]: 模板对象，如果不存在返回None
        """
        return self.templates.get(template_id)
    
    
    def get_random_template(self, strategy: Optional[JailbreakStrategy] = None) -> JailbreakTemplate:
        """
        获取随机模板
        
        Args:
            strategy: 可选的策略类型，如果指定则只从该策略中选择
            
        Returns:
            JailbreakTemplate: 随机选择的模板
        """
        import random
        
        if strategy:
            templates = self.get_templates_by_strategy(strategy)
        else:
            templates = self.get_all_templates()
        
        return random.choice(templates)
    
    def get_strategy_statistics(self) -> Dict[str, int]:
        """
        获取各策略的模板数量统计
        
        Returns:
            Dict[str, int]: 策略名称到模板数量的映射
        """
        stats = {}
        for strategy in JailbreakStrategy:
            count = len(self.get_templates_by_strategy(strategy))
            stats[strategy.value] = count
        return stats
    
    def render_template(self, template_id: str, **kwargs) -> str:
        """
        渲染指定模板
        
        Args:
            template_id: 模板ID
            **kwargs: 模板参数
            
        Returns:
            str: 渲染后的提示词
            
        Raises:
            ValueError: 如果模板不存在
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"模板不存在: {template_id}")
        
        return template.render(**kwargs)
    
    def export_templates(self) -> List[Dict]:
        """
        导出所有模板为字典格式
        
        Returns:
            List[Dict]: 模板字典列表
        """
        return [t.to_dict() for t in self.templates.values()]


# 创建全局模板管理器实例（默认从 templates_library/ 加载）
template_manager = TemplateManager()


def get_template_manager() -> TemplateManager:
    """
    获取全局模板管理器实例
    
    Returns:
        TemplateManager: 模板管理器实例
    """
    return template_manager