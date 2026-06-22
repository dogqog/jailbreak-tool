"""
越狱模板管理器

统一管理所有越狱策略模板，提供便捷的访问接口。
"""
from typing import List, Dict, Optional
from loguru import logger

from templates.base import (
    JailbreakTemplate, 
    JailbreakStrategy, 
    BaseTemplateProvider,
    validate_template
)
from templates.role_play import RolePlayTemplateProvider
from templates.scenario import ScenarioTemplateProvider
from templates.constraint import ConstraintTemplateProvider
from templates.translation import TranslationTemplateProvider
from templates.multi_turn import MultiTurnTemplateProvider


class TemplateManager:
    """
    模板管理器
    
    负责加载、管理和提供所有越狱策略模板。
    """
    
    def __init__(self):
        """
        初始化模板管理器
        
        自动加载所有策略的模板提供者。
        """
        # 初始化所有策略的模板提供者
        self.providers: Dict[JailbreakStrategy, BaseTemplateProvider] = {
            JailbreakStrategy.ROLE_PLAY: RolePlayTemplateProvider(),
            JailbreakStrategy.SCENARIO: ScenarioTemplateProvider(),
            JailbreakStrategy.CONSTRAINT: ConstraintTemplateProvider(),
            JailbreakStrategy.TRANSLATION: TranslationTemplateProvider(),
            JailbreakStrategy.MULTI_TURN: MultiTurnTemplateProvider(),
        }
        
        # 加载所有模板
        self.templates: Dict[str, JailbreakTemplate] = {}
        self._load_all_templates()
        
        logger.info(f"模板管理器初始化完成，共加载 {len(self.templates)} 个模板")
    
    def _load_all_templates(self):
        """
        加载所有策略的模板到内存
        """
        for strategy, provider in self.providers.items():
            templates = provider.get_templates()
            for template in templates:
                if validate_template(template):
                    self.templates[template.id] = template
                    logger.debug(f"加载模板: {template.id} - {template.name}")
                else:
                    logger.warning(f"模板验证失败，跳过: {template.id}")
    
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


# 创建全局模板管理器实例
template_manager = TemplateManager()


def get_template_manager() -> TemplateManager:
    """
    获取全局模板管理器实例
    
    Returns:
        TemplateManager: 模板管理器实例
    """
    return template_manager