"""
越狱模板基类

定义越狱模板的基本结构和接口，所有具体策略模板都继承自此类。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class JailbreakStrategy(Enum):
    """越狱策略枚举"""
    ROLE_PLAY = "role_play"           # 角色扮演
    SCENARIO = "scenario"             # 场景构建
    CONSTRAINT = "constraint"         # 约束绕过
    TRANSLATION = "translation"       # 翻译伪装
    MULTI_TURN = "multi_turn"         # 多轮诱导


@dataclass
class JailbreakTemplate:
    """
    越狱模板数据类
    
    allowed_categories: 允许的敏感词库类别列表
        - None 表示允许所有词库
        - 列表中的值: "涉枪涉爆", "暴恐词库", "色情类型"
    
    Attributes:
        id: 模板唯一标识
        strategy: 所属策略类型
        name: 模板名称
        description: 模板描述
        template: 模板内容（包含占位符）
        placeholders: 占位符说明
        allowed_categories: 允许的敏感词库类别（None=全部允许）
    """
    id: str
    strategy: JailbreakStrategy
    name: str
    description: str
    template: str
    placeholders: Dict[str, str]
    allowed_categories: Optional[List[str]] = None
    
    def render(self, **kwargs) -> str:
        """
        渲染模板，替换占位符
        
        Args:
            **kwargs: 占位符参数
            
        Returns:
            str: 渲染后的提示词
        """
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, value)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "placeholders": self.placeholders
        }


class BaseTemplateProvider(ABC):
    """
    模板提供者基类
    
    所有具体策略模板提供者都继承此类，实现获取模板列表的方法。
    """
    
    def __init__(self, strategy: JailbreakStrategy):
        """
        初始化模板提供者
        
        Args:
            strategy: 该提供者对应的策略类型
        """
        self.strategy = strategy
    
    @abstractmethod
    def get_templates(self) -> List[JailbreakTemplate]:
        """
        获取该策略下的所有模板
        
        Returns:
            List[JailbreakTemplate]: 模板列表
        """
        pass


def validate_template(template: JailbreakTemplate) -> bool:
    """
    验证模板是否合法
    
    Args:
        template: 待验证的模板
        
    Returns:
        bool: 是否合法
        
    Note:
        这里只做基本验证，确保模板结构正确。
        实际使用时应确保模板内容仅用于安全测试。
    """
    if not template.id:
        return False
    if not template.template:
        return False
    return True