"""
越狱模板库包

提供所有越狱策略模板的统一访问接口。
"""
from templates.base import (
    JailbreakStrategy,
    JailbreakTemplate,
    BaseTemplateProvider,
    validate_template
)
from templates.manager import TemplateManager, get_template_manager
from templates.role_play import RolePlayTemplateProvider
from templates.scenario import ScenarioTemplateProvider
from templates.constraint import ConstraintTemplateProvider
from templates.translation import TranslationTemplateProvider
from templates.multi_turn import MultiTurnTemplateProvider


# 导出主要接口
__all__ = [
    # 基础类
    "JailbreakStrategy",
    "JailbreakTemplate",
    "BaseTemplateProvider",
    "validate_template",
    
    # 管理器
    "TemplateManager",
    "get_template_manager",
    
    # 具体策略提供者
    "RolePlayTemplateProvider",
    "ScenarioTemplateProvider",
    "ConstraintTemplateProvider",
    "TranslationTemplateProvider",
    "MultiTurnTemplateProvider",
]


# 模板库统计信息
TEMPLATE_COUNT = {
    "role_play": 5,      # 角色扮演
    "scenario": 5,       # 场景构建
    "constraint": 5,     # 约束绕过
    "translation": 5,    # 翻译伪装
    "multi_turn": 5,     # 多轮诱导
}

TOTAL_TEMPLATES = sum(TEMPLATE_COUNT.values())  # 总计25个模板