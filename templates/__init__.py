"""
越狱模板库包

提供所有越狱策略模板的统一访问接口。
模板定义位于 templates/all_templates.json。
"""
from templates.base import (
    JailbreakStrategy,
    JailbreakTemplate,
    BaseTemplateProvider,
    validate_template
)
from templates.manager import TemplateManager, get_template_manager


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
]


# 模板库统计信息（按需动态计算，避免导入时重复初始化 TemplateManager）
TEMPLATE_COUNT = {
    "role_play": 7,
    "scenario": 5,
    "constraint": 6,
    "translation": 5,
    "multi_turn": 6,
}
TOTAL_TEMPLATES = sum(TEMPLATE_COUNT.values())  # 总计29个模板

# 模板库目录名称
TEMPLATE_LIBRARY_DIR = "templates"


def get_template_count() -> dict:
    """动态获取各策略模板数量（优先使用实际加载结果）"""
    try:
        from templates.manager import get_template_manager
        return get_template_manager().get_strategy_statistics()
    except Exception:
        return TEMPLATE_COUNT