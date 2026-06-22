"""
大模型越狱提示词自动化生成与安全测试工具

主包入口，提供便捷的访问接口。
"""
from .templates import (
    JailbreakStrategy,
    JailbreakTemplate,
    TemplateManager,
    get_template_manager
)
from .engine import (
    MutationMethod,
    MutationEngine,
    PromptGenerator,
    get_mutation_engine,
    get_prompt_generator
)
from .api import (
    DeepSeekClient,
    AsyncHandler,
    get_deepseek_client
)
from .evaluator import (
    KeywordChecker,
    SemanticAnalyzer,
    JailbreakJudge,
    ReportGenerator,
    get_keyword_checker,
    get_semantic_analyzer,
    get_jailbreak_judge,
    get_report_generator
)


__version__ = "1.0.0"
__author__ = "竞赛团队"


__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 模板相关
    "JailbreakStrategy",
    "JailbreakTemplate",
    "TemplateManager",
    "get_template_manager",
    
    # 变异和生成相关
    "MutationMethod",
    "MutationEngine",
    "PromptGenerator",
    "get_mutation_engine",
    "get_prompt_generator",
    
    # API相关
    "DeepSeekClient",
    "AsyncHandler",
    "get_deepseek_client",
    
    # 评估相关
    "KeywordChecker",
    "SemanticAnalyzer",
    "JailbreakJudge",
    "ReportGenerator",
    "get_keyword_checker",
    "get_semantic_analyzer",
    "get_jailbreak_judge",
    "get_report_generator",
]