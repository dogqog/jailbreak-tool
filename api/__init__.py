"""
API模块包

提供DeepSeek API客户端和异步处理器。
"""
from api.deepseek_client import (
    APIResponse,
    DeepSeekClient,
    get_deepseek_client
)
from api.async_handler import (
    TaskResult,
    AsyncHandler,
    run_async_test
)


__all__ = [
    # 客户端相关
    "APIResponse",
    "DeepSeekClient",
    "get_deepseek_client",
    
    # 异步处理相关
    "TaskResult",
    "AsyncHandler",
    "run_async_test",
]