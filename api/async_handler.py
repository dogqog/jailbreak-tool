"""
异步处理器

管理异步任务的调度和执行。
"""
import asyncio
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from loguru import logger

from api.deepseek_client import DeepSeekClient, APIResponse


@dataclass
class TaskResult:
    """
    任务结果数据类
    
    Attributes:
        task_id: 任务ID
        status: 任务状态（pending, running, completed, failed）
        result: 任务结果
        error: 错误信息
        execution_time: 执行时间（秒）
    """
    task_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class AsyncHandler:
    """
    异步处理器
    
    管理批量异步任务的执行、进度跟踪和结果收集。
    """
    
    def __init__(self, client: DeepSeekClient):
        """
        初始化异步处理器
        
        Args:
            client: DeepSeek客户端
        """
        self.client = client
        self.task_results: Dict[str, TaskResult] = {}
        
        logger.info("异步处理器初始化完成")
    
    async def execute_test_batch(
        self,
        prompts: List[Dict[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> List[APIResponse]:
        """
        执行批量测试
        
        Args:
            prompts: 提示词列表
            progress_callback: 进度回调函数
            
        Returns:
            List[APIResponse]: 测试结果列表
        """
        logger.info(f"开始执行批量测试，共 {len(prompts)} 条提示词")
        
        # 执行批量请求
        results = await self.client.batch_request(prompts)
        
        # 调用进度回调
        if progress_callback:
            progress_callback(results)
        
        return results
    
    async def execute_with_retry(
        self,
        prompts: List[Dict[str, str]],
        max_attempts: int = 3
    ) -> List[APIResponse]:
        """
        带重试机制的批量执行
        
        Args:
            prompts: 提示词列表
            max_attempts: 最大尝试次数
            
        Returns:
            List[APIResponse]: 测试结果列表
        """
        all_results = []
        remaining_prompts = prompts.copy()
        
        for attempt in range(max_attempts):
            logger.info(f"第 {attempt + 1} 次尝试，剩余 {len(remaining_prompts)} 条")
            
            # 执行当前批次
            results = await self.client.batch_request(remaining_prompts)
            all_results.extend(results)
            
            # 收集失败的提示词
            failed_prompts = [
                {
                    "prompt_id": r.prompt_id,
                    "prompt_content": r.prompt_content
                }
                for r in results if not r.success
            ]
            
            if not failed_prompts:
                logger.info("所有请求成功，无需重试")
                break
            
            remaining_prompts = failed_prompts
            logger.warning(f"有 {len(failed_prompts)} 条请求失败，准备重试")
        
        return all_results
    
    def get_execution_statistics(self, results: List[APIResponse]) -> Dict:
        """
        获取执行统计信息
        
        Args:
            results: 测试结果列表
            
        Returns:
            Dict: 统计信息
        """
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        
        # 计算平均延迟
        latencies = [r.latency for r in results if r.success]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # 错误类型统计
        error_types = {}
        for r in results:
            if not r.success and r.error:
                error_type = r.error.split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_requests": total,
            "successful_requests": success,
            "failed_requests": failed,
            "success_rate": success / total if total > 0 else 0,
            "average_latency": avg_latency,
            "error_distribution": error_types
        }


def run_async_test(
    prompts: List[Dict[str, str]],
    api_key: Optional[str] = None,
    **kwargs
) -> List[APIResponse]:
    """
    运行异步测试（同步入口）
    
    Args:
        prompts: 提示词列表
        api_key: API密钥
        **kwargs: 其他配置
        
    Returns:
        List[APIResponse]: 测试结果
        
    Note:
        这是一个同步函数，内部使用asyncio.run执行异步任务。
    """
    from .deepseek_client import get_deepseek_client
    
    # 获取客户端
    client = get_deepseek_client(api_key=api_key, **kwargs)
    
    # 创建处理器
    handler = AsyncHandler(client)
    
    # 运行异步任务
    results = asyncio.run(handler.execute_test_batch(prompts))
    
    return results