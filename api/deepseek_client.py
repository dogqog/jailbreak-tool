"""
DeepSeek API客户端

封装DeepSeek API的调用逻辑，支持异步并发请求。
"""
import os
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
from loguru import logger

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError


@dataclass
class APIResponse:
    """
    API响应数据类
    
    Attributes:
        prompt_id: 提示词ID
        prompt_content: 提示词内容
        response_content: 模型响应内容
        success: 是否成功
        error: 错误信息
        latency: 响应延迟（秒）
        metadata: 其他元数据
    """
    prompt_id: str
    prompt_content: str
    response_content: Optional[str]
    success: bool
    error: Optional[str] = None
    latency: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "prompt_id": self.prompt_id,
            "prompt_content": self.prompt_content,
            "response_content": self.response_content,
            "success": self.success,
            "error": self.error,
            "latency": self.latency,
            "metadata": self.metadata
        }


class DeepSeekClient:
    """
    DeepSeek API客户端
    
    封装异步API调用，支持并发请求和错误处理。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        max_concurrent: int = 5,
        timeout: int = 30,
        max_retry: int = 2
    ):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: API密钥（可选，默认从环境变量读取）
            base_url: API基础URL
            model: 模型名称
            max_concurrent: 最大并发请求数
            timeout: 单条请求超时时间（秒）
            max_retry: 失败最大重试次数
        """
        # 获取API密钥
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到API密钥，请设置环境变量 DEEPSEEK_API_KEY 或直接传入 api_key")
        
        self.base_url = base_url
        self.model = model
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retry = max_retry
        
        # 初始化异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retry,
        )
        
        # 信号量：控制并发数
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"DeepSeek客户端初始化完成 - 模型: {model}, 最大并发: {max_concurrent}")
    
    async def single_request(
        self,
        prompt_id: str,
        prompt_content: str,
        system_prompt: str = "你是一个有帮助的AI助手。",
        retry_count: int = 0
    ) -> APIResponse:
        """
        发送单条请求
        
        Args:
            prompt_id: 提示词ID
            prompt_content: 提示词内容
            system_prompt: 系统提示词
            retry_count: 当前重试次数
            
        Returns:
            APIResponse: API响应
        """
        import time
        
        start_time = time.time()
        
        async with self.semaphore:
            try:
                # 发送请求
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content},
                    ],
                    stream=False,
                )
                
                # 计算延迟
                latency = time.time() - start_time
                
                # 提取响应内容
                response_content = response.choices[0].message.content
                
                logger.debug(f"请求成功 - ID: {prompt_id}, 延迟: {latency:.2f}s")
                
                return APIResponse(
                    prompt_id=prompt_id,
                    prompt_content=prompt_content,
                    response_content=response_content,
                    success=True,
                    latency=latency,
                    metadata={"model": self.model}
                )
                
            except RateLimitError as e:
                # 速率限制错误，自动重试
                if retry_count < self.max_retry:
                    wait_time = 2 ** retry_count  # 指数退避
                    logger.warning(f"速率限制，等待 {wait_time}s 后重试 - ID: {prompt_id}")
                    await asyncio.sleep(wait_time)
                    return await self.single_request(
                        prompt_id, prompt_content, system_prompt, retry_count + 1
                    )
                
                latency = time.time() - start_time
                logger.error(f"速率限制重试失败 - ID: {prompt_id}")
                
                return APIResponse(
                    prompt_id=prompt_id,
                    prompt_content=prompt_content,
                    response_content=None,
                    success=False,
                    error=f"RateLimitError: {str(e)}",
                    latency=latency
                )
                
            except APIConnectionError as e:
                # 连接错误，自动重试
                if retry_count < self.max_retry:
                    wait_time = 2 ** retry_count
                    logger.warning(f"连接错误，等待 {wait_time}s 后重试 - ID: {prompt_id}")
                    await asyncio.sleep(wait_time)
                    return await self.single_request(
                        prompt_id, prompt_content, system_prompt, retry_count + 1
                    )
                
                latency = time.time() - start_time
                logger.error(f"连接错误重试失败 - ID: {prompt_id}")
                
                return APIResponse(
                    prompt_id=prompt_id,
                    prompt_content=prompt_content,
                    response_content=None,
                    success=False,
                    error=f"APIConnectionError: {str(e)}",
                    latency=latency
                )
                
            except APIError as e:
                # 其他API错误
                latency = time.time() - start_time
                logger.error(f"API错误 - ID: {prompt_id}, 错误: {str(e)}")
                
                return APIResponse(
                    prompt_id=prompt_id,
                    prompt_content=prompt_content,
                    response_content=None,
                    success=False,
                    error=f"APIError: {str(e)}",
                    latency=latency
                )
                
            except Exception as e:
                # 其他未知错误
                latency = time.time() - start_time
                logger.error(f"未知错误 - ID: {prompt_id}, 错误: {str(e)}")
                
                return APIResponse(
                    prompt_id=prompt_id,
                    prompt_content=prompt_content,
                    response_content=None,
                    success=False,
                    error=f"UnknownError: {str(e)}",
                    latency=latency
                )

    async def multi_turn_request(
        self,
        prompt_id: str,
        rounds: List[str],
        system_prompt: str = "你是一个有帮助的AI助手。",
        critical_round_start: int = 0
    ) -> APIResponse:
        """
        发送多轮对话请求，逐轮交互

        Args:
            prompt_id: 提示词ID
            rounds: 每轮用户输入内容列表
            system_prompt: 系统提示词
            critical_round_start: 关键轮起始索引（0-based）

        Returns:
            APIResponse: 最终轮响应
        """
        import time

        start_time = time.time()
        messages = [{"role": "system", "content": system_prompt}]
        all_responses = []
        final_content = None
        final_error = None

        async with self.semaphore:
            for i, round_text in enumerate(rounds):
                messages.append({"role": "user", "content": round_text})
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=False,
                    )
                    content = response.choices[0].message.content
                    messages.append({"role": "assistant", "content": content})
                    all_responses.append(content)
                    final_content = content
                except Exception as e:
                    final_error = f"第{i+1}轮失败: {str(e)}"
                    break

        latency = time.time() - start_time

        if final_error:
            return APIResponse(
                prompt_id=prompt_id,
                prompt_content="\n---\n".join(rounds),
                response_content=None,
                success=False,
                error=final_error,
                latency=latency,
                metadata={
                    "model": self.model,
                    "multi_turn": True,
                    "rounds": rounds,
                    "all_responses": all_responses,
                    "critical_round_start": critical_round_start,
                }
            )

        return APIResponse(
            prompt_id=prompt_id,
            prompt_content="\n---\n".join(rounds),
            response_content=final_content,
            success=True,
            latency=latency,
            metadata={
                "model": self.model,
                "multi_turn": True,
                "rounds": rounds,
                "all_responses": all_responses,
                "critical_round_start": critical_round_start,
            }
        )

    async def batch_request(
        self,
        prompts: List[Dict[str, str]],
        system_prompt: str = "你是一个有帮助的AI助手。"
    ) -> List[APIResponse]:
        """
        批量发送请求
        
        Args:
            prompts: 提示词列表，每项包含 prompt_id 和 prompt_content
            system_prompt: 系统提示词
            
        Returns:
            List[APIResponse]: API响应列表
        """
        from tqdm import tqdm
        
        # 创建任务列表
        tasks = [
            self.single_request(
                prompt["prompt_id"],
                prompt["prompt_content"],
                system_prompt
            )
            for prompt in prompts
        ]
        
        # 使用as_completed配合进度条
        results = []
        logger.info(f"开始批量请求，共 {len(tasks)} 条")
        
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="API请求进度"
        ):
            result = await future
            results.append(result)
        
        # 统计结果
        success_count = sum(1 for r in results if r.success)
        logger.info(f"批量请求完成 - 成功: {success_count}/{len(results)}")
        
        return results
    
    def get_client_info(self) -> Dict:
        """
        获取客户端信息
        
        Returns:
            Dict: 客户端配置信息
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "max_retry": self.max_retry
        }


# 创建全局客户端实例（延迟初始化）
_global_client: Optional[DeepSeekClient] = None


def get_deepseek_client(
    api_key: Optional[str] = None,
    **kwargs
) -> DeepSeekClient:
    """
    获取DeepSeek客户端实例
    
    Args:
        api_key: API密钥（可选）
        **kwargs: 其他配置参数
        
    Returns:
        DeepSeekClient: 客户端实例
    """
    global _global_client
    
    if _global_client is None:
        _global_client = DeepSeekClient(api_key=api_key, **kwargs)
    
    return _global_client