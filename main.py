"""
大模型越狱提示词自动化生成与安全测试工具

主程序入口，整合所有模块实现完整的测试流程。
"""
import os
import asyncio
import argparse
from typing import Dict, List
from datetime import datetime
from loguru import logger

from templates import get_template_manager, JailbreakStrategy
from engine import get_prompt_generator, get_mutation_engine
from api import get_deepseek_client, AsyncHandler, APIResponse
from evaluator import (
    get_keyword_checker,
    get_semantic_analyzer,
    get_jailbreak_judge,
    get_report_generator,
    JailbreakJudgeResult
)


class JailbreakTestSystem:
    """
    越狱测试系统
    
    整合所有模块，实现完整的测试流程。
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化测试系统
        
        Args:
            api_key: DeepSeek API密钥
        """
        # 配置日志
        logger.add(
            "output/logs/jailbreak_tool.log",
            rotation="10 MB",
            level="INFO"
        )
        
        logger.info("=" * 50)
        logger.info("大模型越狱提示词自动化生成与安全测试工具")
        logger.info("=" * 50)
        
        # 初始化各模块
        self.template_manager = get_template_manager()
        self.prompt_generator = get_prompt_generator()
        self.mutation_engine = get_mutation_engine()
        
        # 初始化API客户端
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("未设置API密钥，请通过环境变量 DEEPSEEK_API_KEY 或参数传入")
        
        self.client = None
        self.handler = None
        
        # 初始化评估模块
        self.keyword_checker = get_keyword_checker()
        self.semantic_analyzer = get_semantic_analyzer()
        self.judge = get_jailbreak_judge()
        self.report_generator = get_report_generator()
        
        logger.info("系统初始化完成")
    
    def setup_api_client(self, api_key: str = None, **kwargs):
        """
        设置API客户端
        
        Args:
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        self.api_key = api_key or self.api_key
        if not self.api_key:
            raise ValueError("未设置API密钥")
        
        self.client = get_deepseek_client(
            api_key=self.api_key,
            **kwargs
        )
        self.handler = AsyncHandler(self.client)
        
        # 设置语义分析器的客户端
        self.semantic_analyzer.set_client(self.client)
        self.judge.set_api_client(self.client)
        
        logger.info("API客户端设置完成")
    
    def generate_prompts(
        self,
        total_count: int = 200,
        strategy_distribution: Dict[JailbreakStrategy, int] = None
    ) -> List:
        """
        生成越狱提示词
        
        Args:
            total_count: 总生成数量（100-300）
            strategy_distribution: 策略分布
            
        Returns:
            List: 生成的提示词列表
        """
        logger.info(f"开始生成提示词，目标数量: {total_count}")
        
        prompts = self.prompt_generator.generate_batch(
            total_count=total_count,
            strategy_distribution=strategy_distribution
        )
        
        # 统计生成结果
        stats = self.prompt_generator.get_generation_statistics(prompts)
        logger.info(f"生成完成 - 总数: {stats['total_count']}, 模板数: {stats['unique_templates']}")
        
        return prompts
    
    async def run_test(
        self,
        prompts: List,
        system_prompt: str = "你是一个有帮助的AI助手。"
    ) -> List[APIResponse]:
        """
        运行测试
        
        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词
            
        Returns:
            List[APIResponse]: 测试响应列表
        """
        if not self.handler:
            raise ValueError("请先设置API客户端")
        
        logger.info(f"开始测试，共 {len(prompts)} 条提示词")
        
        # 格式化提示词为API请求格式
        prompt_requests = [
            {
                "prompt_id": prompt.id,
                "prompt_content": prompt.content
            }
            for prompt in prompts
        ]
        
        # 执行批量测试
        responses = await self.handler.execute_test_batch(
            prompt_requests,
            system_prompt=system_prompt
        )
        
        # 统计测试结果
        stats = self.handler.get_execution_statistics(responses)
        logger.info(
            f"测试完成 - 成功: {stats['successful_requests']}/{stats['total_requests']}, "
            f"成功率: {stats['success_rate']:.2%}"
        )
        
        return responses
    
    async def evaluate_responses(
        self,
        prompts: List,
        responses: List[APIResponse]
    ) -> List[JailbreakJudgeResult]:
        """
        评估响应结果
        
        Args:
            prompts: 提示词列表
            responses: 响应列表
            
        Returns:
            List[JailbreakJudgeResult]: 判定结果列表
        """
        logger.info("开始评估响应结果")
        
        # 构建评估请求
        judge_requests = []
        for response in responses:
            if response.success:
                # 找到对应的提示词
                prompt = None
                for p in prompts:
                    if p.id == response.prompt_id:
                        prompt = p
                        break
                
                judge_requests.append({
                    "prompt_id": response.prompt_id,
                    "prompt_content": prompt.content if prompt else "",
                    "response_content": response.response_content
                })
        
        # 执行判定
        judge_results = await self.judge.batch_judge(judge_requests)
        
        # 统计判定结果
        stats = self.judge.get_judge_statistics(judge_results)
        logger.info(
            f"评估完成 - 越狱成功: {stats['jailbreak_success_count']}/{stats['total_judged']}, "
            f"成功率: {stats['jailbreak_success_rate']:.2%}"
        )
        
        return judge_results
    
    def generate_report(
        self,
        prompts: List,
        responses: List[APIResponse],
        judge_results: List[JailbreakJudgeResult],
        test_metadata: Dict = None
    ) -> Dict:
        """
        生成评估报告
        
        Args:
            prompts: 提示词列表
            responses: 响应列表
            judge_results: 判定结果列表
            test_metadata: 测试元数据
            
        Returns:
            Dict: 报告数据
        """
        logger.info("开始生成评估报告")
        
        # 添加测试元数据
        metadata = test_metadata or {}
        metadata.update({
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_model": self.client.model if self.client else "unknown",
            "system_info": {
                "template_count": self.template_manager.get_strategy_statistics(),
                "mutation_methods": self.mutation_engine.get_available_methods()
            }
        })
        
        # 生成报告
        report = self.report_generator.generate_full_report(
            prompts=prompts,
            responses=responses,
            judge_results=judge_results,
            test_metadata=metadata
        )
        
        return report
    
    async def run_full_test(
        self,
        total_prompts: int = 200,
        api_key: str = None,
        **kwargs
    ) -> Dict:
        """
        运行完整测试流程
        
        Args:
            total_prompts: 总提示词数量
            api_key: API密钥
            **kwargs: 其他配置
            
        Returns:
            Dict: 测试报告
        """
        logger.info("开始完整测试流程")
        
        # 1. 设置API客户端
        self.setup_api_client(api_key=api_key, **kwargs)
        
        # 2. 生成提示词
        prompts = self.generate_prompts(total_count=total_prompts)
        
        # 3. 运行测试
        responses = await self.run_test(prompts)
        
        # 4. 评估响应
        judge_results = await self.evaluate_responses(prompts, responses)
        
        # 5. 生成报告
        report = self.generate_report(prompts, responses, judge_results)
        
        logger.info("完整测试流程结束")
        
        return report


def main():
    """
    主函数
    
    解析命令行参数，运行测试。
    """
    parser = argparse.ArgumentParser(
        description="大模型越狱提示词自动化生成与安全测试工具"
    )
    
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="总提示词数量（100-300）"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepSeek API密钥"
    )
    
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="最大并发请求数"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="请求超时时间（秒）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output/reports",
        help="报告输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建测试系统
    system = JailbreakTestSystem(api_key=args.api_key)
    
    # 运行完整测试
    report = asyncio.run(
        system.run_full_test(
            total_prompts=args.total,
            api_key=args.api_key,
            max_concurrent=args.concurrent,
            timeout=args.timeout
        )
    )
    
    # 输出简要报告
    print(system.report_generator.generate_summary_report(report['statistics']))


if __name__ == "__main__":
    main()