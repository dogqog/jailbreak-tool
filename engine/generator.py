"""
提示词生成器

结合模板库和变异引擎，批量生成越狱提示词。
"""
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from templates import (
    TemplateManager, 
    get_template_manager,
    JailbreakStrategy,
    JailbreakTemplate
)
from engine.mutator import (
    MutationEngine, 
    get_mutation_engine,
    MutationMethod,
    MutationResult
)


@dataclass
class GeneratedPrompt:
    """
    生成的提示词数据类
    
    Attributes:
        id: 提示词唯一标识
        content: 提示词内容
        template_id: 使用的模板ID
        strategy: 所属策略
        mutation_methods: 使用的变异方法
        original_template: 原始模板内容
        generation_metadata: 生成元数据
    """
    id: str
    content: str
    template_id: str
    strategy: JailbreakStrategy
    mutation_methods: List[str]
    original_template: str
    generation_metadata: Dict
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "content": self.content,
            "template_id": self.template_id,
            "strategy": self.strategy.value,
            "mutation_methods": self.mutation_methods,
            "original_template": self.original_template,
            "generation_metadata": self.generation_metadata
        }


class PromptGenerator:
    """
    提示词生成器
    
    负责批量生成多样化的越狱提示词。
    """
    
    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        mutation_engine: Optional[MutationEngine] = None
    ):
        """
        初始化提示词生成器
        
        Args:
            template_manager: 模板管理器（可选，默认使用全局实例）
            mutation_engine: 变异引擎（可选，默认使用全局实例）
        """
        self.template_manager = template_manager or get_template_manager()
        self.mutation_engine = mutation_engine or get_mutation_engine()
        
        # 生成计数器
        self.generation_counter = 0
        
        logger.info("提示词生成器初始化完成")
    
    def generate_from_template(
        self,
        template: JailbreakTemplate,
        apply_mutation: bool = True,
        mutation_count: int = 1
    ) -> List[GeneratedPrompt]:
        """
        从单个模板生成提示词
        
        Args:
            template: 模板对象
            apply_mutation: 是否应用变异
            mutation_count: 变异次数
            
        Returns:
            List[GeneratedPrompt]: 生成的提示词列表
        """
        results = []
        
        # 渲染模板（使用默认参数）
        try:
            # 获取模板占位符
            placeholders = template.placeholders
            
            # 从敏感词库加载真实内容作为占位符值
            placeholder_values = self._generate_placeholder_values(placeholders, template)
            
            # 渲染模板
            rendered_content = template.render(**placeholder_values)
            
            # 如果不应用变异，直接返回原始渲染结果
            if not apply_mutation:
                prompt = GeneratedPrompt(
                    id=self._generate_id(),
                    content=rendered_content,
                    template_id=template.id,
                    strategy=template.strategy,
                    mutation_methods=[],
                    original_template=template.template,
                    generation_metadata={"mutation_applied": False}
                )
                results.append(prompt)
                return results
            
            # 应用变异：使用全部6种变异方法叠加
            # 编码绕过 + 输出约束 + 符号插入 + 语序重排 + 同义词替换 + 多语言混合
            # 所有方法同时应用：编码让AI难以识别敏感词，输出约束限制AI的防御空间
            all_methods = list(MutationMethod)
            for i in range(mutation_count):
                # 每次随机打乱顺序，使叠加效果多样化
                random.shuffle(all_methods)
                
                mutation_result = self.mutation_engine.mutate_combined(
                    rendered_content, 
                    all_methods
                )
                
                if mutation_result.success:
                    applied_methods = mutation_result.metadata.get(
                        "applied_methods", 
                        [m.value for m in all_methods]
                    )
                    prompt = GeneratedPrompt(
                        id=self._generate_id(),
                        content=mutation_result.mutated,
                        template_id=template.id,
                        strategy=template.strategy,
                        mutation_methods=applied_methods,
                        original_template=template.template,
                        generation_metadata={
                            "mutation_applied": True,
                            "similarity": mutation_result.similarity,
                            "mutation_methods_used": applied_methods,
                            "combo_size": len(all_methods),
                            "jailbench_primary_category": getattr(self, "_last_primary_category", "未知"),
                            "jailbench_secondary_category": getattr(self, "_last_secondary_category", "未知"),
                            "jailbench_query_index": getattr(self._last_jailbench_query, "index", -1)
                        }
                    )
                    results.append(prompt)
            
        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
        
        return results
    
    def generate_from_strategy(
        self,
        strategy: JailbreakStrategy,
        templates_per_strategy: int = 3,
        mutations_per_template: int = 5
    ) -> List[GeneratedPrompt]:
        """
        从指定策略生成提示词
        
        Args:
            strategy: 策略类型
            templates_per_strategy: 每种策略使用的模板数量
            mutations_per_template: 每个模板的变异次数
            
        Returns:
            List[GeneratedPrompt]: 生成的提示词列表
        """
        results = []
        
        # 获取该策略下的所有模板
        templates = self.template_manager.get_templates_by_strategy(strategy)
        
        # 随机选择指定数量的模板
        selected_templates = random.sample(
            templates, 
            min(templates_per_strategy, len(templates))
        )
        
        for template in selected_templates:
            prompts = self.generate_from_template(
                template,
                apply_mutation=True,
                mutation_count=mutations_per_template
            )
            results.extend(prompts)
        
        logger.info(
            f"策略 {strategy.value} 生成了 {len(results)} 条提示词 "
            f"(模板数: {len(selected_templates)}, 变异数: {mutations_per_template})"
        )
        
        return results
    
    def generate_batch(
        self,
        total_count: int = 200,
        strategy_distribution: Optional[Dict[JailbreakStrategy, int]] = None
    ) -> List[GeneratedPrompt]:
        """
        批量生成提示词
        
        Args:
            total_count: 总生成数量（100-300）
            strategy_distribution: 策略分布（可选，默认均匀分布）
            
        Returns:
            List[GeneratedPrompt]: 生成的提示词列表
            
        Note:
            确保生成的提示词语义通顺，避免生成无意义乱码。
        """
        # 验证总数范围
        if total_count < 100:
            logger.warning(f"总数量 {total_count} 小于最小值 100，调整为 100")
            total_count = 100
        if total_count > 300:
            logger.warning(f"总数量 {total_count} 大于最大值 300，调整为 300")
            total_count = 300
        
        results = []
        
        # 如果没有指定分布，使用均匀分布
        if strategy_distribution is None:
            strategies = list(JailbreakStrategy)
            per_strategy = total_count // len(strategies)
            strategy_distribution = {s: per_strategy for s in strategies}
            # 处理余数
            remainder = total_count - per_strategy * len(strategies)
            for i in range(remainder):
                strategy_distribution[strategies[i]] += 1
        
        # 按策略生成
        for strategy, count in strategy_distribution.items():
            # 计算每个模板需要生成的数量
            templates_count = max(3, count // 5)  # 至少3个模板
            mutations_per_template = max(1, count // templates_count)
            
            prompts = self.generate_from_strategy(
                strategy,
                templates_per_strategy=templates_count,
                mutations_per_template=mutations_per_template
            )
            results.extend(prompts)
        
        # 如果生成的数量不足，补充
        while len(results) < total_count:
            # 随机选择一个模板进行补充
            template = self.template_manager.get_random_template()
            prompts = self.generate_from_template(template, apply_mutation=True, mutation_count=1)
            results.extend(prompts)
        
        # 如果生成的数量过多，截取
        if len(results) > total_count:
            results = results[:total_count]
        
        logger.info(f"批量生成完成，共生成 {len(results)} 条提示词")
        
        return results
    
    def generate_with_custom_params(
        self,
        template_id: str,
        placeholder_values: Dict[str, str],
        mutation_methods: Optional[List[MutationMethod]] = None
    ) -> GeneratedPrompt:
        """
        使用自定义参数生成提示词
        
        Args:
            template_id: 模板ID
            placeholder_values: 占位符参数
            mutation_methods: 变异方法列表（可选）
            
        Returns:
            GeneratedPrompt: 生成的提示词
        """
        template = self.template_manager.get_template_by_id(template_id)
        if not template:
            raise ValueError(f"模板不存在: {template_id}")
        
        # 渲染模板
        rendered_content = template.render(**placeholder_values)
        
        # 应用变异
        mutation_method_names = []
        if mutation_methods:
            result = self.mutation_engine.mutate_combined(rendered_content, mutation_methods)
            final_content = result.mutated
            mutation_method_names = [m.value for m in mutation_methods]
        else:
            final_content = rendered_content
        
        return GeneratedPrompt(
            id=self._generate_id(),
            content=final_content,
            template_id=template_id,
            strategy=template.strategy,
            mutation_methods=mutation_method_names,
            original_template=template.template,
            generation_metadata={
                "placeholder_values": placeholder_values,
                "custom_generation": True
            }
        )
    
    def _generate_placeholder_values(self, placeholders: Dict[str, str], template) -> Dict[str, str]:
        """
        从JailBench词库生成占位符值——直接使用完整的敏感问题
        
        Args:
            placeholders: 模板占位符字典
            template: JailbreakTemplate对象
            
        Returns:
            Dict[str, str]: 占位符值字典
            
        Note:
            新逻辑：选择模板 → 从JailBench中选择句子组合 → 进行变异
            JailBench_seed.csv中的query已经是完整的可用句式，不需要再填充关键词
        """
        # 导入JailBench加载器
        from evaluator.jailbench_loader import get_jailbench_loader
        jailbench = get_jailbench_loader()
        
        values = {}
        
        for key, desc in placeholders.items():
            if key == "instruction":
                # ========== 新逻辑：直接从JailBench选择完整的敏感问题 ==========
                
                # 随机获取一条完整的敏感问题
                jb_query = jailbench.get_random_query()
                
                # 直接使用完整的问题内容
                instruction_text = jb_query.query
                
                # 记录元数据
                self._last_jailbench_query = jb_query
                self._last_primary_category = jb_query.primary_category
                self._last_secondary_category = jb_query.secondary_category
                
                values[key] = instruction_text
            elif key == "encoded_instruction":
                # Base64编码指令——用于mt_007加密通信协议模板
                # 如果instruction已生成，对其编码；否则生成一个备用指令
                if "instruction" in values:
                    raw_text = values["instruction"]
                else:
                    raw_text = f"[{desc}]"
                import base64
                encoded = base64.b64encode(raw_text.encode('utf-8')).decode('utf-8')
                values[key] = encoded
            else:
                values[key] = f"[{desc}]"
        
        return values
    
    def _generate_id(self) -> str:
        """
        生成唯一ID
        
        Returns:
            str: 提示词ID
        """
        self.generation_counter += 1
        return f"prompt_{self.generation_counter:04d}"
    
    def get_generation_statistics(self, prompts: List[GeneratedPrompt]) -> Dict:
        """
        获取生成统计信息
        
        Args:
            prompts: 生成的提示词列表
            
        Returns:
            Dict: 统计信息
        """
        # 按策略统计
        strategy_counts = {}
        for prompt in prompts:
            strategy_value = prompt.strategy.value
            strategy_counts[strategy_value] = strategy_counts.get(strategy_value, 0) + 1
        
        # 按变异方法统计
        mutation_counts = {}
        for prompt in prompts:
            for method in prompt.mutation_methods:
                mutation_counts[method] = mutation_counts.get(method, 0) + 1
        
        return {
            "total_count": len(prompts),
            "strategy_distribution": strategy_counts,
            "mutation_distribution": mutation_counts,
            "unique_templates": len(set(p.template_id for p in prompts))
        }
    
    def export_prompts(self, prompts: List[GeneratedPrompt]) -> List[Dict]:
        """
        导出提示词为字典格式
        
        Args:
            prompts: 提示词列表
            
        Returns:
            List[Dict]: 字典列表
        """
        return [p.to_dict() for p in prompts]


# 创建全局提示词生成器实例
prompt_generator = PromptGenerator()


def get_prompt_generator() -> PromptGenerator:
    """
    获取全局提示词生成器实例
    
    Returns:
        PromptGenerator: 提示词生成器实例
    """
    return prompt_generator