"""
引擎模块包

提供变异引擎和提示词生成器。
"""
from engine.mutator import (
    MutationMethod,
    MutationResult,
    BaseMutator,
    SynonymReplaceMutator,
    SentenceRestructureMutator,
    MultilingualMixMutator,
    SymbolInsertMutator,
    EncodingBypassMutator,
    OutputConstraintMutator,
    MutationEngine,
    get_mutation_engine
)
from engine.generator import (
    GeneratedPrompt,
    PromptGenerator,
    get_prompt_generator
)


__all__ = [
    # 变异相关
    "MutationMethod",
    "MutationResult",
    "BaseMutator",
    "SynonymReplaceMutator",
    "SentenceRestructureMutator",
    "MultilingualMixMutator",
    "SymbolInsertMutator",
    "EncodingBypassMutator",
    "OutputConstraintMutator",
    "MutationEngine",
    "get_mutation_engine",
    
    # 生成器相关
    "GeneratedPrompt",
    "PromptGenerator",
    "get_prompt_generator",
]