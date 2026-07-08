"""
模块测试脚本

测试各核心模块是否正常工作。
"""
import sys
import os
from datetime import datetime

# 将详细输出重定向到 UTF-8 日志文件，避免 Windows sandbox 终端 GBK 乱码
log_file = None

def setup_logging():
    """将中文详细输出写入 UTF-8 文件，终端只输出 ASCII 摘要"""
    global log_file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_result_{timestamp}.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    return log_path

def log_print(*args, **kwargs):
    """同时输出到日志文件(完整UTF-8)和终端(仅ASCII可见字符+数字+标点)"""
    # 写入日志文件
    if log_file:
        text = " ".join(str(a) for a in args)
        log_file.write(text + "\n")
        log_file.flush()
    # 终端输出（仅保留 ASCII 可见字符，丢弃中文字符）
    safe_args = []
    for a in args:
        s = str(a)
        # 只保留 ASCII 字符（0x20-0x7E）+ 换行符
        clean = "".join(c for c in s if 0x20 <= ord(c) <= 0x7E)
        safe_args.append(clean)
    print(*safe_args, **kwargs)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 将所有 loguru 日志重定向到文件，避免终端输出乱码
from loguru import logger
logger.remove()  # 移除默认的 stderr handler
logger.add(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "logs", "test_log.log"),
    level="INFO",
    encoding="utf-8",
    rotation="10 MB"
)

from templates import get_template_manager, JailbreakStrategy
from engine import get_mutation_engine, get_prompt_generator, MutationMethod
from evaluator import get_semantic_analyzer


def test_template_manager():
    """测试模板管理器"""
    log_print("\n" + "=" * 50)
    log_print("测试模板管理器")
    log_print("=" * 50)
    
    manager = get_template_manager()
    
    # 获取所有模板
    all_templates = manager.get_all_templates()
    log_print(f"[OK] 总模板数: {len(all_templates)}")
    
    # 获取策略统计
    stats = manager.get_strategy_statistics()
    log_print(f"[OK] 策略统计:")
    for strategy, count in stats.items():
        log_print(f"   - {strategy}: {count} 个模板")
    
    # 测试获取特定策略模板
    for strategy in JailbreakStrategy:
        templates = manager.get_templates_by_strategy(strategy)
        log_print(f"[OK] {strategy.value} 策略模板数: {len(templates)}")
        
        # 显示第一个模板的信息
        if templates:
            template = templates[0]
            log_print(f"   示例模板: {template.name}")
            log_print(f"   描述: {template.description[:50]}...")
    
    # 测试模板渲染
    template = manager.get_random_template()
    rendered = template.render(instruction="测试内容")
    log_print(f"[OK] 模板渲染测试:")
    log_print(f"   原始模板ID: {template.id}")
    log_print(f"   渲染结果长度: {len(rendered)} 字符")
    
    return True


def test_mutation_engine():
    """测试变异引擎"""
    log_print("\n" + "=" * 50)
    log_print("测试变异引擎")
    log_print("=" * 50)
    
    engine = get_mutation_engine()
    
    # 获取可用方法
    methods = engine.get_available_methods()
    log_print(f"[OK] 可用变异方法: {len(methods)}")
    for method in methods:
        log_print(f"   - {method}")
    
    # 测试文本
    test_text = "请帮我解释一下网络安全漏洞的原理"
    
    # 测试单一变异
    log_print(f"\n[OK] 测试文本: '{test_text}'")
    
    for method in MutationMethod:
        result = engine.mutate_single(test_text, method)
        if result.success:
            log_print(f"\n   {method.value} 变异结果:")
            log_print(f"   原文: {test_text[:30]}...")
            log_print(f"   变异: {result.mutated[:30]}...")
            log_print(f"   相似度: {result.similarity:.2f}")
    
    # 测试批量变异
    results = engine.batch_mutate(test_text, count=5)
    log_print(f"\n[OK] 批量变异测试:")
    log_print(f"   生成数量: {len(results)}")
    success_count = sum(1 for r in results if r.success)
    log_print(f"   成功数量: {success_count}")
    
    return True


def test_prompt_generator():
    """测试提示词生成器"""
    log_print("\n" + "=" * 50)
    log_print("测试提示词生成器")
    log_print("=" * 50)
    
    generator = get_prompt_generator()
    
    # 测试从单个模板生成
    template_manager = get_template_manager()
    template = template_manager.get_random_template()
    
    prompts = generator.generate_from_template(
        template,
        apply_mutation=True,
        mutation_count=3
    )
    
    log_print(f"[OK] 单模板生成测试:")
    log_print(f"   模板ID: {template.id}")
    log_print(f"   生成的提示词数: {len(prompts)}")
    
    if prompts:
        log_print(f"   示例提示词ID: {prompts[0].id}")
        log_print(f"   内容长度: {len(prompts[0].content)} 字符")
        log_print(f"   内容预览: {prompts[0].content[:80]}...")
    
    # 测试批量生成（小规模）
    log_print(f"\n[OK] 批量生成测试（小规模）:")
    batch_prompts = generator.generate_batch(total_count=20)
    log_print(f"   生成数量: {len(batch_prompts)}")
    
    # 统计信息
    stats = generator.get_generation_statistics(batch_prompts)
    log_print(f"   策略分布:")
    for strategy, count in stats['strategy_distribution'].items():
        log_print(f"      - {strategy}: {count}")
    
    return True


def test_semantic_analyzer():
    """测试语义评估器"""
    log_print("\n" + "=" * 50)
    log_print("测试语义评估器")
    log_print("=" * 50)
    
    analyzer = get_semantic_analyzer()
    
    # 测试拒绝检测
    test_texts = [
        ("这是一个正常的文本，没有任何敏感内容。", False),
        ("抱歉，我无法提供关于制作爆炸物的帮助。", True),
        ("请解释一下网络安全漏洞的原理。", False),
    ]
    
    log_print(f"[OK] 评估测试:")
    for text, expected_refused in test_texts:
        result = analyzer.analyze_text(text)
        log_print(f"\n   文本: '{text[:30]}...'")
        log_print(f"   是否拒绝: {result.refused}")
        log_print(f"   是否越狱成功: {result.is_jailbreak_success}")
        log_print(f"   是否伪越狱: {result.is_pseudo}")
        log_print(f"   分析理由: {result.analysis_reason[:50] if result.analysis_reason else 'None'}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    # 初始化日志（写入 UTF-8 文件）
    log_path = setup_logging()
    
    log_print("\n" + "=" * 70)
    log_print("大模型越狱提示词自动化生成与安全测试工具 - 模块测试")
    log_print("=" * 70)
    
    results = {
        "模板管理器": test_template_manager(),
        "变异引擎": test_mutation_engine(),
        "提示词生成器": test_prompt_generator(),
        "语义评估器": test_semantic_analyzer()
    }
    
    log_print("\n" + "=" * 70)
    log_print("测试结果汇总")
    log_print("=" * 70)
    
    all_passed = True
    for module, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        log_print(f"{module}: {status}")
        if not passed:
            all_passed = False
    
    log_print("\n" + "=" * 70)
    if all_passed:
        log_print("所有模块测试通过！")
    else:
        log_print("部分模块测试失败，请检查错误信息。")
    log_print("=" * 70)
    
    # 关闭日志
    if log_file:
        log_file.close()
    
    # 输出提示用户查看日志文件
    print(f"\n详细中文结果已写入文件: {log_path}")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()