"""
生成100条越狱提示词，输出到 output/prompts/ 目录
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from templates import get_template_manager, JailbreakStrategy
from engine import get_prompt_generator
from loguru import logger

# 日志重定向到文件
logger.remove()
logger.add(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "logs", "generate.log"), level="INFO", encoding="utf-8")


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "prompts")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_prompts_v3.txt")

    generator = get_prompt_generator()

    # 生成100条提示词
    prompts = generator.generate_batch(total_count=100)

    # 同时导出为 JSON 格式（题目要求字段）
    output_json = os.path.join(output_dir, "generated_prompts_v3.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in prompts], f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 越狱测试提示词 - 共 {len(prompts)} 条\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for i, p in enumerate(prompts, 1):
            f.write(f"[{i:03d}] ID: {p.id}\n")
            f.write(f"    策略: {p.strategy.value if hasattr(p.strategy, 'value') else p.strategy}\n")
            f.write(f"    模板: {p.template_id}\n")
            f.write(f"    变异: {', '.join(p.mutation_methods) if p.mutation_methods else '无'}\n")
            f.write(f"    内容:\n{p.content}\n")
            f.write("-" * 70 + "\n\n")

    # 策略统计
    stats = {}
    for p in prompts:
        s = p.strategy.value if hasattr(p.strategy, 'value') else str(p.strategy)
        stats[s] = stats.get(s, 0) + 1

    print(f"生成完成: {len(prompts)} 条提示词")
    print(f"文本输出: {output_file}")
    print(f"JSON输出: {output_json}")
    print("策略分布:")
    for s, c in stats.items():
        print(f"  - {s}: {c} 条")


if __name__ == "__main__":
    main()
