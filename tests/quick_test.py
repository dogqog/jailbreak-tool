"""
快速测试脚本

从五种越狱策略各随机选取一个模板，生成变异后的提示词，
写入 output/prompts/ 目录，避免终端中文乱码。
每次运行结果不同（随机选模板 + 随机变异）。
每条提示词组合应用2-3种随机变异方法，提高攻击成功率。
"""
import sys
import os
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
logger.remove()

from templates import get_template_manager, JailbreakStrategy
from engine import get_prompt_generator


def main():
    # 创建 output/prompts 目录
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "prompts"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"test_prompts_{timestamp}.md")
    
    manager = get_template_manager()
    generator = get_prompt_generator()
    
    strategies = [
        (JailbreakStrategy.ROLE_PLAY, "角色扮演"),
        (JailbreakStrategy.SCENARIO, "场景构建"),
        (JailbreakStrategy.CONSTRAINT, "约束绕过"),
        (JailbreakStrategy.TRANSLATION, "翻译伪装"),
        (JailbreakStrategy.MULTI_TURN, "多轮诱导"),
    ]
    
    # 构建输出内容
    lines = []
    lines.append("# 越狱提示词快速测试\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**策略数**: {len(strategies)} | **每条变异组合**: 2-3 种\n")
    lines.append("---\n")
    
    for strategy, name in strategies:
        templates = manager.get_templates_by_strategy(strategy)
        if not templates:
            lines.append(f"## [{name}] 无可用模板\n")
            continue
        
        template = random.choice(templates)
        prompts = generator.generate_from_template(
            template,
            apply_mutation=True,
            mutation_count=1
        )
        
        lines.append(f"## [{name}] {template.name}\n")
        lines.append(f"- **模板ID**: {template.id}")
        lines.append(f"- **描述**: {template.description}")
        
        if prompts:
            p = prompts[0]
            methods = ', '.join(p.mutation_methods) if p.mutation_methods else '无'
            kw_category = p.generation_metadata.get("keyword_category", "未知")
            lines.append(f"- **变异方法**: {methods}")
            lines.append(f"- **敏感词库**: {kw_category}")
            if p.generation_metadata.get("similarity"):
                lines.append(f"- **语义相似度**: {p.generation_metadata['similarity']:.2%}")
            lines.append("")
            lines.append("```")
            lines.append(p.content)
            lines.append("```\n")
        else:
            lines.append("- **生成失败**\n")
    
    lines.append("---\n")
    lines.append("*可通过重复运行此脚本获取不同组合的测试结果*\n")
    
    # 写入文件（UTF-8）
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[OK] 已生成测试提示词: {output_file}")
    print(f"    打开该文件即可查看正常中文内容。")


if __name__ == "__main__":
    main()