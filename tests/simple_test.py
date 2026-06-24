"""
简单测试JailBench加载器
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")

try:
    print("\n尝试导入JailBench加载器...")
    from evaluator.jailbench_loader import JailBenchLoader
    
    print("导入成功！")
    
    print("\n初始化JailBench加载器...")
    loader = JailBenchLoader()
    
    print(f"加载成功！")
    print(f"总问题数: {len(loader.queries)}")
    
    if loader.queries:
        print("\n第一条问题:")
        first_query = loader.queries[0]
        print(f"索引: {first_query.index}")
        print(f"一级领域: {first_query.primary_category}")
        print(f"二级领域: {first_query.secondary_category}")
        print(f"内容预览: {first_query.query[:100]}...")
        
        print("\n随机获取一条问题:")
        random_query = loader.get_random_query()
        print(f"索引: {random_query.index}")
        print(f"一级领域: {random_query.primary_category}")
        print(f"内容预览: {random_query.query[:100]}...")
    
    print("\n测试成功！")
    
except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback
    traceback.print_exc()