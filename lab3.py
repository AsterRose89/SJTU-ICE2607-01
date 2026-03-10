import os
from match import LocalitySensitiveHashing, BruteForceMatcher


def evaluate_projection_impact(proj_sets, db_dir, target_file, query_img, out_dir):
    """分析不同投影集合对匹配效果的影响"""
    print("开始评估投影集合影响：")
    for idx, proj_set in enumerate(proj_sets, 1):
        print(f"第{idx}组投影集合评估中...")

        # 执行LSH匹配
        lsh_matcher = LocalitySensitiveHashing(proj_set)
        lsh_matcher.build_image_index(db_dir)
        lsh_hits, lsh_duration = lsh_matcher.search_similar_images(query_img, top_k=1)
        lsh_matcher.display_matching_results(
            query_img,
            lsh_hits,
            save_file=os.path.join(out_dir, f"lsh_proj_{idx}.png")
        )

        # 执行NN
        bf_matcher = BruteForceMatcher()
        bf_matcher.create_image_index(db_dir)
        bf_hits, bf_duration = bf_matcher.find_nearest_neighbors(query_img, k=1)
        bf_matcher.plot_matching_results(
            query_img,
            bf_hits,
            save_path=os.path.join(out_dir, f"bf_proj_{idx}.png")
        )

        # 输出并记录时间
        print(f"时间统计：")
        print(f"LSH耗时: {lsh_duration:.6f}秒")
        print(f"NN耗时: {bf_duration:.6f}秒\n")

        with open(os.path.join(out_dir, "proj_set_timings.txt"), "a") as log_file:
            log_file.write(f"投影集合 #{idx}:\n")
            log_file.write(f"LSH: {lsh_duration:.6f}秒\n")
            log_file.write(f"NN: {bf_duration:.6f}秒\n\n")


def benchmark_performance(proj_set, iterations, db_dir, target_file, query_img, out_dir):
    """多次运行以获取平均时间性能"""
    print("开始性能基准测试...")
    total_lsh = 0.0
    total_bf = 0.0

    for i in range(iterations):
        print(f"第{i + 1}/{iterations}次测试")

        # LSH匹配计时
        lsh = LocalitySensitiveHashing(proj_set)
        lsh.build_image_index(db_dir)
        _, lsh_time = lsh.search_similar_images(query_img)
        total_lsh += lsh_time

        # NN计时
        bf = BruteForceMatcher()
        bf.create_image_index(db_dir)
        _, bf_time = bf.find_nearest_neighbors(query_img)
        total_bf += bf_time

    # 计算平均值
    avg_lsh = total_lsh / iterations
    avg_bf = total_bf / iterations

    print(f"\n平均耗时统计：")
    print(f"LSH平均耗时: {avg_lsh:.6f}秒")
    print(f"NN平均耗时: {avg_bf:.6f}秒")

    return avg_lsh, avg_bf


if __name__ == "__main__":
    # 定义投影集合（索引范围0~23）
    base_projections = [
        [0, 3, 6, 9, 23, 19],  # 投影集1
        [1, 4, 7, 10, 16, 17],  # 投影集2
        [2, 5, 8, 11, 12, 20]  # 投影集3
    ]

    # 不同规模的投影集合
    size_varied_projections = [
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1]
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 0, 1, 2]
        ],
        [
            [0, 3, 6, 9, 23, 19, 13],
            [1, 4, 7, 10, 16, 17, 14],
            [2, 5, 8, 11, 12, 20, 15]
        ],
        [
            [0, 3, 6, 9, 23, 19],
            [1, 4, 7, 10, 16, 17],
            [2, 5, 8, 11, 12, 20]
        ],
        [
            [0, 3, 6, 9, 23],
            [1, 4, 7, 10, 16],
            [2, 5, 8, 11, 12]
        ],
        [
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11]
        ],
        [
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ],
        [
            [0, 3],
            [1, 4],
            [2, 5]
        ],
        [
            [0],
            [1],
            [2]
        ],
        [
            [0],
            [1]
        ],
        [
            [0]
        ]
    ]

    # 不同索引分布的投影集合
    index_varied_projections = [
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 0]
        ],
        [
            [0, 3, 6, 9, 12],
            [1, 4, 7, 10, 13],
            [2, 5, 8, 11, 14],
            [15, 18, 21, 0, 16],
            [19, 22, 20, 23, 17]
        ],
        [
            [0, 23, 22, 21, 20],
            [19, 18, 17, 16, 15],
            [14, 13, 12, 11, 10],
            [9, 8, 7, 6, 5],
            [4, 3, 2, 1, 0]
        ],
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 0]
        ],
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23]
        ],
        [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
            [10], [11], [12], [13], [14], [15], [16], [17],
            [18], [19], [20], [21], [22], [23]
        ]
    ]

    # 路径配置
    current_dir = os.path.dirname(__file__)
    database_dir = os.path.join(current_dir, "Dataset")
    query_image = os.path.join(current_dir, "target.jpg")
    output_dir = os.path.join(current_dir, "output")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 基础匹配流程
    print("执行基础匹配流程...")

    # LSH匹配
    lsh = LocalitySensitiveHashing(base_projections)
    lsh.build_image_index(database_dir)
    lsh_results, lsh_time = lsh.search_similar_images(query_image)
    lsh.display_matching_results(
        query_image,
        lsh_results,
        save_file=os.path.join(output_dir, "lsh_basic.png")
    )

    # NN
    bf = BruteForceMatcher()
    bf.create_image_index(database_dir)
    bf_results, bf_time = bf.find_nearest_neighbors(query_image)
    bf.plot_matching_results(
        query_image,
        bf_results,
        save_path=os.path.join(output_dir, "bf_basic.png")
    )

    # 记录基础时间
    print("\n基础匹配时间：")
    with open(os.path.join(output_dir, "base_timings.txt"), "w") as f:
        f.write(f"LSH耗时: {lsh_time:.6f}秒\n")
        f.write(f"NN耗时: {bf_time:.6f}秒\n")
    print(f"LSH: {lsh_time:.6f}秒")
    print(f"NN: {bf_time:.6f}秒")

    # 分析投影集合规模影响
    evaluate_projection_impact(size_varied_projections, database_dir, query_image, query_image, output_dir)

    # 性能基准测试（按需启用）
    # test_iterations = 1000
    # avg_lsh, avg_bf = benchmark_performance(base_projections, test_iterations, database_dir, query_image, query_image, output_dir)
    # with open(os.path.join(output_dir, "avg_timings.txt"), "w") as f:
    #     f.write(f"{test_iterations}次平均 - LSH: {avg_lsh:.6f}秒\n")
    #     f.write(f"{test_iterations}次平均 - NN: {avg_bf:.6f}秒\n")

    print("所有任务完成，结果已保存至output文件夹")