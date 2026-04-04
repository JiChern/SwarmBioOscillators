import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

if __name__ == '__main__':
    # ← 只改这一行即可（路径保持和 Part 1 一致）
    dataset ='Minesweeper'
    system = 'graphcon'   #sies graphcon
    csv_file = f'{dataset}_{system}_pca_test_split0.csv'

    df = pd.read_csv(csv_file)
    print(f"✅ 已加载 CSV: {csv_file} | 总记录 {len(df):,} 行")

    # 自动解析参数
    dataset = csv_file.split('_')[0] if 'exp_data' not in csv_file else csv_file.split('/')[-1].split('_')[0]
    system = csv_file.split('_')[1] if 'exp_data' not in csv_file else csv_file.split('/')[-1].split('_')[1]
    split_idx = int(csv_file.split('split')[-1].split('.')[0])

    steps_to_show = df['timestep'].nunique()
    print(f"总时间步: {steps_to_show} | 测试节点数: {df['local_node_idx'].max()+1}")

    # ================== 【核心改进】动态高质量 colormap ==================
    n_classes = int(df['label'].max() + 1)
    print(f"🎨 检测到 {n_classes} 个类别 → 为每一种类别分配独特颜色")

    if n_classes <= 10:
        cmap_name = 'tab10'
    elif n_classes <= 20:
        cmap_name = 'tab20'          # ← roman-empire 刚好 18 类，最适合这个
    else:
        cmap_name = 'tab20'          # 超过 20 类时仍用 tab20（颜色最清晰）

    print(f"   使用 colormap: {cmap_name}（确保每类一个独特颜色）")

    # 按 timestep 重建 X_2d_list
    groups = [group.sort_values('local_node_idx') for _, group in df.groupby('timestep', sort=True)]
    X_2d_list = [g[['pca_x', 'pca_y']].values for g in groups]
    y_test = groups[0]['label'].values

    # ================== 创建动画（颜色已优化） ==================
    fig, ax = plt.subplots(figsize=(11, 9))

    sc = ax.scatter(X_2d_list[0][:, 0], X_2d_list[0][:, 1],
                    c=y_test, cmap=cmap_name, s=14, alpha=0.85,   # alpha 稍微调高一点更清晰
                    edgecolors='white', linewidth=0.4)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f'{dataset} - {system.upper()} - Test Split (split {split_idx})\n'
                         f'Node Feature Evolution (PCA) - Step 0 / {steps_to_show-1}',
                         fontsize=16, pad=20)

    plt.colorbar(sc, ax=ax, label='Node Label', ticks=range(n_classes))

    def update(frame):
        sc.set_offsets(X_2d_list[frame])

        # 每帧自动适应坐标轴
        X_current = X_2d_list[frame]
        x_min, x_max = X_current[:, 0].min(), X_current[:, 0].max()
        y_min, y_max = X_current[:, 1].min(), X_current[:, 1].max()
        x_padding = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        y_padding = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        title.set_text(f'{dataset} - {system.upper()} - Test Split (split {split_idx})\n'
                       f'Node Feature Evolution (PCA) - Step {frame} / {steps_to_show-1}')
        return sc,

    print(f"🎬 正在生成**颜色优化版**的慢速动图...")

    ani = animation.FuncAnimation(fig, update, frames=steps_to_show,
                                  interval=500, blit=False, repeat=True)

    # ================== 保存动画 ==================
    gif_name = f'{dataset}_{system}_pca_evolution_animation_TEST_split{split_idx}_fromCSV.gif'
    ani.save(gif_name, writer='pillow', fps=3, dpi=200)
    print(f"✅ 动画GIF保存完成: {gif_name}")
    # ani.save(gif_name.replace('.gif', '.mp4'), writer='ffmpeg', fps=3, dpi=220)  # 需要 ffmpeg 时取消注释

    plt.show()