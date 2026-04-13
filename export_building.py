import json
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# 路径设置
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'website', 'building_data.json')
IMG_DIR   = os.path.join(BASE_DIR, 'IC_campus_streetview')
OUT_BASE  = os.path.join(BASE_DIR, 'seperate_building')

# 读取数据
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打印所有建筑名称
print("\n📋 所有建筑列表：")
print("-" * 50)
buildings = {v['name']: v for v in data.values() if v.get('has_name')}
for i, name in enumerate(buildings.keys(), 1):
    print(f"{i:2}. {name}")
print("-" * 50)

# ── 用户输入（支持多个建筑，用逗号分隔）──────────────
print("\n💡 提示：可以输入多个建筑名称，用逗号分隔")
print("   例如：Queen's Tower, Roderic Hill, ACE Extension")
user_input = input("\n请输入建筑名称（或部分名称）：").strip()

# 拆分多个关键词
keywords = [k.strip() for k in user_input.split(',') if k.strip()]

# ── 对每个关键词进行匹配 ─────────────────────────────
final_selected = {}  # {name: info}

for keyword in keywords:
    matched = {name: info for name, info in buildings.items()
               if keyword.lower() in name.lower()}

    if not matched:
        print(f"❌ 没有找到匹配「{keyword}」的建筑，跳过")
        continue

    if len(matched) > 1:
        print(f"\n「{keyword}」找到 {len(matched)} 个匹配的建筑：")
        names = list(matched.keys())
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
        try:
            choice = int(input("  请输入编号选择：")) - 1
            selected_name = names[choice]
        except:
            print(f"  ⚠️  输入无效，跳过「{keyword}」")
            continue
    else:
        selected_name = list(matched.keys())[0]

    final_selected[selected_name] = matched[selected_name]
    print(f"✅ 已加入：{selected_name}")

if not final_selected:
    print("\n❌ 没有选择任何建筑，退出")
    exit()

print(f"\n📦 共选择 {len(final_selected)} 个建筑，开始处理...")
print("=" * 50)

# ── 生成俯视轮廓图 ──────────────────────────────────
def generate_footprint(selected_name, selected, out_dir, images):
    print(f"\n🗺️  正在生成俯视轮廓图...")
    footprint = selected.get('footprint', [])
    if footprint:
        coords = np.array(footprint)
        xs, ys = coords[:, 0], coords[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        polygon = Polygon(coords, closed=True)
        patch = PatchCollection([polygon], facecolor='#4fc3f7',
                                 edgecolor='#81d4fa', linewidth=2, alpha=0.85)
        ax.add_collection(patch)
        ax.scatter(xs, ys, color='#81d4fa', s=20, zorder=5)
        margin = max((xs.max()-xs.min()), (ys.max()-ys.min())) * 0.2
        ax.set_xlim(xs.min()-margin, xs.max()+margin)
        ax.set_ylim(ys.min()-margin, ys.max()+margin)
        ax.set_title(selected_name, color='white', fontsize=14, pad=15)
        ax.set_xlabel('X (m)', color='#aaaaaa', fontsize=10)
        ax.set_ylabel('Y (m)', color='#aaaaaa', fontsize=10)
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        info_text = f"高度: {selected.get('height_m','N/A')} m\n图片数量: {len(images)} 张"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                color='#aaaaaa', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#2a2a4a', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'footprint.png'),
                    dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"   ✅ 轮廓图已保存：footprint.png")
    else:
        print(f"   ⚠️  没有轮廓数据，跳过")

# ── 处理每个选中的建筑 ──────────────────────────────
total_success, total_fail = 0, 0

for selected_name, selected in final_selected.items():
    print(f"\n🏢 正在处理：{selected_name}")
    print("-" * 40)

    # 创建输出文件夹
    safe_name = selected_name.replace('/', '-').replace('\\', '-')
    out_dir = os.path.join(OUT_BASE, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    # 复制图片
    images = selected.get('images', [])
    print(f"📸 共 {len(images)} 张图片，开始复制...")
    success, fail = 0, 0
    for img in images:
        src = os.path.join(IMG_DIR, img['filename'])
        dst = os.path.join(out_dir, img['filename'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            success += 1
        else:
            print(f"  ⚠️  找不到图片：{img['filename']}")
            fail += 1

    total_success += success
    total_fail += fail
    print(f"   ✅ 成功复制：{success} 张 | ❌ 失败：{fail} 张")

    # 保存建筑信息
    info_path = os.path.join(out_dir, 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 建筑信息已保存：info.json")

    # 生成轮廓图
    generate_footprint(selected_name, selected, out_dir, images)

    print(f"   📁 输出位置：{out_dir}")

# ── 完成 ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"🎉 全部完成！")
print(f"   🏢 处理建筑数：{len(final_selected)} 个")
print(f"   ✅ 成功复制图片：{total_success} 张")
if total_fail:
    print(f"   ❌ 失败：{total_fail} 张")
print(f"   📁 输出根目录：{OUT_BASE}")