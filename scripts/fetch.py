import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.config import DATA_DIR
SAVE_DIR = os.path.join(DATA_DIR, 'manual_images')
CHECK_INTERVAL = 2  # 检查间隔（秒）
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(SAVE_DIR, exist_ok=True)

# 启动浏览器（非无头模式）
chrome_opts = Options()
chrome_opts.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=chrome_opts)

print("✅ 启动完成，请手动打开目标页面（例如 Blizzard 图集）")
input("👉 打开页面后按 Enter 开始监控...")
# https://hearthstone.fandom.com/wiki/Full_art
last_src = None
saved = set()

while True:
    try:
        # 执行 JavaScript 直接获取当前展示图片的 URL
        img_src = driver.execute_script("""
            let imgs = Array.from(document.querySelectorAll('img'));
            // 筛掉小图标类元素
            let bigs = imgs.filter(i => i.width > 300 && i.height > 300);
            if (bigs.length === 0) return null;
            // 取最大的一张
            let main = bigs.sort((a,b)=> (b.width*b.height)-(a.width*a.height))[0];
            return main?.src || null;
        """)

        if img_src and img_src != last_src and img_src not in saved:
            print(f"[NEW IMAGE] {img_src}")
            try:
                r = requests.get(img_src, headers=HEADERS, timeout=10)
                if r.status_code == 200:
                    ext = ".jpg" if ".jpg" in img_src else ".png"
                    fname = f"img_{len(saved)+1}{ext}"
                    path = os.path.join(SAVE_DIR, fname)
                    with open(path, "wb") as f:
                        f.write(r.content)
                    saved.add(img_src)
                    last_src = img_src
                    print(f"✅ Saved as {fname}")
            except Exception as e:
                print(f"❌ Download failed: {e}")

        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n🧩 手动退出，共保存 {len(saved)} 张图片。")
        break
    except Exception as e:
        print(f"[WARN] {e}")
        time.sleep(CHECK_INTERVAL)
