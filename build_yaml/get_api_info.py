import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
import pickle
from pathlib import Path
import re
import time
import os
import sys

# Utils_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if Utils_DIR not in sys.path:
#     sys.path.insert(0, Utils_DIR)
from utils import *

# ===== 配置部分 =====
SCRIPT_DIR = Path(__file__).resolve().parent

API_PICKLE_PATH = SCRIPT_DIR / "api_union_4o.pkl"  # 你的 api 列表 pkl
SPARSE_URL = "https://pytorch.org/docs/stable/sparse.html"
BASE_URL = "https://pytorch.org/docs/stable/"      # 用于拼 generated/... 链接
OUTPUT_DIR = SCRIPT_DIR / "sparse_api_info"        # 保存文档的目录
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CuFuzzDocBot/1.0; +https://example.com)"
}

# ===== 小工具函数 =====

def load_api_list(path: Path):
    with open(path, "rb") as f:
        apis = pickle.load(f)
    # 确保是 list[str]
    apis = [str(a).strip() for a in apis if str(a).strip()]
    return apis

def normalize_raw_name(name: str) -> str:
    """
    去掉括号等干扰：比如 "torch.sparse_coo_tensor()" -> "torch.sparse_coo_tensor"
    """
    name = name.strip()
    name = re.sub(r"\(.*\)$", "", name)  # 去掉尾部括号和参数
    return name

def normalize_display_name(api_name: str) -> str:
    """
    把 'torch.sparse.Tensor.to_sparse_bsr' 归一成文档中常见形式，例如 'Tensor.to_sparse_bsr'
    """
    name = normalize_raw_name(api_name)

    if name.startswith("torch.sparse.Tensor."):
        return "Tensor." + name.split("torch.sparse.Tensor.", 1)[1]
    if name.startswith("torch.Tensor."):
        return "Tensor." + name.split("torch.Tensor.", 1)[1]

    # 其他情况（如 torch.sparse_csr_tensor）保持原样
    return name

def fetch_html(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def find_api_links_on_sparse_page(soup: BeautifulSoup):
    """
    在 sparse.html 页面中构建一个 { visible_text: href } 的映射，
    visible_text 用的是 <a> 的可见文本。
    """
    api_to_href = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if not text:
            continue
        href = a["href"]
        api_to_href[text] = href
    return api_to_href

def match_api_to_link(api_name: str, link_map: dict):
    """
    在 link_map 中根据 api_name 找到最合适的链接。

    策略：
      1. 归一化为文档里的显示形式（如 'Tensor.to_sparse_bsr'）
      2. 先用函数名短名去匹配 href 中的 generated/...to_sparse_bsr...
      3. 再用可见文本精确匹配 / endswith 匹配
      4. 实在找不到，按 PyTorch generated 规则猜一个 URL
    """
    display_name = normalize_display_name(api_name)  # 例如 'Tensor.to_sparse_bsr'
    short = display_name.split(".")[-1]              # 'to_sparse_bsr'

    # 1) 优先：href 中包含短名（如 generated/torch.Tensor.to_sparse_bsr.html#...）
    candidates = []
    for text, href in link_map.items():
        if short in href:
            candidates.append(href)

    # 优先选择 generated 页面
    for href in candidates:
        if "generated" in href:
            return urljoin(BASE_URL, href)
    if candidates:
        return urljoin(BASE_URL, candidates[0])

    # 2) 其次：可见文本精确匹配 display_name 或 short
    for text, href in link_map.items():
        t = text.strip()
        if t == display_name or t == short:
            return urljoin(BASE_URL, href)

    # 3) 再退一步：文本以短名结尾（比如 'Tensor.to_sparse_bsr'）
    for text, href in link_map.items():
        t = text.strip()
        if t.endswith("." + short):
            return urljoin(BASE_URL, href)

    # 4) 实在找不到：按 generated 规则构造
    raw = normalize_raw_name(api_name)
    # 对于 Tensor 方法，构造 torch.Tensor.short
    if raw.startswith("torch.sparse.Tensor.") or raw.startswith("torch.Tensor."):
        dotted = f"torch.Tensor.{short}"
    else:
        dotted = raw  # 比如 torch.sparse_csr_tensor 这类函数形式

    guessed = f"generated/{dotted}.html"
    return urljoin(BASE_URL, guessed)

def extract_main_text_from_doc(soup: BeautifulSoup) -> str:
    """
    尝试从 pytorch 文档页面中提取正文区域。
    优先找 role="main" 的 div，其次退回整个 body 文本。
    """
    main = soup.find("div", attrs={"role": "main"})
    if main is None:
        main = soup.find("div", class_="document")
    if main is None:
        main = soup.body or soup

    # 去掉一些导航/侧边栏之类的噪声（简单做一下）
    for nav in main.find_all(["nav", "header", "footer"]):
        nav.decompose()

    text = main.get_text("\n", strip=True)
    return text

def normalize_api_name(api_name: str) -> str:
    """
    把 'torch.sparse.Tensor.to_sparse_bsr' 之类的名字
    归一化成页面中的文本形式，比如 'Tensor.to_sparse_bsr'
    """
    name = api_name.strip()

    # 常见几种前缀处理
    if name.startswith("torch.sparse.Tensor."):
        return "Tensor." + name.split("torch.sparse.Tensor.", 1)[1]
    if name.startswith("torch.Tensor."):
        return "Tensor." + name.split("torch.Tensor.", 1)[1]

    # 普通函数（如 torch.sparse_csr_tensor）直接返回
    return name

def extract_api_section(api_name: str, soup: BeautifulSoup) -> str:
    """
    尝试只提取当前 API 对应的那一小块文档，而不是整页。
    """
    # 先规范一下 API 名，再映射到 Sphinx 用的 id
    norm = normalize_api_name(api_name)

    # 对于方法：torch.sparse.Tensor.to_sparse_bsr -> 文档 id 是 torch.Tensor.to_sparse_bsr
    html_id = norm
    if ".Tensor." in norm:
        html_id = "torch.Tensor." + norm.split(".Tensor.", 1)[-1]

    # 对于函数：torch.sparse_bsr_tensor -> 文档 id 一般就是这个全名
    dt = soup.find("dt", id=html_id)
    if not dt:
        # 找不到就退回到原来的“大块 main 提取”
        return extract_main_text_from_doc_fallback(soup)

    dd = dt.find_next_sibling("dd")
    if not dd:
        return extract_main_text_from_doc_fallback(soup)

    pieces = []

    # 标题（带签名）
    title = dt.get_text(" ", strip=True)
    pieces.append(title)

    # 依次遍历 dd 里的结构化内容
    for child in dd.children:
        if isinstance(child, NavigableString):
            continue
        if not getattr(child, "name", None):
            continue

        # 普通段落
        if child.name == "p":
            txt = child.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)

        # 参数列表（Sphinx 的 field-list）
        elif child.name == "dl" and "field-list" in (child.get("class") or []):
            pieces.append("Parameters:")
            for item_dt in child.find_all("dt", recursive=False):
                name = item_dt.get_text(" ", strip=True)
                item_dd = item_dt.find_next_sibling("dd")
                desc = item_dd.get_text(" ", strip=True) if item_dd else ""
                pieces.append(f"- {name}: {desc}")

        # 代码块
        elif child.name == "div" and "highlight" in (child.get("class") or []):
            code = child.get_text("\n", strip=True)
            if code:
                pieces.append("```python\n" + code + "\n```")

        # 其它（比如小标题等）可以按需再加
        else:
            txt = child.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)

    return "\n\n".join(pieces)


def extract_main_text_from_doc_fallback(soup: BeautifulSoup) -> str:
    """
    找不到对应 dt/dd 时的兜底：退回到 role=main，但仍尽量去掉 nav/header/footer。
    """
    main = soup.find("div", attrs={"role": "main"}) or soup.find("div", class_="document") or soup.body or soup

    for nav in main.find_all(["nav", "header", "footer"]):
        nav.decompose()

    return main.get_text("\n", strip=True)


def save_api_doc(api_name: str, content: str):
    # 文件名中去掉不能用的字符
    safe_name = re.sub(r"[^\w\.-]+", "_", normalize_raw_name(api_name))
    out_path = OUTPUT_DIR / f"{safe_name}.txt"
    out_path.write_text(content, encoding="utf-8")
    print(f"[+] Saved doc for {api_name} -> {out_path}")

# ===== 主流程 =====

def main():
    # 1. 加载 API 列表
    api_list = load_api_list(API_PICKLE_PATH)
    print(f"[+] Loaded {len(api_list)} APIs from {API_PICKLE_PATH}")

    # 2. 抓 sparse 主页面并解析所有可点击链接
    print(f"[+] Fetching sparse page: {SPARSE_URL}")
    sparse_soup = fetch_html(SPARSE_URL)
    link_map = find_api_links_on_sparse_page(sparse_soup)
    print(f"[+] Found {len(link_map)} clickable entries on sparse page")

    # 3. 遍历每个 API，看看有没有对应链接
    #    调试时先跑前几个：api_list[:5]，正式跑就用 api_list
    for api in api_list[100:]:
        href = match_api_to_link(api, link_map)
        if not href:
            print(f"[-] No link found for API: {api}")
            continue

        print(f"\n [+] Fetching detail for {api}: {href}")

        try:
            doc_soup = fetch_html(href)
            # text = extract_api_section(api, doc_soup)
            # save_api_doc(api, text)
            time.sleep(0.5)
        except Exception as e:
            print(f"[!] Error fetching {api} from {href}: {e}")

def check_api():
    api_list = load_api_list(API_PICKLE_PATH)
    print(f"[+] Loaded {len(api_list)} APIs from {API_PICKLE_PATH}")

    file_list = os.listdir(OUTPUT_DIR)

    for file in file_list:
        api_name = file.replace(".txt", "")
        if api_name in api_list:
            continue
        else:
            print(f"[-] No doc found for API: {api_name}")
            print(top_similar(api_name, api_list))

def check_api_2():
    api_list = load_api_list(API_PICKLE_PATH)
    print(f"[+] Loaded {len(api_list)} APIs from {API_PICKLE_PATH}")

    file_list = os.listdir(OUTPUT_DIR)
    for file in api_list:
        if file + ".txt" in file_list:
            continue
        else:
            print(f"[-] No doc found for API: {file}")
            print(top_similar(file, file_list))


def update_pkl(delete_api, updata_api):
    api_list = load_api_list(API_PICKLE_PATH)
    print(f"[+] Loaded {len(api_list)} APIs from {API_PICKLE_PATH}")

    for api in delete_api:
        if api in api_list:
            api_list.remove(api)

    for api in updata_api:
        if api not in api_list:
            api_list.append(api)

    with open(API_PICKLE_PATH, "wb") as f:
        pickle.dump(api_list, f)
    print(f"[+] Updated API list saved to {API_PICKLE_PATH}, total {len(api_list)} APIs.")



if __name__ == "__main__":
    # main()
    check_api()

    delete_api = ['Tensor.is_sparse_csr', 'Tensor.dense_dim', 'torch.sparse.Tensor.dense_dim', 'torch.sspaddmm', 'Tensor.sparse_resize_',
                  'Tensor.is_sparse', 'Tensor.sparse_resize_and_clear_', 'Tensor.sparse_dim', 'Tensor.is_coalesced']

    updata_api = ['torch.Tensor.is_sparse_csr', 'torch.Tensor.smm', 'torch.Tensor.dense_dim', 'torch.Tensor.sspaddmm', 'torch.Tensor.sparse_resize_',
                  'torch.Tensor.is_sparse', 'torch.Tensor.sparse_resize_and_clear_', 'torch.Tensor.sparse_dim', 'torch.Tensor.is_coalesced']
    
    # update_pkl(delete_api, updata_api)

    #check_api_2()
    delete_api_2 = ['torch.sparse.Tensor.to_sparse_bsr', 'sparse.log_softmax', 'sparse.log_softmax', 'torch.sparse.Tensor.to_sparse_csc',
                    'Tensor.to_sparse_csr','sparse.addmm','Tensor.col_indices','torch.sparse.Tensor.to_sparse_bsc', 'torch.sparse.Tensor.col_indices',
                    'sparse.sampled_addmm', 'torch.sparse.Tensor.indices','Tensor.to_sparse_csc','torch.sparse.Tensor.coalesce','torch.sparse.Tensor.row_indices',
                    'torch.sparse.Tensor.sparse_resize_','torch.sparse.Tensor.to_sparse_csr','torch.sparse.Tensor.is_coalesced',
                    'torch.sparse.Tensor.sparse_mask','Tensor.sparse_mask','Tensor.ccol_indices','sparse.sum','torch.to_sparse_csr','torch.to_sparse_bsc',
                    'sparse.as_sparse_gradcheck','Tensor.to_sparse_bsc','smm','Tensor.to_sparse_bsr','sparse.spdiags','Tensor.to_dense',
                    'torch.to_sparse','sparse.mm','torch.smm','torch.sparse.Tensor.to_dense','torch.sparse.Tensor.is_sparse','Tensor.to_sparse_coo',
                    'torch.tensor.to_sparse','sparse.spsolve','torch.sparse.Tensor.sparse_dim','torch.tensor.to_sparse_csc','Tensor.to_sparse',
                    'torch.tensor.to_sparse_bsr','torch.sparse.Tensor.to_sparse','torch.to_sparse_csc','torch.sparse.Tensor.crow_indices',
                    'torch.sparse.Tensor.ccol_indices', 'sspaddmm', 'torch.sparse.Tensor.values','torch.sparse.Tensor.sparse_resize_and_clear_',
                    'torch.to_sparse_semi_structured','torch.sparse.Tensor.to_sparse_coo','sparse.check_sparse_tensor_invariants','Tensor.row_indices',
                    'Tensor.values','sparse.softmax','Tensor.crow_indices','torch.sparse',' torch.tensor.to_sparse_bsc', 'torch.sparse.Tensor.is_sparse_csr',
                    'hspmm', 'Tensor.coalesce', 'Tensor.indices','torch.to_sparse_coo']
    
    # print(len(delete_api_2))
    # update_pkl(['torch.tensor.to_sparse_bsc'], [])