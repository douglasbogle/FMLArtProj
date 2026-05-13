import asyncio
import json
import random
import re
import ssl
from pathlib import Path
import aiohttp
import aiofiles
from playwright.async_api import async_playwright

# --- CONFIG ---
OUTPUT_DIR = Path("phillips_data")
IMAGE_DIR = OUTPUT_DIR / "contemporary_images"
CHECKPOINT_FILE = OUTPUT_DIR / "contemporary_lots.jsonl"
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

MAX_LOTS = 20000
CONCURRENT_DOWNLOADS = 10

# Extra checks to ensure we dont get other auction items
BAD_KEYWORDS = ['watch', 'jewelry', 'handbag', 'wine', 'car', 'sportswear', 'design', 'photographs']

SKIP_LOT_TITLE_KEYWORDS = [
    'vase', 'bowl', 'lamp', 'armchair', 'sofa', 'cabinet', 'table', 'desk', 
    'bench', 'mirror', 'chest', 'wardrobe', 'sideboard', 'rug', 'carpet', 'box'
]

async def get_art_sale_urls(page) -> list[dict]:
    print("Finding all Contemporary past auctions with dates...")
    url = "https://www.phillips.com/auctions/past/filter/Departments%3DContemporary/sort/newest"
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(3)

    sales = await page.evaluate("""
        () => {
            // Target the list items containing the auctions
            return [...document.querySelectorAll('li.auction')]
                .map(li => {
                    const link = li.querySelector('h2 a');
                    const dateText = li.querySelector('.content-body p')?.innerText || "";
                    
                    return {
                        url: link?.href,
                        title: link?.innerText.toLowerCase(),
                        auction_date_raw: dateText // e.g., "Hong Kong Auction 29 March 2026"
                    };
                })
                .filter(item => item.url && /\\/auction\\/[A-Z]{2}\\d+/.test(item.url));
        }
    """)
    
    unique_sales = {s['url']: s for s in sales}.values()
    filtered = [s for s in unique_sales if not any(kw in s['title'] for kw in BAD_KEYWORDS)]
    print(f"Found {len(filtered)} Fine Art auctions.")
    return filtered

async def scrape_sale_lots(page, sale_url: str) -> list[dict]:
    try:
        await page.goto(sale_url, wait_until="domcontentloaded", timeout=45000)
        for _ in range(15): 
            await page.evaluate("window.scrollBy(0, 2000)")
            await asyncio.sleep(0.5)
    except Exception:
        return []

    return await page.evaluate("""
        () => {
            return [...document.querySelectorAll('a.seldon-object-tile')].map(tile => {
                const img = tile.querySelector('img.seldon-seldon-image-img');
                const srcset = img?.srcset || '';
                const hiRes = srcset ? srcset.split(',').pop().trim().split(' ')[0] : img?.src;

                return {
                    lot_url: tile.href,
                    artist: tile.querySelector('.seldon-object-tile__maker')?.innerText?.trim(),
                    title: tile.querySelector('.seldon-object-tile__title')?.innerText?.trim(),
                    estimate: tile.querySelector('.seldon-object-tile__estimate .seldon-detail__value')?.innerText?.trim(),
                    sold_for: tile.querySelector('.seldon-bid-snapshot .seldon-detail__value')?.innerText?.trim(),
                    image_url: hiRes,
                    sale_url: window.location.href
                };
            });
        }
    """)

async def download_image(session, lot, sem):
    url = lot.get("image_url")
    if not url or not url.startswith('http'): return
    
    lot_id = lot["lot_url"].rstrip('/').split('/')[-1]
    img_path = IMAGE_DIR / f"{lot_id}.jpg"

    if img_path.exists():
        lot["local_image"] = str(img_path)
        return

    async with sem:
        try:
            async with session.get(url, timeout=20) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    async with aiofiles.open(img_path, "wb") as f:
                        await f.write(content)
                    lot["local_image"] = str(img_path)
        except:
            pass

async def main():
    all_lots = []
    seen_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = await context.new_page()
        
        sales = await get_art_sale_urls(page)
        random.shuffle(sales)

        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ckpt:
            for sale in sales:
                if len(all_lots) >= MAX_LOTS: break
                print(f"Scraping Sale: {sale['title']} ({sale['auction_date_raw']})")
                lots = await scrape_sale_lots(page, sale['url'])
                
                for lot in lots:
                    if len(all_lots) >= MAX_LOTS: break
                    
                    # Attach the auction date to the lot
                    lot['auction_date'] = sale['auction_date_raw']
                    
                    title = (lot.get('title') or "").lower()
                    is_blacklisted = any(kw in title for kw in SKIP_LOT_TITLE_KEYWORDS)
                    has_edition_ref = bool(re.search(r'\d{1,3}/\d{1,3}', title))
                    
                    if is_blacklisted or has_edition_ref:
                        continue

                    if lot["lot_url"] not in seen_urls and lot.get('sold_for'):
                        all_lots.append(lot)
                        seen_urls.add(lot["lot_url"])
                        ckpt.write(json.dumps(lot, ensure_ascii=False) + "\n")
                        ckpt.flush()
                
                await asyncio.sleep(random.uniform(1.0, 2.0))
        await browser.close()

    print(f"Collected {len(all_lots)} items. Starting downloads...")
    
    connector = aiohttp.TCPConnector(ssl=False)
    sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, lot, sem) for lot in all_lots]
        await asyncio.gather(*tasks)

    print("Data collection complete.")

if __name__ == "__main__":
    asyncio.run(main())