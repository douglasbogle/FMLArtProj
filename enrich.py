import asyncio
import json
import random
import re
from pathlib import Path
from playwright.async_api import async_playwright

# --- CONFIG ---
OUTPUT_DIR = Path("phillips_data")
CHECKPOINT_FILE = OUTPUT_DIR / "contemporary_lots.jsonl"
ENRICHED_FILE = OUTPUT_DIR / "contemporary_lots_enriched.jsonl"
IMAGE_SUBDIR = "contemporary_images"

def clean_val(val):
    if not val: return None
    nums = re.sub(r'[^0-9]', '', val)
    return int(nums) if nums else None

async def scrape_lot_detail(page, lot_url: str) -> dict:
    try:
        await page.goto(lot_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector('[data-testid="text-lot-cataloging-section"]', timeout=10000)
        
        return await page.evaluate("""
            () => {
                const results = { medium: null, dimensions: null, year_created: null };
                const section = document.querySelector('[data-testid="text-lot-cataloging-section"]');
                if (!section) return results;

                const lines = Array.from(section.querySelectorAll('.pah-html-parser'))
                                   .map(d => d.innerText.trim())
                                   .filter(t => t.length > 0);
                
                // 1. DIMENSIONS & MEDIUM
                const dimIndex = lines.findIndex(t => t.includes('in.') || t.includes('cm'));
                if (dimIndex !== -1) {
                    results.dimensions = lines[dimIndex];
                    if (dimIndex > 0) results.medium = lines[dimIndex - 1];
                }

                // 2. SPECIFIC "PAINTED IN" SEARCH
                // We look for the line specifically mentioning when it was painted/executed
                const createdLine = lines.find(t => t.includes('Painted in') || t.includes('Executed in'));
                if (createdLine) {
                    const yearMatch = createdLine.match(/\\b(18|19|20)\\d{2}\\b/);
                    if (yearMatch) results.year_created = yearMatch[0];
                }

                // 3. FALLBACK YEAR SEARCH
                // If specific line wasn't found, look through everything for a likely year
                if (!results.year_created) {
                    const allText = lines.join(' ');
                    const fallbackMatch = allText.match(/\\b(18|19|20)\\d{2}\\b/);
                    if (fallbackMatch) results.year_created = fallbackMatch[0];
                }

                return results;
            }
        """)
    except Exception:
        return {"medium": None, "dimensions": None, "year_created": None}

async def main():
    if not CHECKPOINT_FILE.exists():
        print(f"Error: {CHECKPOINT_FILE} not found.")
        return
    
    # 1. Track progress to allow resume
    done_urls = set()
    if ENRICHED_FILE.exists():
        with open(ENRICHED_FILE, "r", encoding="utf-8") as f:
            for line in f: 
                try:
                    data = json.loads(line)
                    done_urls.add(data['lot_url'])
                except: pass

    # 2. Filter out already processed AND null titles
    all_lots = []
    dropped_null_titles = 0
    
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                lot = json.loads(line)
                # STRICT CHECK: Must have a title and not be already enriched
                if lot.get('title') and lot['lot_url'] not in done_urls:
                    all_lots.append(lot)
                elif not lot.get('title'):
                    dropped_null_titles += 1
            except: continue

    print(f"Skipping {dropped_null_titles} lots with null titles.")
    print(f"Enriching {len(all_lots)} remaining lots...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1")
        page = await context.new_page()

        with open(ENRICHED_FILE, "a", encoding="utf-8") as out:
            for i, lot in enumerate(all_lots):
                details = await scrape_lot_detail(page, lot['lot_url'])
                lot.update(details)
                
                # Enrichment Post-Processing
                medium = (lot.get('medium') or "").lower()
                lot['is_oil'] = 'oil' in medium and 'paper' not in medium
                lot['is_acrylic'] = 'acrylic' in medium
                lot['sold_price_int'] = clean_val(lot.get('sold_for'))
                
                # Image Mapping
                lot_id = lot['lot_url'].rstrip('/').split('/')[-1]
                lot['local_image_path'] = f"{IMAGE_SUBDIR}/{lot_id}.jpg"

                # Safety check for logging
                safe_title = str(lot.get('title', 'Untitled'))[:30]
                print(f"[{i+1}/{len(all_lots)}] ENRICHED: {lot.get('artist', 'Unknown')} - {safe_title}...")
                
                out.write(json.dumps(lot, ensure_ascii=False) + "\n")
                
                # Periodic flush to disk
                if i % 10 == 0:
                    out.flush()

                # Anti-throttle delay
                await asyncio.sleep(random.uniform(0.6, 1.2))
        
        await browser.close()
    print(f"Finished! All data saved to {ENRICHED_FILE}")

if __name__ == "__main__":
    asyncio.run(main())