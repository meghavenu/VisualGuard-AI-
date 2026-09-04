
import sys
import asyncio
from playwright.sync_api import sync_playwright

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

url = sys.argv[1]
output_path = sys.argv[2]
login_url = sys.argv[3] if len(sys.argv) > 3 else ""
username = sys.argv[4] if len(sys.argv) > 4 else ""
password = sys.argv[5] if len(sys.argv) > 5 else ""
username_selector = sys.argv[6] if len(sys.argv) > 6 else ""
password_selector = sys.argv[7] if len(sys.argv) > 7 else ""
submit_selector = sys.argv[8] if len(sys.argv) > 8 else ""

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    if login_url and username and password and username_selector and password_selector:
        page.goto(login_url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(2000)
        page.fill(username_selector, username)
        page.fill(password_selector, password)

        if submit_selector:
            page.click(submit_selector)
        else:
            page.keyboard.press("Enter")

        page.wait_for_timeout(4000)

    page.goto(url, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(4000)
    page.screenshot(path=output_path, full_page=True)
    browser.close()
