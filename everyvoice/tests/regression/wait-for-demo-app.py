from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    for _ in range(60):
        try:
            page.goto("http://127.0.0.1:7860/", timeout=5000)
            print("Page loaded")
            break
        except Exception as e:
            print("Error loading page:", e)
            page.wait_for_timeout(1000)
