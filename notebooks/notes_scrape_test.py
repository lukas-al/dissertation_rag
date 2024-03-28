from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
import os

# Works but the running of the chrome executable is blocked. There is something blocking it so not going to bother with this battle...
def run(playwright):
    print('running')

    browser = playwright.chromium.launch(
        # executable_path=r"C:\Users\335257\AppData\Local\ms-playwright\chromium-1055\chrome-win\chrome.exe"
    )
    context = browser.new_context()

    page = context.new_page()
    page.goto(r'http://intranet/sites/MANotes/Documents/NotesSearch.aspx?sb=2;a&For=34&')

    content = page.content()
    print(content)

    browser.close()

if __name__ == '__main__':

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.environ["CONDA_PREFIX_1"]+"\ms-playwright"

    with sync_playwright() as playwright:
        run(playwright)
