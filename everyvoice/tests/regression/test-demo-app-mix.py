"""
Test the Demo App with the regress-mix demo model.
To run this test:
 - Run the regression test suite to get the regress-mix model.
 - cd regression/regress-<date>-<suffix>/regress-mix/regress
 - everyvoice demo logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt
 - python test_demo_app_mix.py
"""

from unittest import TestCase, main

from playwright.sync_api import sync_playwright


class TestDemoAppMix(TestCase):
    def test_rundemo(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto("http://127.0.0.1:7860/")
            input_textbox = page.get_by_label("Input Text")
            input_textbox.click()
            input_textbox.fill("This is a test.")
            synthesize_button = page.get_by_role("button", name="Synthesize")
            synthesize_button.click()
            page.get_by_label("Play", exact=True).click()
            with page.expect_download() as download_info:
                page.get_by_label("Download").click()
            download = download_info.value
            self.assertTrue(download.suggested_filename.endswith(".wav"))
            page.get_by_label("Output Format").click()
            page.get_by_label("spec").click()
            synthesize_button.click()
            page.get_by_label("Play", exact=True).click()
            with page.expect_download() as download1_info:
                page.locator("#file_output").get_by_role("link").click()
            download = download1_info.value
            self.assertTrue(download.suggested_filename.endswith(".pt"))
            page.get_by_label("Output Format").click()
            page.get_by_label("textgrid").click()
            synthesize_button.click()
            with page.expect_download() as download2_info:
                page.locator("#file_output").get_by_role("link").click()
            download = download2_info.value
            self.assertTrue(download.suggested_filename.endswith(".TextGrid"))
            page.get_by_label("Output Format").click()
            page.get_by_label("readalong-xml").click()
            synthesize_button.click()
            with page.expect_download() as download3_info:
                page.locator("#file_output").get_by_role("link").click()
            download = download3_info.value
            self.assertTrue(download.suggested_filename.endswith(".readalong"))
            page.get_by_label("Output Format").click()
            page.get_by_label("readalong-html").click()
            synthesize_button.click()
            with page.expect_download() as download4_info:
                page.locator("#file_output").get_by_role("link").click()
            download = download4_info.value
            self.assertTrue(download.suggested_filename.endswith(".html"))
            page.get_by_label("Language").click()
            page.get_by_label("und").click()
            page.get_by_label("Speaker", exact=True).click()
            page.get_by_label("isixhosa_speaker").click()
            page.get_by_label("Output Format").click()
            page.get_by_label("wav").click()
            input_textbox.click()
            input_textbox.fill("enye intlanzi imke negwegwe lam.")
            synthesize_button.click()
            page.get_by_label("Play", exact=True).click()
            page.get_by_label("Speaker", exact=True).click()
            page.get_by_label("sinhalese_speaker").click()
            input_textbox.click()
            input_textbox.fill("අටවිසියක් සෙනෙවියෝ හාත්පස රැකවල්ලා")
            synthesize_button.click()
            page.get_by_label("Play", exact=True).click()
            input_textbox.click()
            input_textbox.fill("This is a test.")
            synthesize_button.click()
            page.get_by_label("Play", exact=True).click()

            # ---------------------
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
