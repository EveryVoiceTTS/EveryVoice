"""
Test the Demo App with the regress-lj-full demo model with a denylist.
To run this test:
 - Run the regression test suite to get the regress-lj-full model.
 - cd regression/regress-<date>-<suffix>/regress-lj-full/regress
 - everyvoice demo logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt --denylist <(echo test) --output-format wav
 - python test_demo_app_mix.py
"""

from unittest import TestCase, main

from playwright.sync_api import expect, sync_playwright


class TestDemoAppErros(TestCase):
    def test_rundemo(self) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto("http://127.0.0.1:7860/")
            synthesize_button = page.get_by_role("button", name="Synthesize")
            synthesize_button.click()
            expect(
                page.get_by_text("Text for synthesis was not provided.")
            ).to_be_visible()
            page.get_by_test_id("toast-close").click()
            expect(page.get_by_label("Play", exact=True)).to_have_count(0)

            input_textbox = page.get_by_label("Input Text")
            input_textbox.click()
            input_textbox.fill("test denylist")
            synthesize_button.click()
            expect(
                page.get_by_text(
                    "Oops, the word test denylist is not allowed to be synthesized by this model."
                )
            ).to_be_visible()
            page.get_by_test_id("toast-close").click()
            expect(page.get_by_label("Play", exact=True)).to_have_count(0)

            input_textbox.click()
            input_textbox.fill("This is allowed.")
            synthesize_button.click()
            expect(page.get_by_label("Play", exact=True)).to_be_visible()

            # ---------------------
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
