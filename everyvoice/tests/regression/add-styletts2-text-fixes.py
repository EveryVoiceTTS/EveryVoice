#!/usr/bin/env python
"""Patch a wizard-generated regression project's configs so StyleTTS2 training
works with IPA symbols the pretrained StyleTTS2 symbol table doesn't cover.

The wizard has no step for to_replace rules or target_text_representation_level,
so this script edits the generated yaml configs directly. Run from inside the
project root (after `new-project`, before `preprocess`).
"""

import yaml

TO_REPLACE = {
    "t͡ʃ": "ʧ",
    "ɜ˞": "ɞ",
}

TEXT_CONFIG_PATH = "config/everyvoice-shared-text.yaml"
with open(TEXT_CONFIG_PATH, encoding="utf8") as f:
    text_config = yaml.safe_load(f)
text_config.setdefault("to_replace", {}).update(TO_REPLACE)
with open(TEXT_CONFIG_PATH, "w", encoding="utf8") as f:
    yaml.dump(text_config, f, default_flow_style=None, allow_unicode=True)

# FastSpeech2 (text-to-spec) is intentionally left targeting "characters".
E2E_CONFIG_PATH = "config/everyvoice-text-to-wav.yaml"
with open(E2E_CONFIG_PATH, encoding="utf8") as f:
    e2e_config = yaml.safe_load(f)
e2e_config.setdefault("model", {})["target_text_representation_level"] = "phones"
with open(E2E_CONFIG_PATH, "w", encoding="utf8") as f:
    yaml.dump(e2e_config, f, default_flow_style=None, allow_unicode=True)
