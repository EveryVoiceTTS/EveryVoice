""" Setup for everyvoice
"""

from os import path

from setuptools import find_packages, setup

# Ugly hack to read the current version number without importing everyvoice:
with open("everyvoice/_version.py", "r", encoding="utf8") as version_file:
    namespace: dict = {}
    exec(version_file.read(), namespace)
    VERSION = namespace["VERSION"]
    # [N!]N(.N)*[{a|b|rc}N][.postN][.devN]

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf8") as f:
    long_description = f.read()

with open(path.join(this_directory, "requirements.txt"), encoding="utf8") as f:
    REQS = f.read().splitlines()

# Get a list of submodules
SUBMODULE_PATHS = [
    "everyvoice/model/feature_prediction/FastSpeech2_lightning/fs2",
    "everyvoice/model/vocoder/HiFiGAN_iSTFT_lightning/hfgl",
    "everyvoice/model/aligner/DeepForcedAligner/dfaligner",
    "everyvoice/model/aligner/wav2vec2aligner/aligner",
]
SUBMODULE_PACKAGES = []

# For each submodule
for submodule_path in SUBMODULE_PATHS:
    # append the submodule path to the list of submodule packages
    submodule = submodule_path.replace("/", ".")
    SUBMODULE_PACKAGES.append(submodule)
    # then use find_packages to automatically find the rest
    packages = find_packages(submodule_path)
    SUBMODULE_PACKAGES += [submodule + f".{pkg}" for pkg in packages]

setup(
    name="everyvoice",
    python_requires=">=3.10, <3.12",
    version=VERSION,
    author="Aidan Pine",
    author_email="hello@aidanpine.ca",
    license="MIT",
    url="https://github.com/roedoejet/EveryVoice",
    description="Text-to-Speech Synthesis for the Speech Generation for Indigenous Language Education Small Teams Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platform=["any"],
    packages=find_packages() + SUBMODULE_PACKAGES,
    include_package_data=True,
    install_requires=REQS,
    entry_points={"console_scripts": ["everyvoice = everyvoice.cli:app"]},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
)
