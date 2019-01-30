import os
import re

import setuptools


class CleanCommand(setuptools.Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


directory = os.path.dirname(os.path.abspath(__file__))

# Extract version information
path = os.path.join(directory, "metal", "__init__.py")
with open(path) as read_file:
    text = read_file.read()
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# Extract long_description
path = os.path.join(directory, "README.md")
with open(path) as read_file:
    long_description = read_file.read()

setuptools.setup(
    name="snorkel-metal",
    version=version,
    url="https://github.com/HazyResearch/metal",
    description="A system for quickly generating training data with multi-task weak supervision",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "networkx>=2.2",
        "numpy",
        "pandas",
        "torch>=1.0",
        "scipy",
        "tqdm",
        "scikit-learn",
    ],
    include_package_data=True,
    keywords="machine-learning ai information-extraction weak-supervision mtl multitask multi-task-learning",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    project_urls={  # Optional
        "Homepage": "https://hazyresearch.github.io/snorkel/",
        "Source": "https://github.com/HazyResearch/metal/",
        "Bug Reports": "https://github.com/HazyResearch/metal/issues",
        "Citation": "https://arxiv.org/abs/1810.02840",
    },
    cmdclass={"clean": CleanCommand},
)
