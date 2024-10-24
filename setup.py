from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ergodic",
    version="0.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    author="ergodic.ai",
    author_email="andre@ergodic.ai",
    description="Python client for ergodic.ai",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ergodic-ai/ergodic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
