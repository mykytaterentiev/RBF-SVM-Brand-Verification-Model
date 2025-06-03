"""Setup configuration for RBF-SVM Brand Verification package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rbf-svm-brand-verification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RBF-SVM for brand verification with long-tail sensitivity",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/your-username/rbf-svm-vertex",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rbf-svm-train=rbf_svm.scripts.train_model:main",
            "rbf-svm-predict=rbf_svm.scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rbf_svm": ["*.yaml", "*.json"],
    },
    keywords="machine-learning svm brand-verification imbalanced-data long-tail",
    project_urls={
        "Bug Reports": "https://huggingface.co/your-username/rbf-svm-vertex/discussions",
        "Source": "https://huggingface.co/your-username/rbf-svm-vertex",
        "Documentation": "https://huggingface.co/your-username/rbf-svm-vertex",
    },
)
