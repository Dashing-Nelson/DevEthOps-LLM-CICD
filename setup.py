#!/usr/bin/env python3
"""
Setup script for DevEthOps-LLM-CICD package.
"""

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
    name="devethops-llm-cicd",
    version="1.0.0",
    author="DevEthOps Team",
    author_email="contact@devethops.com",
    description="Ethical CI/CD Pipeline for Machine Learning with specialized LLM support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Dashing-Nelson/ci_cd_ethical_checkpoint",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
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
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "devethops-pipeline=devethops.pipeline:main",
            "devethops-run=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "devethops": ["configs/*.yaml", "configs/*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Dashing-Nelson/ci_cd_ethical_checkpoint/issues",
        "Source": "https://github.com/Dashing-Nelson/ci_cd_ethical_checkpoint",
        "Documentation": "https://github.com/Dashing-Nelson/ci_cd_ethical_checkpoint/docs",
    },
)
