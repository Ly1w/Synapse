from setuptools import setup, find_packages

setup(
    name="agent_framework",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0",
        "tiktoken",
        "aiofiles",
    ],
    description="Hierarchical multi-agent framework with Master/Head/Node architecture",
)
