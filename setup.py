from setuptools import setup, find_packages

setup(
    name="agent_framework",
    version="0.2.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"agent_framework.web": ["static/*.html", "static/*.css", "static/*.js"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0",
        "tiktoken",
        "aiofiles",
    ],
    extras_require={
        "research": [
            "mcp[cli]>=1.20,<2",
            "httpx",
            "beautifulsoup4",
        ],
        "web": [
            "fastapi",
            "uvicorn[standard]",
            "mcp>=1.20,<2",
            "httpx",
            "beautifulsoup4",
        ],
    },
    description="Interruptible and retained Master/Head/Node agent runtime",
)
