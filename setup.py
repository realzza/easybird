import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easybird",
    version="0.0.2",
    author="Ziang Zhou",
    author_email="ziang.zhou518@gmail.com",
    description="A toolkit for Bird Activity Detection (BAD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/realzza/easybird",
    project_urls={
        "Bug Tracker": "https://github.com/realzza/easybird/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data = {
        'easybird': ['models/*']
    },
    python_requires=">=3.6",
)