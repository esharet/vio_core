from setuptools import setup, find_packages

setup(
    name="if_vio",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "myapp=my_python_app.app:main"
        ]
    },
    install_requires=[],  # Add dependencies here if any
    author="Your Name",
    description="A simple Python app",
)