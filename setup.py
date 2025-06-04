from setuptools import find_packages, setup

setup(
    name="cmaes_bezier_demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "scipy"],
    author="Weather Routing Research",
    description="Routing tools for continuous vector fields",
    long_description_content_type="text/markdown",
    url="https://github.com/Weather-Routing-Research/cmaes_bezier_demo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
