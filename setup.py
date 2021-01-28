from setuptools import setup, find_packages


# The text of the README file
with open('README.md') as f:
    rm = f.read()

# This call to setup() does all the work
setup(
    name="pyma",
    version="0.0.1",
    description="Modal Analysis in Python",
    long_description=rm,
    long_description_content_type="text/markdown",
    url="https://github.com/TimothyRogers/pyMA",
    author="Tim Rogers",
    author_email="tim.rogers@sheffield.ac.uk",
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: BSD-3 License",
        "Programming Language :: Python :: 3.4+",
    ],
    packages=['pyma'],
    package_dir={'':'src'},
    include_package_data=False,
    install_requires=[
        "numpy"
    ],
)