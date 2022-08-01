#!/usr/bin/env python3

import os
import shlex

from setuptools import find_packages, setup
from setuptools.command.test import test

src = os.path.abspath(os.path.join(__file__, ".."))


class DiscoverTests(test):
    """Discover and dispatch tests.
    """

    user_options = [
            ("args=", "a", "Extra arguments for pytest"),
    ]

    def initialize_options(self):
        test.initialize_options(self)
        self.args = ""

    def finalize_options(self):
        pass

    def run_tests(self):
        import pytest

        path = os.path.join(src, "tests")
        args = shlex.split(self.args)

        pytest.main([src, *args])


# Pull the version number:
with open(os.path.join(src, "ebcc", "__init__.py"), "r"):
    for line in f.readlines():
        if line.startswith("__version__"):
            __version__ = line.split()[2].strip("\"").strip("\'")
            break
    else:
        raise ValueError("Couldn't find a version number.")


setup(
    name="ebcc",
    version=__version__,
    description="Coupled cluster calculations on electron-boson systems",
    download="https://github.com/BoothGroup/ebcc",
    keywords=[
        "quantum", "chemistry",
        "electronic", "structure",
        "coupled", "cluster",
        "electron", "boson",
        "ccsd",
    ],
    author=",".join([
        "O. J. Backhouse",
        "G. H. Booth",
        "C. J. C. Scott",
    ]),
    author_email="vayesta.embedding@gmail.com",
    license="Apache License 2.0",  # TODO
    platforms=[
        "Linux",
        "Mac OS-X",
    ],
    python_requires=">=3.7",
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "License :: OSI Approved :: Apache Software License",  # TODO
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "numpy>=1.19.0",
        "pyscf>=2.0.0",
    ],
    cmdclass={
        "test": DiscoverTests,
    },
    tests_require=[
        "pytest",
        "pytest-cov",
    ],
    zip_safe=False,
)
