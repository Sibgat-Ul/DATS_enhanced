import os.path
import sys
import setuptools

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "our trainer"))
    DISTNAME = "our trainer"

    setuptools.setup(
        name=DISTNAME,
        packages=setuptools.find_packages(),
        version="0.1",
        long_description_content_type="text/markdown",
    )
