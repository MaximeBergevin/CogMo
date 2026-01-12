from setuptools import setup, find_packages

# Run pip install -e . to install the package in editable mode
# This allows you to import the package from anywhere on your system during development
setup(
    name = 'cogmo_toolkit',
    version = '0.1.0',
    packages = find_packages(where = 'src'),
    package_dir = {'': 'src'}
)