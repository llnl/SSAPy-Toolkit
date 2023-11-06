from setuptools import setup, find_packages

setup(
    name="yeager_utils",
    version="1.0",
    description='Helpful python wrapped utilities',
    author='Travis R. Yeager',
    author_email='yeagerastro@gmail.com',
    packages=find_packages(),
    install_requires=[
    ],
    zip_safe=False,
    include_package_data=True,
)

# 'astropy',
# 'h5py',
# 'numpy',
# 'pandas',
# 'rebound',
# 'ssapy',