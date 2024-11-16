# HELLO WORLD
# WELCOME TO YEAGER UTILITLIES.

# Currently these are just helpful tools for data production, analysis and visualization.

# BUILD ENVIRONMENT
/usr/gapps/python/toss_4_x86_64_ib/python-3.11.5/bin/python -m venv python3_env
source python3_env/bin/activate
pip install --upgrade pip


# INSTALLATION #
# PRE GIT CLONE A VERSION OF SSAPY
cd ssapy/
git submodule update --init --recursive
python3 setup.py build
python3 setup.py install
cd ../

cd yeager_utils/
python3 setup.py build
python3 setup.py install
cd ../
