python3.10 -m venv ~/.venvs/env-stereopsis/
. ~/.venvs/env-stereopsis/bin/activate
pip install --upgrade pip
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
pip install -r ~/git/stereopsis/env-recipe.txt
