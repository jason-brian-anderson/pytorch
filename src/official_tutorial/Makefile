install:
	pip install --upgrade pip &&\
	pip install -r /workspaces/pytorch/requirements.txt

test:
	clear && python -m pytest -vv code.py 

format:
	black *.py

lint:
    #https://pypi.org/project/pylint-exit/
	clear && pylint --disable=R,C  code.py

all: install format lint test