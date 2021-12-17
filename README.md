# Performant Nonholonomic RRT 

This was a term project for ENGR-E399: Autonomous Robotics taught by Prof. Lantou Liu during the Fall
2021 semester.

[Quick Demo](https://www.youtube.com/watch?v=FJg4Kc1kCQo)


### Installation

This project has only been tested on Python 3.8.12. Since it was a class project,
I don't intend to ever support other versions, though any version of 3.8 should work. 
Use pyenv or pyenv-win if you need to install a new version of Python.

After installing pyenv or pyenv-win, install required Python. Any version of 3.8 should work.

```shell
pyenv install 3.8.12
pyenv rehash
pyenv shell 3.8.12
```

Install Python's preferred package manager, pipenv.

```shell
pip install --user pipenv
```

Then download and install dependencies.

```shell
git clone https://github.com/EmDash00/NonholonomicRRT.git
cd NonholonomicRRT
pipenv install
```

You should now be able to run the app with `pipenv run python3 src/app.py`.

### Report

Checkout the Jupyter Notebook titled "Report.ipynb" included with the app.

To view:

Run: `pipenv run jupyter-lab` in the `NonholonomicRRT/` directory.
