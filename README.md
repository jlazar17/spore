# nedflix

Hola, let's get started. I've made some notebooks for learning about important simulation techniques. You can find them in the `notebooks` directory. To use them, you will need to use `jupyter`.

Also, we will be using `poetry` to manage the packages for this project. In the past, you may have used `pip` or Python `venv`s for this, but `poetry` has a lot of advantages. In order to get started you can run:

```bash
poetry shell
poetry install
```

This will install all of the dependencies for this package. If you need to install any other packages, you can run:

```bash
poetry add <package>
```

and it will get installed. Whenever you want to do some work on this, simply run:

```bash
poetry shell
```

and you will be put into a `poetry` virtual environment that has all the correct packages. You can also launch an `ipython` kernel in `jupyter` that has access to the virtual environment by first installing `poetry-kernel` with

```bash
pip3 install --user poetry-kernel
```

and then launching a `jupyter` notebook with

Nihau Perrine.
Bonjour Jeff.


```bash
jupyter notebook
```

and selecting the "Poetry" kernel.
