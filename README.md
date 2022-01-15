# kaggle-titanic
Kaggle Titanic Machine Learning Competition - Available on https://www.kaggle.com/c/titanic/

The goal of this repo is to apply Python programming best practices to a real project. This includes:

1. Divide code into modules and functions according to the *Single Resposabiliy Principle*
2. Use *Inheritance* to avoid code duplication (DRY Principle)
3. Use *Version Control* and *Conventional Commits*
4. Isolate the requirements with *virtual environments* using *venv*
5. Test the code using *pytest*
6. Apply *type hinting* and *type checking* with *mypy*
7. Define a *style guide* and check consistency with *pylint*

The project is composed by a *preprocess* concrete class and *model* base, which is base for the ML concrete models: Multi-layer Perceptron, Random Forest and K-Nearest Neighbors.

The pipeline module calls the necessary process modules and provide the output file. The final output is given by an ensemble method of the ML models individual outputs.
