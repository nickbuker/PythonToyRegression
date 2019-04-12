# Python Rosetta Stone

## Author
- Nick Buker

## Introduction
As I explore a variety of programming languages, I want to have a Rosetta Stone of sorts. For each language, I would like to explore the following topics:

- Project are structure
- Dependency handling
- Mathematical operations (particularly involving matrices)
- Testing
- Building GUIs

The hope is that build this project in each language will help me understand the relative strengths and weaknesses of each language as they relate to the above areas of interest.

## Project Structure
```
├── README.md
├── requirements.txt
└── src
    ├── LinearRegression.py
    ├── Model.py
    └── scoring.py
```
- `requirements.txt` - Python packages required to run this project
- `src` - Directory containing project source code
    - `LinearRegression.py` - Concrete implementation of the abstract Model class
    - `Model.py` - Abstract base class to set the project API
    - `scoring.py` - Functions used to score the model
