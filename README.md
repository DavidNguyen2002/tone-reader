# tonereader

A Python library that detects sarcasm in text.

[![GitHub](https://img.shields.io/github/license/DavidNguyen2002/tonereader)]("https://github.com/DavidNguyen2002/tonereader/blob/main/LICENSE")
[![GitHub issues](https://img.shields.io/github/issues/DavidNguyen2002/tonereader)]("https://github.com/DavidNguyen2002/tonereader/issues")
[![Build Status](https://github.com/DavidNguyen2002/tonereader/workflows/Build%20Status/badge.svg?branch=main)](https://github.com/DavidNguyen2002/tonereader/actions?query=workflow%3A%22Build+Status%22)
[![codecov](https://codecov.io/gh/DavidNguyen2002/tonereader/branch/main/graph/badge.svg?token=58NMOY5XZE)](https://codecov.io/gh/DavidNguyen2002/tonereader)
[![PyPI](https://img.shields.io/pypi/v/tonereader)](https://pypi.org/project/tonereader/)
[![Read the Docs](https://img.shields.io/readthedocs/tonereader)](https://tonereader.readthedocs.io/en/latest/)

## Overview

Have you ever read a message from someone and were unsure about whether they were being serious or not? Is it difficult for you to figure out of someone is being sarcastic through text? Now, tonereader can help you with that!

Using training data from Reddit, tonereader can analyze text and determine whether or not the speaker is being sarcastic. So far, this library uses an ngram model which does not yield great results; however, I would like to make this project more sophisticated in the future.

## Installation

To install tonereader, simply run

```bash
pip install tonereader
```

## Usage

Right now, most of the methods written are used to train the model. These methods allow you to pass in more training data into the model.

To use the sarcasm-detecting function, simply run

```python
is_sarcastic(text)
```

which will return a boolean.

## Future plans

For the future, I would like this project to use a more sophisticated model to determine sarcasm. I would also like to add the ability to detect more tones/emotions as well.
