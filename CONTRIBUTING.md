## Contributing

Quick Guide to get started with contributing to tonereader!

### How to get started

#### Prerequisites

- Python >= 3.7

#### Development Process

1. Install and set up tonereader on your computer

   - Go to the GitHub page for tonereader (https://github.com/DavidNguyen2002/tonereader) and click the "fork" button

   - Clone your forked copy of the repo onto your local computer
   - Go into the directory with `cd tonereader`
   - Run `make develop` to install all dev-dependencies

2. Make your changes/additions

   - Make a new branch with `git branch NAME_OF_BRANCH`
     - Since the name of the branch will be visible in your pull request, please name your branch something descriptive
   - Switch to the branch that you just made with `git checkout NAME_OF_BRANCH`
   - Start making your changes and additions to the project
   - After making your changes, be sure to write tests for any new features that you add
     - These tests go in `tonereader/tests/`

3. Submitting your contribution
   - When you finish making your changes, run the following commands:
     - `make test` to make sure that your changes do not accidentally break anything
     - `make coverage` to make sure that this project still has a high code coverage percentage
     - `make lint` to make sure that your code is consistent with the rest of the codebase
     - `make format` to automatically format your code to be consistent with the rest of the codebase
   - Once this is done, commit all of your changes and push them to GitHub with `git push origin NAME_OF_BRANCH`
   - Then submit a pull request for these changes on GitHub
     - Follow [GitHub's guide to making PRs from forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) if you are unsure on how to do this

Thanks for contributing!
