set dotenv-load := true
files    := "distribution_algebra tests"

# List all recipes.
list:
    @just --list

# Install the project and its dependencies.
install:
    uv python pin 3.10  # Minimal supported python version.
    uv sync --all-extras

# Update all packages. Create type-stubs.
update:
    uv sync --all-extras --upgrade
    uv run safety check
    uv run pyright --verbose --createstub numpy
    uv run pyright --verbose --createstub scipy
    uv run pyright --verbose --createstub nptyping
    uv run pyright --verbose --createstub matplotlib
    uv run pyright --verbose --createstub matplotlib.pyplot
    uv run pyright --verbose --createstub multipledispatch
    uv run pyright --verbose --createstub attr

# Typecheck the code using pyright & mypy.
typecheck:
    uv run pyright {{ files }}
    uv run -m mypy --config-file pyproject.toml {{ files }}

# Run tests.
test flags="-v --cov=distribution_algebra":
    uv run pytest tests {{ flags }}

# Lint the files.
lint:
    uv run isort {{ files }}

# Create website docs and serve on localhost.
docs:
    just _write_md
    uv install
    uv run mkdocs serve

# Create website docs and push to github-pages.
publish:
    # Run `just docs` if you need to debug this command.
    just _write_md
    uv run mkdocs gh-deploy --force

# Write the markdown files.
_write_md:
    #!/usr/bin/env bash
    for file in `ls distribution_algebra/*.py`; do
      base=$(basename -- $file .py)
      if [ "$base" = "__init__" ]; then
        continue
      fi
      #echo "## $base.py" > docs/$base".md"
      echo "::: distribution_algebra.$base" > docs/$base".md"
    done

# Bump package version, tag on git, and push package to PyPI.
pypi type="patch":
    # Check if pyroject.toml has already been modified.
    git diff --exit-code pyproject.toml && exit
    # Bump package version in pyproject.toml.
    uv version {{ type }}
    # Commit the pyproject.toml file.
    git add pyproject.toml
    git commit -m "bump package version"
    # Create lightweight git tag.
    git tag `uv version --short`
    git push origin `uv version --short`
    # Publish to PyPI.
    @uv publish \
    --username={{ env_var("PYPI_USERNAME_DISTRIBUTION_ALGEBRA") }} \
    --password={{ env_var("PYPI_PASSWORD_DISTRIBUTION_ALGEBRA") }} \
    --build --verbose


_badges:
    uv run python -m make_badges
