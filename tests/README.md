# Tests

This directory contains the tests for the project that can be run with pytest or by running the `test_*.py` or `manual_test_*.py` files directly.

Manual tests may require significant amount of time or manual comparison of results, so they are not run by CI/CD.

The `build_test_*.sh` scripts test the build process of the project under different distributions and environments.
To run them docker and cibuildwheels is required.
