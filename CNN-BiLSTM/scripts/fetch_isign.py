"""Explicitly invoked future acquisition; ordinary package imports never download assets."""
from isl_slt.data.acquisition import fetch_main

if __name__ == "__main__":
    fetch_main()
