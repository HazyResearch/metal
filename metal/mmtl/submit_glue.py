import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model and make glue submission bundle", add_help=False
    )
    parser.add_argument("-mf", "--model-file")
    args = parser.parse_args()
