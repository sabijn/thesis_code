import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file')

    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()
    
    print(lines[900_000])