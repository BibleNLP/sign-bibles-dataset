import pathlib
import sys

def main():
    input_path = pathlib.Path("/inputs/sample.txt")   # hardcoded for now
    text = input_path.read_text()

    words = text.split()

    output_path = pathlib.Path("/outputs/words.txt")
    output_path.write_text("\n".join(words))

    print(f"Processed {len(words)} words. Output saved to {output_path}")

if __name__ == "__main__":
    main()
