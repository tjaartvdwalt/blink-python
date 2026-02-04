#!/usr/bin/python3
import glob
import sys
# import typer


def main(data_dir: str = "sample_data"):
    for prefix in ["blink", "non_blink"]:
        out_files = []

        for files in list(
            set(glob.glob(f"{data_dir}/{prefix}_*.ear"))
            - set(glob.glob(f"{data_dir}/{prefix}_dist.ear"))
        ):
            out_files.append(files)

        out_files.sort()
        with open(f"{data_dir}/{prefix}_dist.ear", "w") as out_file:
            for fname in out_files:
                print(fname)
                with open(fname) as in_file:
                    i = 1
                    for line in in_file:
                        if len(line.split()) == 13:
                            out_file.write(line)
                        else:
                            print(f"Error in file: {fname} line: {i + 1}")
                            print(f"{line}")
                        i += 1



if __name__ == "__main__":
    main(sys.argv[1])
#   typer.run(main)

