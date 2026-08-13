import os
import sys
import argparse
import subprocess
import time

# This script:
#   1. Accepts test input directory
#   2. Accepts output directory
#   3. Accepts trained model weights
#   4. Runs the existing inference pipeline
#   5. Produces restored .npy images
# No training is performed.


def main():

    parser = argparse.ArgumentParser(
        description="KLA Hackathon 2026 - Final Evaluation Script"
    )
    parser.add_argument(
    "--input_dir",
    default=os.path.join("Data-public", "Test_NoisyLR", "NoisyLR")
)

    parser.add_argument(
    "--output_dir",
    default="Restored_Test_Output"
)

    parser.add_argument(
    "--weights",
    default=os.path.join("weights", "best_model.pth")
)
    

    args = parser.parse_args()

    print("=" * 70)
    print("KLA HACKATHON 2026 - FINAL EVALUATION")
    print("=" * 70)

    print("\nInput directory :")
    print(args.input_dir)

    print("\nOutput directory:")
    print(args.output_dir)

    print("\nModel weights:")
    print(args.weights)

   
    # Check paths


    if not os.path.isdir(args.input_dir):
        print("\nERROR: Input directory does not exist.")
        print(args.input_dir)
        sys.exit(1)

    if not os.path.isfile(args.weights):
        print("\nERROR: Model weights not found.")
        print(args.weights)
        sys.exit(1)

    # Create output directory
    

    os.makedirs(args.output_dir, exist_ok=True)

   
    # Check inference.py
   

    project_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(project_dir, "inference.py")

    if not os.path.isfile(inference_script):
        print("\nERROR: inference.py not found.")
        print(inference_script)
        sys.exit(1)

    
    # Count input files
   

    input_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(".npy")
    ]

    print("\nInput files:", len(input_files))

    if len(input_files) == 0:
        print("\nERROR: No .npy files found in input directory.")
        sys.exit(1)

    # Run inference
   

    print("\n" + "=" * 70)
    print("STARTING MODEL INFERENCE")
    print("=" * 70)

    start_time = time.time()

    command = [
        sys.executable,
        inference_script,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--weights",
        args.weights
    ]

    print("\nRunning:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        print("\n" + "=" * 70)
        print("EVALUATION FAILED")
        print("=" * 70)
        print(
            "\ninference.py returned an error."
        )
        sys.exit(result.returncode)

    elapsed = time.time() - start_time

   
    # Verify outputs
   

    output_files = [
        f for f in os.listdir(args.output_dir)
        if f.lower().endswith(".npy")
    ]

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(f"\nInput files     : {len(input_files)}")
    print(f"Output files    : {len(output_files)}")
    print(f"Total time      : {elapsed:.2f} seconds")

   
    # Output verification
  

    if len(output_files) == 0:
        print("\nERROR: No restored outputs were generated.")
        sys.exit(1)

    if len(output_files) != len(input_files):
        print(
            "\nWARNING: Number of output files does not match "
            "number of input files."
        )

    print("\nRestored outputs:")
    print(args.output_dir)

    print("\nNo training was performed.")
    print("The trained checkpoint was used only for inference.")

    print("\n" + "=" * 70)
    print("READY FOR KLA BENCHMARKING")
    print("=" * 70)


if __name__ == "__main__":
    main()