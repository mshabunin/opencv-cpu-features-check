### About:

This tool configures OpenCV library with specific CPU parameters and verifies results. Uses Python test for verification.

### Prerequisites:

1. x86_64, Linux, Docker
2. Folder hierarchy should be as follows:
   - <some-folder>
     - `opencv/` (OpenCV sources)
     - `opencv-cpu-features-check/` (this repository)
       - `scripts/` (main test script to be run inside container)
       - `workspace/` (location for builds and caches - will be created automatically)
       - `run.sh` (main script to be run on host)

### Instruction:

1. `git clone <this-repo>`
2. `cd <repo-dir>`
3. `./run.sh`