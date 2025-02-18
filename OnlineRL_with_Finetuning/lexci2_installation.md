# Installing LExCI 2 on a Raspberry Pi 4

- Install an arm64/aarch64 operating system on the Pi. This how-to was tested
  with [Ubuntu MATE 22.04 LTS](https://ubuntu-mate.org/download/arm64/) and
  pip 22.0.4. Newer pip versions may not be able to install gym 0.21.0 because
  setuptools >= 66 doesn't support the way it lists its dependencies.
- Install some required packages by running
  `sudo apt install build-essential curl unzip psmisc npm`.
- Download the latest version of
  [Bazelisk](https://github.com/bazelbuild/bazelisk/releases) for the arm64
  architecture. Rename the file to *bazel* and move it somewhere you can later
  refer to. Here, we move the file to `/home/pi/Applications`.
- Create a configuration file for Bazel in your home directory. Here, the file
  is `/home/pi/.bazelrc`. Add the line
  `build --local_ram_resources=256 --local_cpu_resources=1` to the file in order
  to limit Bazel's usage of system resources (adjust the values if needed). Be
  aware that this doesn't constitute a hard restriction so the Pi could end up
  using more memory than that as we'll see below.
- If you want to perform a completely clean re-build and installation, you have
  to clear the cache of the utilised tools in `/home/pi/.cache`. The specific
  folders are `bazel`, `bazelisk`, and `pip`.
- Run LExCI 2's installation script. It'll be able to download and install the
  correct Python version, set up the LExCI 2 virtual environment, and install
  most necessary packages. However, there will be error messages concerning
  Ray/RLlib as it offers no Python wheels for arm64 processors.
- Activate the environment with `source /home/pi/.venv/lexci2/bin/activate` and
  run `pip install --upgrade pip` to update pip. Then, run
  `pip install setuptools==58.1.0 gym==0.21.0 tensorflow==2.11.0 pandas dm_tree gputil asammdf tabulate opencv-python scipy matplotlib`.
- Download Ray/RLlib 1.13.0 from the
  [official GitHub repository](https://github.com/ray-project/ray/tree/ray-1.13.0)
  and unzip the file. `cd` into the directory where the the extracted folder is
  located. Here, it's `/home/pi/Downloads`.
- Activate the LExCI 2 virtual environment by typing
  `source /home/pi/.venv/lexci2/bin/activate` and then run
  `export PATH="/home/pi/Applications":$PATH` so that Bazel/Bazelisk can be
  found. Also, type `export USE_BAZEL_VERSION="4.2.1"` to tell Bazelisk which
  version of Bazel to use.
- Execute `ray-ray-1.13.0/ci/env/install-bazel.sh`.
- Run `pushd ray-ray-1.13.0/dashboard/client`, `npm install`, `npm run build`,
  and `popd`.
- Type `cd ray-ray-1.13.0/python` and then run `pip install . --verbose`. The
  installation process takes some time and, depending on your hardware setup,
  may run out of memory a couple of times so that it is stopped by the OS. When
  the process has been killed, you must repeat `pip install . --verbose` until
  it finishes successfully. Bazel keeps files that have already been compiled so
  you don't always start at zero again.
- Patch the Ray/RLlib installation by manually performing the copy commands that
  are run in LExCI 2's installation script.
- When re-installing after Ray/RLlib has already been built, just activate the
  LExCI 2 virtual environment by typing
  `source /home/pi/.venv/lexci2/bin/activate` and then run
  `export PATH="/home/pi/Applications":$PATH` so that Bazel/Bazelisk can be
  found. Also, type `export USE_BAZEL_VERSION="4.2.1"` to tell Bazelisk which
  version of Bazel to use. Then, type `cd ray-ray-1.13.0/python` and then run
  `pip install . --verbose`. The other steps aren't necessary at this point.

