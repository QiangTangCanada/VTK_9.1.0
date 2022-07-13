# Moment Invariants Filters

The theory and the algorithm are described in [Roxana Bujack and
Hans Hagen: "Moment Invariants for Multi-Dimensional Data"](http://www.informatik.uni-leipzig.de/~bujack/2017TensorDagstuhl.pdf)

Developed by Roxana Bujack and Karen Tsai at Los Alamos National Laboratory.

This repository is a remote module for [VTK](https://www.vtk.org). It follows the structure described in [Adding Remote Modules to VTK](http://www.vtk.org/Wiki/VTK/Remote_Modules).

The FFT computation is done with DFFTLIB (https://github.com/jglaser/dfftlib).

## Workflow

Our recommended workflow is a bit different than a normal git repository.

### Clone and build VTK

1. [Clone with git][]
2. [Configure and build](https://www.vtk.org/Wiki/VTK/Configure_and_Build)
	1. Keep 'BUILD_TESTING' checked for module tests to be enabled.
3. Using cmake-gui, enable options for this remote module:
	1. Check the 'advanced' box.
	2. Check `VTK_MODULE_ENABLE_VTK_MomentInvariants` and click `Configure` to download the module, and let the following option appear:
	3. Check `VTK_MODULE_ENABLE_VTK_ParallelMomentInvariants` if parallel version is desired.
4. Build VTK
	1. Typically `ninja` in the build directory
5. Run tests
	1. In the build directory, `ctest` runs all the enabled tests. Add `-V` to see test output.
	2. For serial moments test: `ctest -R patternDetectionTestSimple`
	3. For parallel moments test: `ctest -R TestParallelComputeShort`

### Development

After running CMake, this repository is cloned in the VTK source tree, under `Remote\MomentInvariants`. To start making changes:

1. `cd Remote\MomentInvariants`
2. `git fetch`
3. `git checkout master`
4. `git pull`

This ensures you have the latest contents of the repository. You need to change the `GIT_TAG` in `...remote.cmake` to `master` as well. After making changes, in the VTK build directory:

1. `cmake . ; ninja` to rebuild
2. `ctest -R Moment` to run tests.

To add your changes to this repository (which includes updating this file), in the `Remote\MomentInvariants` directory:

1. `git add .`
2. `git commit`
3. `git push`

Finally, when you are ready to update the module in VTK for distribution:

1. `git log -n 1`, and copy the hash string after the word `commit`
2. Edit the file `Remote\MomentInvariants.remote.cmake`, and replace the hash string after `GIT_TAG`
3. Commit and submit this change to VTK as a pull request, as detailed in [VTK developer docs][Clone with git].


[Clone with git]: https://gitlab.kitware.com/vtk/vtk/blob/master/Documentation/dev/git/develop.md
