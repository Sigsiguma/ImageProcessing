# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/siguma/CLionProjects/ImageProcessing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/siguma/CLionProjects/ImageProcessing/cmake-build-default

# Include any dependencies generated for this target.
include CMakeFiles/ImageProcessing.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageProcessing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageProcessing.dir/flags.make

CMakeFiles/ImageProcessing.dir/main.cpp.o: CMakeFiles/ImageProcessing.dir/flags.make
CMakeFiles/ImageProcessing.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/siguma/CLionProjects/ImageProcessing/cmake-build-default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ImageProcessing.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageProcessing.dir/main.cpp.o -c /Users/siguma/CLionProjects/ImageProcessing/main.cpp

CMakeFiles/ImageProcessing.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageProcessing.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/siguma/CLionProjects/ImageProcessing/main.cpp > CMakeFiles/ImageProcessing.dir/main.cpp.i

CMakeFiles/ImageProcessing.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageProcessing.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/siguma/CLionProjects/ImageProcessing/main.cpp -o CMakeFiles/ImageProcessing.dir/main.cpp.s

CMakeFiles/ImageProcessing.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/ImageProcessing.dir/main.cpp.o.requires

CMakeFiles/ImageProcessing.dir/main.cpp.o.provides: CMakeFiles/ImageProcessing.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageProcessing.dir/build.make CMakeFiles/ImageProcessing.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/ImageProcessing.dir/main.cpp.o.provides

CMakeFiles/ImageProcessing.dir/main.cpp.o.provides.build: CMakeFiles/ImageProcessing.dir/main.cpp.o


CMakeFiles/ImageProcessing.dir/sample_code.cpp.o: CMakeFiles/ImageProcessing.dir/flags.make
CMakeFiles/ImageProcessing.dir/sample_code.cpp.o: ../sample_code.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/siguma/CLionProjects/ImageProcessing/cmake-build-default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ImageProcessing.dir/sample_code.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageProcessing.dir/sample_code.cpp.o -c /Users/siguma/CLionProjects/ImageProcessing/sample_code.cpp

CMakeFiles/ImageProcessing.dir/sample_code.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageProcessing.dir/sample_code.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/siguma/CLionProjects/ImageProcessing/sample_code.cpp > CMakeFiles/ImageProcessing.dir/sample_code.cpp.i

CMakeFiles/ImageProcessing.dir/sample_code.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageProcessing.dir/sample_code.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/siguma/CLionProjects/ImageProcessing/sample_code.cpp -o CMakeFiles/ImageProcessing.dir/sample_code.cpp.s

CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.requires:

.PHONY : CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.requires

CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.provides: CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageProcessing.dir/build.make CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.provides.build
.PHONY : CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.provides

CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.provides.build: CMakeFiles/ImageProcessing.dir/sample_code.cpp.o


CMakeFiles/ImageProcessing.dir/plot.cpp.o: CMakeFiles/ImageProcessing.dir/flags.make
CMakeFiles/ImageProcessing.dir/plot.cpp.o: ../plot.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/siguma/CLionProjects/ImageProcessing/cmake-build-default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ImageProcessing.dir/plot.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageProcessing.dir/plot.cpp.o -c /Users/siguma/CLionProjects/ImageProcessing/plot.cpp

CMakeFiles/ImageProcessing.dir/plot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageProcessing.dir/plot.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/siguma/CLionProjects/ImageProcessing/plot.cpp > CMakeFiles/ImageProcessing.dir/plot.cpp.i

CMakeFiles/ImageProcessing.dir/plot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageProcessing.dir/plot.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/siguma/CLionProjects/ImageProcessing/plot.cpp -o CMakeFiles/ImageProcessing.dir/plot.cpp.s

CMakeFiles/ImageProcessing.dir/plot.cpp.o.requires:

.PHONY : CMakeFiles/ImageProcessing.dir/plot.cpp.o.requires

CMakeFiles/ImageProcessing.dir/plot.cpp.o.provides: CMakeFiles/ImageProcessing.dir/plot.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageProcessing.dir/build.make CMakeFiles/ImageProcessing.dir/plot.cpp.o.provides.build
.PHONY : CMakeFiles/ImageProcessing.dir/plot.cpp.o.provides

CMakeFiles/ImageProcessing.dir/plot.cpp.o.provides.build: CMakeFiles/ImageProcessing.dir/plot.cpp.o


# Object files for target ImageProcessing
ImageProcessing_OBJECTS = \
"CMakeFiles/ImageProcessing.dir/main.cpp.o" \
"CMakeFiles/ImageProcessing.dir/sample_code.cpp.o" \
"CMakeFiles/ImageProcessing.dir/plot.cpp.o"

# External object files for target ImageProcessing
ImageProcessing_EXTERNAL_OBJECTS =

../bin/ImageProcessing: CMakeFiles/ImageProcessing.dir/main.cpp.o
../bin/ImageProcessing: CMakeFiles/ImageProcessing.dir/sample_code.cpp.o
../bin/ImageProcessing: CMakeFiles/ImageProcessing.dir/plot.cpp.o
../bin/ImageProcessing: CMakeFiles/ImageProcessing.dir/build.make
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_stitching.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_superres.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_videostab.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_aruco.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_bgsegm.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_bioinspired.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_ccalib.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_dpm.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_fuzzy.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_line_descriptor.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_optflow.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_reg.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_saliency.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_stereo.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_structured_light.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_surface_matching.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_tracking.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_xfeatures2d.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_ximgproc.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_xobjdetect.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_xphoto.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_shape.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_phase_unwrapping.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_rgbd.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_calib3d.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_video.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_datasets.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_dnn.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_face.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_plot.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_text.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_features2d.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_flann.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_objdetect.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_ml.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_highgui.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_photo.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_videoio.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_imgcodecs.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_imgproc.3.2.0.dylib
../bin/ImageProcessing: /usr/local/opt/opencv3/lib/libopencv_core.3.2.0.dylib
../bin/ImageProcessing: CMakeFiles/ImageProcessing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/siguma/CLionProjects/ImageProcessing/cmake-build-default/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/ImageProcessing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageProcessing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageProcessing.dir/build: ../bin/ImageProcessing

.PHONY : CMakeFiles/ImageProcessing.dir/build

CMakeFiles/ImageProcessing.dir/requires: CMakeFiles/ImageProcessing.dir/main.cpp.o.requires
CMakeFiles/ImageProcessing.dir/requires: CMakeFiles/ImageProcessing.dir/sample_code.cpp.o.requires
CMakeFiles/ImageProcessing.dir/requires: CMakeFiles/ImageProcessing.dir/plot.cpp.o.requires

.PHONY : CMakeFiles/ImageProcessing.dir/requires

CMakeFiles/ImageProcessing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageProcessing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageProcessing.dir/clean

CMakeFiles/ImageProcessing.dir/depend:
	cd /Users/siguma/CLionProjects/ImageProcessing/cmake-build-default && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/siguma/CLionProjects/ImageProcessing /Users/siguma/CLionProjects/ImageProcessing /Users/siguma/CLionProjects/ImageProcessing/cmake-build-default /Users/siguma/CLionProjects/ImageProcessing/cmake-build-default /Users/siguma/CLionProjects/ImageProcessing/cmake-build-default/CMakeFiles/ImageProcessing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageProcessing.dir/depend

