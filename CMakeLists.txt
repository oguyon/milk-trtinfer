# =====================================================================
# plugins/trtinfer/CMakeLists.txt
# =====================================================================

set(LIBNAME "milktrtinfer")
set(SRCNAME "trtinfer")

message(STATUS "Adding plugin: ${SRCNAME} -> ${LIBNAME}")

project(lib_${LIBNAME}_project)

set(SOURCEFILES
    trtinfer_datagen.c
    trtinfer_exec.c
    trt_backend.cpp
)

set(INCLUDEFILES
    trtinfer.h
    trtinfer_datagen.h
    trtinfer_exec.h
    trt_backend.h
    trtinfer_model.h
)

add_library(${LIBNAME} SHARED ${SRCNAME}.c ${SOURCEFILES})

target_include_directories(${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/${SRCNAME}>
)

target_link_libraries(${LIBNAME} PUBLIC
    CLIcore
    ImageStreamIO
    milkfps
    milkprocessinfo
    milkCOREMODmemory
    milkCOREMODiofits
    m
)

# Optional CUDA detection
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "[trtinfer] CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    target_compile_definitions(${LIBNAME} PUBLIC HAVE_CUDA=1)
    target_include_directories(${LIBNAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${LIBNAME} PUBLIC CUDA::cudart)
else()
    message(STATUS "[trtinfer] CUDA Toolkit not found. Building for CPU.")
endif()

# Optional TensorRT detection
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include /usr/local/cuda/include /opt/tensorrt/include
)
find_library(TENSORRT_LIB nvinfer
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/tensorrt/lib
)
find_library(TENSORRT_ONNX_LIB nvonnxparser
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/tensorrt/lib
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB AND TENSORRT_ONNX_LIB)
    message(STATUS "[trtinfer] TensorRT found: ${TENSORRT_LIB}")
    target_compile_definitions(${LIBNAME} PUBLIC HAVE_TENSORRT=1)
    target_include_directories(${LIBNAME} PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${LIBNAME} PUBLIC ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB})
else()
    message(STATUS "[trtinfer] TensorRT not found. Building with standalone/fallback backend.")
endif()

install(TARGETS ${LIBNAME}
    EXPORT milkTargets
    DESTINATION lib
)

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Standalone compute unit: trtinfer_datagen
add_milk_standalone(trtinfer_datagen trtinfer_datagen.c)
target_link_libraries(milk-fpsexec-trtinfer_datagen PUBLIC ${LIBNAME} milkCOREMODiofits)

# Standalone compute unit: trtinfer (inference engine)
add_milk_standalone(trtinfer trtinfer_exec.c)
target_link_libraries(milk-fpsexec-trtinfer PUBLIC ${LIBNAME})
set_target_properties(milk-fpsexec-trtinfer PROPERTIES LINKER_LANGUAGE CXX)

# Standalone C neural network trainer
add_executable(trtinfer_train_mlp trtinfer_train_mlp.c)
target_include_directories(trtinfer_train_mlp PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
target_link_libraries(trtinfer_train_mlp PRIVATE
    ${LIBNAME}
    ImageStreamIO
    m
)

# Standalone C validator
add_executable(trtinfer_check_accuracy trtinfer_check_accuracy.c)
target_include_directories(trtinfer_check_accuracy PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)
target_link_libraries(trtinfer_check_accuracy PRIVATE
    ${LIBNAME}
    ImageStreamIO
    m
)
