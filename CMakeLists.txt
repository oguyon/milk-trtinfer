# =====================================================================
# plugins/trtinfer/CMakeLists.txt
# =====================================================================

set(LIBNAME "milktrtinfer")
set(SRCNAME "trtinfer")

message(STATUS "Adding plugin: ${SRCNAME} -> ${LIBNAME}")

project(lib_${LIBNAME}_project)

set(SOURCEFILES
    trtinfer_datagen.c
)

set(INCLUDEFILES
    trtinfer.h
    trtinfer_datagen.h
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

install(TARGETS ${LIBNAME}
    EXPORT milkTargets
    DESTINATION lib
)

install(FILES ${INCLUDEFILES} DESTINATION include/${SRCNAME})

# Standalone compute unit
add_milk_standalone(trtinfer_datagen trtinfer_datagen.c)
target_link_libraries(milk-fpsexec-trtinfer_datagen PUBLIC ${LIBNAME} milkCOREMODiofits)

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
