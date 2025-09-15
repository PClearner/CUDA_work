add_executable(rmsnorm m_rmsnorm.cu)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(rmsnorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()
target_compile_options(rmsnorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
target_compile_options(rmsnorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)

# target_compile_options(rmsnorm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64>)
