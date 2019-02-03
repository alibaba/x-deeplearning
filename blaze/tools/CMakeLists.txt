add_executable(gpu_status "gpu_status.cc")
target_link_libraries(gpu_status blaze)
install(TARGETS gpu_status DESTINATION bin)

install(FILES build_qed.py model_converter.py model_optimizer.py DESTINATION tools)
install(DIRECTORY example_model DESTINATION tools)
