cmake_minimum_required(VERSION 3.0.2)
project(image)

find_package(catkin REQUIRED COMPONENTS cv_bridge pcl_ros image_transport message_generation sensor_msgs)

# add the resized image message
generate_messages(DEPENDENCIES sensor_msgs)

catkin_package(CATKIN_DEPENDS cv_bridge image_transport message_runtime sensor_msgs)

find_package(OpenCV)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


# add the subscriber example
add_executable(image src/compressed_sub.cpp)
add_dependencies(image ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(image ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})



# Mark executables and/or libraries for installation
install(TARGETS image 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES resized_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
