cmake_minimum_required(VERSION 3.0.2)
project(b2rm)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY urdf meshes launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
