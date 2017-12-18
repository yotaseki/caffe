project(testFCN)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time system filesystem)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

include_directories(
	${Boost_INCLUDE_DIRS}
	)
set(testFCNLibs
	main.cpp
	)

add_executable(testFCN
	${testFCNLibs}
	)

target_link_libraries(testFCN
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
	)
