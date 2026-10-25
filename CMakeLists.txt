cmake_minimum_required(VERSION 3.24)
project(wordle-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Word lists compiled into the binaries.
set(WORDS_CPP ${CMAKE_BINARY_DIR}/words_data.cpp)
add_custom_command(OUTPUT ${WORDS_CPP}
	COMMAND ${CMAKE_COMMAND} -DOUT=${WORDS_CPP} -DALLOWED=${CMAKE_SOURCE_DIR}/data/allowed.txt
		-DANSWERS=${CMAKE_SOURCE_DIR}/data/answers.txt -DNYT=${CMAKE_SOURCE_DIR}/data/nyt-answers.txt -P ${CMAKE_SOURCE_DIR}/cmake/embed_words.cmake
	DEPENDS data/allowed.txt data/answers.txt data/nyt-answers.txt cmake/embed_words.cmake)

find_package(Threads REQUIRED)
add_library(wordlecore STATIC src/engine.cu src/exact.cpp src/game.cpp ${WORDS_CPP})
target_include_directories(wordlecore PUBLIC src)
target_link_libraries(wordlecore PUBLIC Threads::Threads)
target_compile_options(wordlecore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)

add_executable(wordle-gpu-cli src/cli.cpp)
target_link_libraries(wordle-gpu-cli wordlecore)

add_executable(wordle-gpu-solve src/solve.cpp)
target_link_libraries(wordle-gpu-solve wordlecore)

# The games database (wordle-gpu-import-games fills it).
find_package(SQLite3)
if(SQLite3_FOUND)
	add_library(gamesdb STATIC src/gamesdb.cpp)
	target_include_directories(gamesdb PUBLIC src)
	if(TARGET SQLite3::SQLite3)
		target_link_libraries(gamesdb PUBLIC SQLite3::SQLite3)
	else()
		target_link_libraries(gamesdb PUBLIC SQLite::SQLite3) # CMake before 4.3
	endif()
	add_executable(wordle-gpu-games src/games.cpp)
	target_link_libraries(wordle-gpu-games wordlecore gamesdb)
else()
	message(STATUS "SQLite3 not found: no games database support")
endif()

find_package(Qt6 COMPONENTS Widgets Concurrent)
if(Qt6_FOUND)
	set(CMAKE_AUTOMOC ON)
	add_executable(wordle-gpu src/gui/main.cpp src/gui/mainwindow.cpp src/gui/board.cpp src/gui/model.cpp)
	target_link_libraries(wordle-gpu wordlecore Qt6::Widgets)
	if(SQLite3_FOUND)
		target_sources(wordle-gpu PRIVATE src/gui/gamesdialog.cpp src/gui/gamesdialog.h)
		target_link_libraries(wordle-gpu gamesdb)
		target_compile_definitions(wordle-gpu PRIVATE WORDLE_GPU_GAMES)
	endif()
else()
	message(STATUS "Qt6 Widgets not found: building the CLI only")
endif()

install(TARGETS wordle-gpu-cli wordle-gpu-solve DESTINATION bin)
install(PROGRAMS tools/wordle-gpu-update-answers DESTINATION bin)
if(SQLite3_FOUND)
	install(TARGETS wordle-gpu-games DESTINATION bin)
	install(PROGRAMS tools/wordle-gpu-import-games DESTINATION bin)
	# The importer checks downloaded games against the same word lists.
	install(FILES data/allowed.txt data/nyt-answers.txt DESTINATION share/wordle-gpu/lists)
endif()
if(Qt6_FOUND)
	install(TARGETS wordle-gpu DESTINATION bin)
	install(FILES wordle-gpu.desktop DESTINATION share/applications)
endif()
