all:
	g++ main.cpp qute.cpp -o fastconvtest -fopenmp -march=sandybridge -mtune=sandybridge -O3 -lopenblas
