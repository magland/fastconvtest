all:
	/scda/tools/bin/g++ main.cpp qute.cpp -o fastconvtest -fopenmp -O3 -lopenblas
