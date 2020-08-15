package main

import (
	"algorithms"
	"fmt"
	"time"
)

func main() {
	size := 500
	points := algorithms.Generator(size)
	matrix := algorithms.Adjacency(points)
	length, tour := algorithms.Greedy(matrix)

	start := time.Now()
	opt := algorithms.TwoOpt{Length: length, Tour: tour, Matrix: matrix}
	opt.Optimize()
	end := time.Now()
	fmt.Println(end.Sub(start))
}
