package algorithms

import (
	"math"
	"math/rand"
)

func Adjacency(points [][2]float64) [][]float64 {
	matrix := make([][]float64, len(points))
	for i := range matrix {
		matrix[i] = make([]float64, len(points))
	}
	for i := 0; i < len(matrix); i++ {
		for j := i + 1; j < len(matrix); j++ {
			temp := math.Sqrt(math.Pow(points[j][0]-points[i][0], 2) + math.Pow(points[j][1]-points[i][1], 2))
			matrix[j][i] = temp
			matrix[i][j] = temp
		}
	}
	return matrix
}

func Generator(size int) [][2]float64 {
	points := make([][2]float64, size)
	for i := 0; i < len(points); i++ {
		points[i][0] = rand.Float64() * float64(size)
		points[i][1] = rand.Float64() * float64(size)
	}
	return points
}

func Min(prices []float64, visited []bool) (int, float64) {
	index, min := 0, math.MaxFloat64
	for i, val := range prices {
		if val < min && val > 0 && !visited[i] {
			index, min = i, val
		}
	}
	return index, min
}

func Greedy(matrix [][]float64) (float64, []int) {
	length, tour := 0.0, make([]int, len(matrix))
	visited := make([]bool, len(matrix))
	point := rand.Intn(len(matrix))

	visited[point] = true
	tour[0] = point

	for i := 1; i < len(matrix); i++ {
		newPoint, price := Min(matrix[point], visited)
		visited[newPoint] = true
		length += price
		tour[i] = newPoint
		point = newPoint
	}
	return length, tour
}
