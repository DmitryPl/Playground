package algorithms

import "fmt"

type TwoOpt struct {
	Length float64
	Tour   []int
	Matrix [][]float64
}

func swap(tour []int, x int, y int) []int {
	size, temp := len(tour), 0
	if x < y {
		temp = (y - x + 1) / 2
	}
	if x > y {
		temp = ((size - x) + y + 2) / 2
	}
	for i := 0; i < temp; i++ {
		first, second := (x+i)%size, (y-i)%size
		tour[first], tour[second] = tour[second], tour[first]
	}
	return tour
}

func (opt TwoOpt) Optimize() (float64, []int) {
	iteration, gain := 0, 1.0
	fmt.Println(fmt.Sprintf("%d: %f", iteration, opt.Length))
	for gain > 0 {
		gain = opt.twoOpt()
		opt.Length -= gain
		iteration++
		fmt.Println(fmt.Sprintf("%d: %f", iteration, opt.Length))
	}
	return opt.Length, opt.Tour
}

func (opt TwoOpt) twoOpt() float64 {
	x, y, gain := opt.improve()

	if gain > 0 {
		opt.Tour = swap(opt.Tour, x+1, y)
	}
	return gain
}

func (opt TwoOpt) improve() (int, int, float64) {
	i, j, bestChange := 0, 0, 0.0
	size := len(opt.Tour)

	for n := 0; n < size; n++ {
		for m := n + 2; m < size; m++ {
			x1, x2 := opt.Tour[n%size], opt.Tour[m%size]
			y1, y2 := opt.Tour[(n+1)%size], opt.Tour[(m+1)%size]
			change := opt.Matrix[x1][x2] + opt.Matrix[y1][y2]
			change -= opt.Matrix[x1][y1] + opt.Matrix[x2][y2]
			if change < bestChange {
				bestChange = change
				i, j = n, m
			}
		}
	}
	return i, j, -bestChange
}
