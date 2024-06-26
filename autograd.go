package main

import (
	"fmt"
	"math"
	"math/rand"
	"encoding/csv"
    "os"
    "strconv"
)



// -- Structs --

type Value struct {
	data float64
	grad float64
	backward func()
	prev []*Value
	op   string
}

type Neuron struct {
	w []*Value
	b *Value
	nonlin bool

}

type Layer struct {
	neurons []*Neuron
}

type MLP struct {
	sizes []int
	layers []*Layer
}



// -- Ops --

func New(f float64) *Value {
	return &Value{data: f}
}

func Add(a, b *Value) *Value {
	out := &Value{
		data: a.data + b.data,
		prev: []*Value{a, b},
		op: "+",
	}
	out.backward = func() {
		a.grad += out.grad
		b.grad += out.grad
	}
	return out
}

func Mul(a, b *Value) *Value {
	out := &Value{
		data: a.data * b.data,
		prev: []*Value{a, b},
		op: "*",
	}
	out.backward = func() {
		a.grad += b.data * out.grad
		b.grad += a.data * out.grad
	}
	return out
}


func Neg(x *Value) *Value {
	return Mul(x, New(-1))
}

func Sub(a, b *Value) *Value {
	return Add(a, Neg(b))
}


func Pow(a, b *Value) *Value {
	out := &Value{
		data: math.Pow(a.data, b.data),
		prev: []*Value{a, b},
		op:   "Pow",
	}
	out.backward = func() {
		a.grad += (b.data * math.Pow(a.data, b.data-1)) * out.grad
		b.grad += (a.data * math.Pow(b.data, a.data-1)) * out.grad
	}
	return out
}

func Div(a, b *Value) *Value {
	return Mul(a, Pow(b, New(-1)))
}

func ReLU(a *Value) *Value {
	out := &Value{
		data: func() float64 {
			if a.data > 0 {
				return a.data
			}
			return 0
		}(),
		prev: []*Value{a,},
		op: "ReLU",
	}
	out.backward = func() {
		if a.data > 0 {
			a.grad += out.grad
		}
	}
	return out
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}
	topo = buildTopo(v, topo, visited)

	v.grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		if len(topo[i].prev) != 0 {
			topo[i].backward()
		}
	}
}

func buildTopo(v *Value, topo []*Value, visited map[*Value]bool) []*Value {
	if !visited[v] {
		visited[v] = true
		for _, prev := range v.prev {
			topo = buildTopo(prev, topo, visited)
		}
		topo = append(topo, v)
	} 
	return topo

}



// -- Neuron --

func NewNeuron(size int, nonlin bool) *Neuron {
    w := make([]*Value, size)
    for i := 0; i < size; i++ {
        w[i] = New(rand.NormFloat64() * math.Sqrt(2.0 / float64(size)))
    }
    b := New(0)

    n := &Neuron{
        w:      w,
        b:      b,
        nonlin: nonlin,
    }
    return n
}

func (n *Neuron) Forward(x []*Value) *Value {
	out := n.b
	for i := 0; i < len(x); i++ {
		out = Add(out, Mul(n.w[i], x[i]))
	}
	if n.nonlin {
		out = ReLU(out)
	}
	
	return out
}

func (n *Neuron) Parameters() []*Value {
	return append(n.w, n.b)
}



// -- Layer --

func NewLayer(in, out int, nonlin bool) *Layer {
	neurons := make([]*Neuron, out)
	for i := 0; i < out; i++ {

		neurons[i] = NewNeuron(in, nonlin)
	}
	layer := &Layer{neurons: neurons}
	return layer
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.neurons))

	for i := 0; i < len(l.neurons); i++ {
		out[i] = l.neurons[i].Forward(x)
	}
	return out
}
func (l *Layer) Parameters() []*Value {
	res := []*Value{}
	for _, n := range l.neurons {
		res = append(res, n.Parameters()...)
	}
	return res
}



// -- MLP --

func NewMLP(nin int, nouts []int) *MLP {
	layers := make([]*Layer, len(nouts))
	sizes := append([]int{nin}, nouts...)

	for i := 0; i < len(nouts); i++ {
		layers[i] = NewLayer(sizes[i], sizes[i+1], i != len(nouts)-1)
	}
	MLP := &MLP{sizes:sizes, layers:layers}
	return MLP
}

func (mlp *MLP) Forward(x []*Value) []*Value {
	for i := 0; i < len(mlp.layers); i++ {
		x = mlp.layers[i].Forward(x)
	}
	return x
}

func (mlp *MLP) Parameters() []*Value {
	res := []*Value{}
	for _, l := range mlp.layers {
		res = append(res, l.Parameters()...)
	}
	return res
}



// -- Helpers

func MSE(x, y []*Value) *Value {
	loss := New(0)

	for i := 0; i < len(x); i++ {
		loss = Add(loss, Pow(Sub(x[i], y[i]), New(2)))
	}

	loss = Div(loss, New(float64(len(x))))
	return loss
}

// LoadCSV loads CSV data into a slice of slices of *Value
func LoadCSV(filename string) ([][]*Value, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, err
    }

    data := make([][]*Value, len(records))
    for i, record := range records {
        row := make([]*Value, len(record))
        for j, valueStr := range record {
            value, err := strconv.ParseFloat(valueStr, 64)
            if err != nil {
                return nil, err
            }
            row[j] = New(value)
        }
        data[i] = row
    }

    return data, nil
}

// LoadSingleColumnCSV loads CSV data into a slice of *Value
func LoadSingleColumnCSV(filename string) ([]*Value, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, err
    }

    data := make([]*Value, len(records))
    for i, record := range records {
        value, err := strconv.ParseFloat(record[0], 64)
        if err != nil {
            return nil, err
        }
        data[i] = New(value)
    }

    return data, nil
}



func main() {
	xs, err := LoadCSV("features.csv")
    if err != nil {
        fmt.Println("Error loading features data:", err)
        return
    }

    // Load output dataset ys from CSV
    ys, err := LoadSingleColumnCSV("targets.csv")
    if err != nil {
        fmt.Println("Error loading targets data:", err)
        return
    }

	fmt.Printf("%v\n",len(xs))
	fmt.Printf("%v\n",len(xs[0]))
	fmt.Printf("%v\n",len(ys))

	features := len(xs[0])

	n := NewMLP(features, []int{4, 4, 1})

	for k := 0; k < 50; k++ {

		// forward pass
		ypred := make([]*Value, len(ys))
		for i, x := range xs {
			
			ypred[i] = n.Forward(x)[0]
			// fmt.Printf("%v\n",ypred[i])
		}
		loss := MSE(ypred, ys)

		// backwards pass
		for _, p := range n.Parameters() {
			p.grad = 0
		}
		loss.Backward()

		// update weights
		for _, p := range n.Parameters() {
			p.data += -0.1 * p.grad
		}

		fmt.Printf("Iter: %2v, Loss: %v\n", k, loss.data)
	}
}