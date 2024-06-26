package main

import (
	"fmt"
	"math"
	"math/rand"

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
		w[i] = New(2*rand.Float64() - 1)
	}
	b := New(2*rand.Float64() - 1)

	n := &Neuron{
		w: w,
		b: b,
		nonlin: nonlin,
	}
	return n
}

func (n *Neuron) Forward(x []*Value) *Value {
	out := n.b
	for i := 0; i < len(x); i++ {
		// fmt.Printf("%v\n",i)
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


func main() {
	n := NewMLP(3, []int{4, 4, 1})

    xs := [][]*Value{
        {New(2), New(3), New(-1)},
        {New(3), New(-1), New(0.5)},
        {New(0.5), New(1), New(1)},
        {New(1), New(1), New(-1)},
        {New(1.5), New(2), New(-0.5)},
        {New(2.5), New(0.5), New(-1.5)},
        {New(-0.5), New(-1), New(2)},
        {New(-1.5), New(1.5), New(0.5)},
        {New(3), New(-0.5), New(1)},
        {New(-2), New(2), New(0.5)},
        {New(2), New(-2), New(-0.5)},
        {New(1.2), New(2.1), New(0.3)},
        {New(2.7), New(-1.3), New(1.2)},
        {New(-1.2), New(0.5), New(1.7)},
        {New(0.1), New(-1.5), New(2.3)},
        {New(1.8), New(1.2), New(-0.7)},
        {New(-1.3), New(-0.8), New(2.1)},
        {New(0.4), New(1.5), New(-1.8)},
        {New(2.2), New(1.8), New(0.6)},
        {New(-0.9), New(-1.7), New(2.4)},
    }

    // Define the output dataset ys
    ys := []*Value{
        New(1), New(-1), New(-1), New(1),
        New(1), New(-1), New(-1), New(1),
        New(-1), New(1), New(-1), New(1),
        New(1), New(-1), New(1), New(-1),
        New(-1), New(1), New(-1),New(-1),
    }

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