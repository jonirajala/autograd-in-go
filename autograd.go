package main

import (
	"fmt"
	"math"
	"math/rand"

)

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

}

type Layer struct {
	neurons []*Neuron
}

type MLP struct {
	layers []*Neuron
}


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


func Pow(a *Value, b float64) *Value {
	out := &Value{
		data: math.Pow(a.data, b),
		prev: []*Value{a,},
		op: "**",
	}
	out.backward = func() {
		a.grad += (b* math.Pow(a.data, (b-1))) * out.grad
	}
	return out
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


func NewNeuron(size int) *Neuron {
	w := make([]*Value, size)
	for i := 0; i < size; i++ {
		w[i] = New(2*rand.Float64() - 1)
	}
	b := New(2*rand.Float64() - 1)

	n := &Neuron{
		w: w,
		b: b,
	}
	return n
}

func (n *Neuron) Forward(x []*Value) *Value {
	out := n.b
	for i := 0; i < len(x); i++ {
		out = Add(out, Mul(n.w[i], x[i]))
	}
	out = ReLU(out)
	
	return out
}

func NewLayer(in, out int) *Layer {
	neurons := make([]*Neuron, out)
	for i := 0; i < out; i++ {
		neurons[i] = NewNeuron(in)
	}
	layer := &Layer{neurons: neurons}
	return layer
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.neurons))

	for i := 0; i < len(x); i++ {
		out[i] = l.neurons[i].Forward(x)
	}
	return out
}

func NewMLP(in, out int) *Layer {
	neurons := make([]*Neuron, out)
	for i := 0; i < out; i++ {
		neurons[i] = NewNeuron(in)
	}
	layer := &Layer{neurons: neurons}
	return layer
}


func main() {
	x := New(2)
	w := New(0.4) // pretend random init
	y := New(4)

	for k := 0; k < 6; k++ {

		// forward pass
		ypred := Mul(w, x)
		loss := Pow(Sub(ypred, y), 2)

		// backward pass
		w.grad = 0 // zero previous gradients
		loss.Backward()

		// update weights
		w.data += -0.1 * w.grad

		fmt.Printf("Iter: %2v, Loss: %.4v, w: %.4v\n",
            k, loss.data, w.data)
	}
}