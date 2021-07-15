module main

import arrays

fn main() {
	a := arrays.max([1, 2])
	println(a.str())
	b := arrays.min([1, 2])
	println(b.str())
}
