(* Tensorfield propagation sketch *)
ClearAll[propagateTensorfield]
propagateTensorfield[tensor_, hypergraph_, steps_:3] := Module[{T = tensor},
  Do[
    T = gConvolve[T, 0.8];
    T = disseminationStep[T, hypergraph, 0.4];
  , {steps}];
  T
]
