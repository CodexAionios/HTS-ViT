(* Dissemination operator prototype *)
ClearAll[disseminationStep]
disseminationStep[tensor_, hyperedges_, weight_:0.5] := Module[{X = tensor},
  Do[
    Module[{avg = Mean[X[[edge]]]},
      X[[edge]] = (1 - weight) X[[edge]] + weight avg
    ],
    {edge, hyperedges}
  ];
  X
]
