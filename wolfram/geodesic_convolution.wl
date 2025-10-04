(* Geodesic convolution sketch in WL *)
latentFeatures[data_] := Normalize /@ data;
geodesicDistance[Ti_, Tj_, r_:1] := r ArcCos[Clip[Ti.Tj, {-1,1}]];
adaptiveKernel[d_, sigma_:0.8] := Exp[-(d^2)/(2 sigma^2)];

gConvolve[tensorField_, sigma_:0.8] := Module[{X, K},
  X = latentFeatures[tensorField];
  K = Table[ adaptiveKernel[ geodesicDistance[X[[i]], X[[j]]], sigma], {i, Length[X]}, {j, Length[X]} ];
  K = K/Total[K, {2}];
  K.tensorField
]
