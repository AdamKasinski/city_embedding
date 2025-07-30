### Project goal
This repository delivers a Julia implementation of the node2vec algorithm. It handles every step of the pipeline except the final skip‑gram optimiser, 
which is delegated to Word2Vec.jl so that we can rely on Google’s tested C binary for fast training.

### High‑level workflow  

1. **Build the graph**  
   Convert an OpenStreetMap extract into a sparse adjacency matrix via  *`OSMXGraph.create_road_graph`* or the fallback  *`generate_sparse_matrix`*.  


```julia
using CSV, Word2Vec
include("../OSMXGraph.jl/src/OSMXGraph.jl")
include("node2vec.jl")
using .OSMXGraph
```

    WARNING: replacing module OSMXGraph.



```julia
dir_in = "data"
dir_out = "graph_data"
road_types = ["motorway", "trunk", "primary", "secondary", 
            "tertiary", "residential", "service", "living_street", 
            "motorway_link", "trunk_link", "primary_link", "secondary_link", 
            "tertiary_link"]
```


    13-element Vector{String}:
     "motorway"
     "trunk"
     "primary"
     "secondary"
     "tertiary"
     "residential"
     "service"
     "living_street"
     "motorway_link"
     "trunk_link"
     "primary_link"
     "secondary_link"
     "tertiary_link"



```julia
df, sparse_index, road_index, node_ids = OSMXGraph.create_road_graph("Ochota.osm", road_types,"Ochota_graph.csv","Ochota_nodes.json",dir_in=dir_in,dir_out=dir_out)
```


    ([1m16041×9 DataFrame[0m
    [1m   Row [0m│[1m id    [0m[1m from_id [0m[1m to_id [0m[1m from        [0m[1m to          [0m[1m from_LLA            [0m ⋯
           │[90m Int64 [0m[90m Int64   [0m[90m Int64 [0m[90m Int64       [0m[90m Int64       [0m[90m String              [0m ⋯
    ───────┼────────────────────────────────────────────────────────────────────────
         1 │     1     1984   1983   1949648310   9265228166  OpenStreetMapX.LLA(5 ⋯
         2 │     2     1983   1984   9265228166   1949648310  OpenStreetMapX.LLA(5
         3 │     3     6218   3505  10061787376   4737816385  OpenStreetMapX.LLA(5
         4 │     4     3505   6218   4737816385  10061787376  OpenStreetMapX.LLA(5
         5 │     5     3507   6218   4737816383  10061787376  OpenStreetMapX.LLA(5 ⋯
         6 │     6     6218   3507  10061787376   4737816383  OpenStreetMapX.LLA(5
         7 │     7     3506   3507   4737816384   4737816383  OpenStreetMapX.LLA(5
         8 │     8     3507   3506   4737816383   4737816384  OpenStreetMapX.LLA(5
       ⋮   │   ⋮       ⋮       ⋮         ⋮            ⋮                       ⋮    ⋱
     16035 │ 16035     8037   8036  12467212838  12467212836  OpenStreetMapX.LLA(5 ⋯
     16036 │ 16036     1018   3578    302913213    302913215  OpenStreetMapX.LLA(5
     16037 │ 16037     3578   1018    302913215    302913213  OpenStreetMapX.LLA(5
     16038 │ 16038     5558   5560   9264715225   9264715235  OpenStreetMapX.LLA(5
     16039 │ 16039     5560   5558   9264715235   9264715225  OpenStreetMapX.LLA(5 ⋯
     16040 │ 16040     5561   5558   9264715241   9264715225  OpenStreetMapX.LLA(5
     16041 │ 16041     5558   5561   9264715225   9264715241  OpenStreetMapX.LLA(5
    [36m                                                4 columns and 16026 rows omitted[0m, sparse([2, 58, 1, 4, 4431, 6446, 3, 670, 1959, 1806  …  3769, 3814, 3758, 3985, 3988, 6790, 1137, 1139, 5274, 5177], [1, 1, 2, 3, 3, 3, 4, 4, 4, 5  …  8103, 8103, 8104, 8104, 8104, 8106, 8107, 8107, 8107, 8108], [5111, 3569, 5112, 15462, 12638, 5878, 15463, 3941, 5330, 4487  …  6954, 10511, 15675, 2054, 15674, 14058, 2687, 3217, 3216, 9447], 8108, 8108), NearestNeighbors.KDTree{StaticArraysCore.SVector{2, Float64}, Distances.Euclidean, Float64, StaticArraysCore.SVector{2, Float64}}
      Number of points: 8108
      Dimensions: 2
      Metric: Distances.Euclidean(0.0)
      Reordered: false, [3956, 5157, 4846, 900, 4640, 6039, 3725, 7009, 2937, 2608  …  1506, 654, 5792, 204, 2596, 569, 1375, 3283, 6348, 5426])


2. **Generate node2vec walks**  
   Create biased random walks with the \(p,q\) scheme using  *`generate_corpus`*.

   The walk generator is implemented with 4 functions:

* `get_neighbors_info`: Returns *views* over the target IDs and edge weights for one column of the sparse adjacency matrix.

* `generate_first_move`: Samples the **first hop** from the start node (unbiased).

* `generate_biased_move`: Samples every **subsequent hop** with the node2vec bias:<br> • 1 / *p* if the candidate node is the previous vertex (return)<br> • weight      if the candidate is a neighbour of the previous vertex (stay close)<br> • 1 / *q* otherwise (explore further). 
* `generate_path`: Combines the two helpers above to build one complete walk of length *ℓ*. 

* `generate_corpus(n_paths, ℓ, A, p, q)`**, does the following for each of the *n_paths* walks:

   1. **Choose a random start node** (`rand(1:nv(A))`).  
   2. **First hop** → `next = generate_first_move(start, A)`  
      *If the start node is a sink, the walk ends immediately.*  
   3. **Remaining hops** (`ℓ‑2` iterations)  
      ```julia
      next = generate_biased_move(curr, prev, A, p, q)
      ```  
      *Stops early if the current node has no out‑neighbours.*  
   4. **Store the walk** as one row in an `(n_paths × ℓ)` integer matrix, padding with zeros when a walk ended early.



```julia
corp = generate_corpus(10,5,sparse_index,0.3,0.4)
```


    10×5 Matrix{Int64}:
     6547   947  6544   947  6544
     5587  5586  5587  5586  1006
     2959  2830  2959  2960  2959
     6678   485  7587  1839  3312
     1156  3294  3252  3246  3252
     8047  8049   985  5924   984
     1468  5935  5940  5938  5940
     5186  7726  7724  7726  7724
     2354    14  2354  2783  2354
     5389  5387  5389  5387  5389


3. **Train skip‑gram embeddings**  
   Feed the walk corpus to `Word2Vec.word2vec` and let the C backend learn the vectors.  



```julia
open("walks.txt", "w") do io
    for walk in eachrow(corp)
        println(io, join(string.(walk), ' '))
    end
end

model = word2vec("walks.txt","out.bin", cbow = 0)
```

    Starting training using file walks.txt
    Vocab size: 1
    Words in train file: 10



    Process(`[4m/home/adamkas/.julia/artifacts/0c86f7feb8f6b4ab5f9fb793f1fde1278e3a6021/bin/word2vec[24m [4m-train[24m [4mwalks.txt[24m [4m-output[24m [4mout.bin[24m [4m-size[24m [4m100[24m [4m-window[24m [4m5[24m [4m-sample[24m [4m0.001[24m [4m-hs[24m [4m0[24m [4m-negative[24m [4m5[24m [4m-threads[24m [4m12[24m [4m-iter[24m [4m5[24m [4m-min-count[24m [4m5[24m [4m-alpha[24m [4m0.025[24m [4m-debug[24m [4m2[24m [4m-binary[24m [4m0[24m [4m-cbow[24m [4m0[24m`, ProcessExited(0))


### Aknowledgments

This research was funded by National Science Centre, Poland, grant number 2021/41/B/HS4/03349
The module builds upon two different packages for manipulating OpenStreetMap data: OpenStreetMapX.jl (https://github.com/pszufe/OpenStreetMapX.jl) and OSMXGraph.jl (https://github.com/AdamKasinski/OSMXGraph.jl). 
