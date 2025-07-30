### node2vec
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
sparse_index
```


    8108×8108 SparseMatrixCSC{Int64, Int64} with 15819 stored entries:
    ⎡⣻⣮⣿⣶⢢⡖⠧⣻⡧⣿⢾⢯⣛⣦⡷⡾⠕⣵⡿⡾⠈⡮⣹⣖⡲⢯⠤⢸⢺⢆⣂⡇⠀⠄⠰⢃⡦⢠⡨⡣⎤
    ⎢⣺⣷⣽⣿⡣⡗⡞⢬⣗⣿⣳⠿⢧⣟⡟⢛⣀⣟⣷⢔⣞⠿⣟⡒⢛⣗⣒⡼⠻⡅⢺⣷⡒⠽⣉⡎⢹⣀⢐⣟⎥
    ⎢⢨⠝⢭⠮⢟⢕⢿⣳⣯⣽⢞⣺⠄⡟⢥⣼⣭⣧⣲⡝⢗⣯⠟⡪⣦⣼⢽⠬⡌⢗⢏⢝⣗⡀⠘⢷⢼⣿⣒⣯⎥
    ⎢⣬⣯⡛⣍⠿⣺⢻⣶⡫⠆⣩⡸⣠⡽⠄⢑⢻⢗⣇⣽⡗⡹⢹⢛⡙⣺⡙⢱⣞⠎⡳⣟⢁⠌⡤⠐⢓⢢⠠⠐⎥
    ⎢⣍⡯⢽⣵⣇⡿⡫⠮⣿⣿⡇⣁⡨⣝⣤⣰⡹⡋⠙⢷⣼⢳⢋⡿⠹⢙⡷⣍⣩⠇⣚⡖⢾⣁⡠⢰⣘⠧⠹⣍⎥
    ⎢⡽⣷⣿⡎⣲⣹⣏⡺⡍⢽⢻⣿⣪⣫⡍⡭⡭⡬⢴⣍⠑⠬⣮⠂⠁⠧⠁⡔⡱⠭⣚⣶⡥⠈⠦⠑⣢⢁⢄⣅⎥
    ⎢⡷⣼⣯⢷⣤⠥⣴⡖⣆⣄⢿⣼⠿⣧⡣⡴⣤⣄⡴⢦⡄⠢⣤⣄⠈⣒⣄⢌⣐⣦⠧⣻⠌⠀⣄⢛⢦⣰⠀⠃⎥
    ⎢⡭⣭⣿⢩⣑⣷⢄⢱⣁⣻⡇⠭⢉⣮⡿⣯⣪⢍⠢⣍⣹⣆⢳⡏⢬⣀⡅⠭⣅⡁⢠⡈⡁⠖⠦⣉⢌⣘⠠⡍⎥
    ⎢⢇⣥⣤⢼⠧⣿⣾⠖⡷⠫⡃⡯⠀⢿⡮⢺⣿⣯⡝⠟⣀⡜⠿⡕⣾⠂⠱⢂⢽⢕⡸⠷⠒⣄⢄⠂⣲⡧⠠⣆⎥
    ⎢⣿⡷⢙⢷⣌⢾⣯⣷⢷⣄⡔⢷⠰⣇⡌⢦⣷⠝⣻⣾⣾⣮⣼⡞⠾⣉⣂⠖⣀⣌⣽⣇⡀⡖⡤⣈⡳⡻⠲⢝⎥
    ⎢⣀⠦⣾⡗⣽⣵⡝⡭⢶⣉⡑⡄⠠⠉⠶⣾⣀⢜⡺⡿⠻⣾⠾⡒⢼⢏⢼⣬⣞⡧⠒⠖⢠⡔⠁⡈⣄⡞⠪⡆⎥
    ⎢⢸⢶⢯⠹⠹⡡⢷⢒⣭⣴⠡⠛⠀⢿⡹⠾⢟⠧⣲⠿⢺⠣⢳⣶⠠⣡⠐⡠⢆⢠⡠⠳⠥⠈⢝⣳⣢⠓⠤⡄⎥
    ⎢⡬⣎⢽⡼⣈⣿⢰⣨⣗⣒⠥⣄⢢⢠⢀⠻⠺⠛⢞⣣⡒⢗⠤⣂⢻⣶⡸⢐⢛⢖⢋⠨⢅⡀⠊⢟⣬⢬⠀⢤⎥
    ⎢⡀⢃⣸⣼⡓⡕⢗⣈⡝⣯⣁⠬⡀⣝⡅⡍⠑⣂⢨⠜⡒⣗⠠⡠⠒⢊⠿⣧⡀⡀⣠⡇⠄⠀⡰⣅⢆⡀⢈⠈⎥
    ⎢⠞⢕⠗⢗⢦⢍⡾⠔⠣⠞⡔⡎⠰⡽⠥⡸⢓⢗⠊⢼⠾⡭⠘⣑⢻⢔⠁⠨⠿⣿⣚⠎⣐⠣⠤⢪⢼⠅⠰⠄⎥
    ⎢⣦⠚⢾⣆⣏⢕⣿⢮⢺⠘⢞⣬⣭⣣⡀⠲⢶⡎⠳⢿⢸⠄⢤⡊⡛⡰⠤⠾⡾⠘⡿⢏⡀⡁⢄⣘⢝⢝⠸⠴⎥
    ⎢⢠⠄⣼⡌⠙⠽⡁⠔⠞⡳⡁⠃⠂⠁⢡⡌⠚⢤⢠⠬⢀⠶⠁⠃⠁⠱⠀⠁⠴⡘⠀⠨⠻⣦⠀⠱⠐⠂⠀⠠⎥
    ⎢⠼⠄⠧⠼⣶⣆⢀⠋⢀⢋⠈⢃⣦⢉⡈⢣⡠⠑⡐⢫⡁⠠⠷⣱⣮⢄⠔⢦⡠⢃⣀⢰⢄⡀⡻⢎⣖⢪⡀⡄⎥
    ⎢⠀⡣⠳⢒⣶⣭⠹⡐⢲⡘⠌⢚⢈⣳⣂⢱⠼⠶⣽⡺⣠⠽⢬⡚⡂⣟⡘⠑⠾⠗⣗⢍⠰⠀⡸⣙⢿⣷⣄⣂⎥
    ⎣⠦⡾⣴⢴⡼⣼⢀⠂⡗⣂⠆⢵⠤⠀⡄⠦⠠⢦⣜⢖⠺⠦⠀⠧⠀⣄⡂⠐⠐⠄⢒⡄⠀⡀⠀⠬⠠⢹⠻⣶⎦


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
      next = generate_biased_move(curr, prev, A, p, q)
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
""
```

    Starting training using file walks.txt
    Vocab size: 1
    Words in train file: 10



    ""


### Aknowledgments

This research was funded by National Science Centre, Poland, grant number 2021/41/B/HS4/03349
The module builds upon two different packages for manipulating OpenStreetMap data: OpenStreetMapX.jl (https://github.com/pszufe/OpenStreetMapX.jl) and OSMXGraph.jl (https://github.com/AdamKasinski/OSMXGraph.jl). 
