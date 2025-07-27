using Distributions
include("../OSMXGraph.jl/src/OSMXGraph.jl")
using .OSMXGraph
using OSMToolset
using Graphs
using SparseArrays
using Random
using StatsBase

dir_in = "data"
road_types = ["motorway", "trunk", "primary", "secondary", 
            "tertiary", "residential", "service", "living_street", 
            "motorway_link", "trunk_link", "primary_link", "secondary_link", 
            "tertiary_link"]

df, sparse_index, road_index, node_ids = OSMXGraph.create_road_graph("Ochota.osm", road_types,"Ochota_graph.csv","Ochota_nodes.json",dir_in=dir_in)
POI_df = OSMToolset.find_poi(string(dir_in,"/","Ochota.osm"))
POI_xs = POI_df.lat
POI_ys = POI_df.lon
poi_with_nearest_points = OSMXGraph.add_nearest_road_point(POI_df,POI_xs, POI_ys, road_index, node_ids)
OSMXGraph.save_file(poi_with_nearest_points,"poi_with_nearest_points.csv")


"""

generate_sparse_matrix(graph; is_weighted = false) -> SparseMatrixCSC{Float64,Int}

Create the adjacency matrix of `graph` as a column-compressed sparse matrix.

* `graph` must be any subtype of `Graphs.AbstractGraph`.
* If `is_weighted == false` (default) every existing edge gets weight 1.0.
  When `true` you are expected to extend the inner loop to read the weights
  that your graph type stores.

only for testing - OSMXGraph already returns sparse matrix  
"""
function generate_sparse_matrix(graph::Graphs.AbstractGraph)
    num_of_nodes::Int = nv(graph)                       

    cols::Vector{Int} = Int[]                       
    rows::Vector{Int} = Int[]                       
    vals::Vector{Float64} = Float64[]                   

    for node::Int in 1:num_of_nodes
        nbrs::Vector{Int} = outneighbors(graph, node)   
        n_nbrs::Int = length(nbrs)
        append!(cols, fill(node, n_nbrs))               
        append!(rows, nbrs)                             
        append!(vals, ones(Float64, n_nbrs))
        return sparse(rows,cols,vals,num_of_nodes,num_of_nodes)
    end
end



"""
    get_neighbors_info(sparse_matrix, node) -> (rows_view, vals_view)

Return views over the non-zero entries that constitute the **out-neighbours
of one node in a sparse adjacency matrix.

* `sparse_matrix`   – `SparseMatrixCSC{Float64,Int}`
* `node`– `Int` in the range `1:nv`.

`rows_view` represents nodes  
`vals_view` represents weight
"""
@inline function get_neighbors_info(sparse_matrix::SparseMatrixCSC{Tv,Ti},node::Int) where {Tv,Ti}
    ind = sparse_matrix.colptr[node]: sparse_matrix.colptr[node+1]-1
    return @view(sparse_matrix.rowval[ind]), @view(sparse_matrix.nzval[ind])
end

"""
Randomly pick one out-neighbour of `start_node` from a sparse adjacency
matrix `sparse_matrix`, using the edge weights in column `start_node` as probabilities.

Arguments  
* `start_node::Int` – 1-based vertex ID.  
* `sparse_matrix::SparseMatrixCSC{Tv,Ti}` – adjacency / weight matrix.

Returns `Ti` (chosen neighbour) or `nothing` if the node has no outgoing
edges.
"""
function generate_first_move(start_node::Int,sparse_matrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    neighbs, probs = get_neighbors_info(sparse_matrix, start_node)
    isempty(neighbs) && return nothing    
    return sample(neighbs,Weights(probs))
end

"""
    generate_biased_move(curr_node, prev_node, sparse_matrix, p, q)

Return the next vertex in a node2vec walk, applying the p, q bias.

Arguments  
* `curr_node::Int` – current vertex.  
* `prev_node::Int` – previous vertex.  
* `sparse_matrix::SparseMatrixCSC{Tv,Ti}` – adjacency / weight matrix.  
* `p::Real`, `q::Real` – return-parameter p and in-out-parameter q.

The chosen neighbour is sampled with probability proportional to these scaled
weights. If `curr_node` is a sink, the function returns `nothing`.

Returns `Union{Ti,Nothing}` – next vertex id or `nothing`.
"""
function generate_biased_move(curr_node::Int, prev_node::Int, sparse_matrix::SparseMatrixCSC{Tv,Ti}, p::Float64, q::Float64) where {Tv,Ti}
    neighbs, probs = get_neighbors_info(sparse_matrix, curr_node)
    isempty(neighbs) && return nothing
    p_q_probs::Vector{Float64} = Vector{Float64}(undef, length(probs))
    prev_node_neighs = Set(get_neighbors_info(sparse_matrix,prev_node)[1])
    @inbounds for idx in eachindex(neighbs)
        dst = neighbs[idx]
        pr = probs[idx]
        if dst == prev_node
            p_q_probs[idx] = pr/p
        elseif dst in prev_node_neighs
            p_q_probs[idx] = pr  
        else
            p_q_probs[idx] = pr/q
        end
    end
    return sample(neighbs,Weights(p_q_probs))
end

"""
generate_path(start_node, A, path_length, p, q) -> Vector{Ti}

Perform one **node2vec random walk** starting at `start_node`.

Arguments  
* `start_node::Int` - 1-based vertex ID where the walk begins.
* `sparse_matrix::SparseMatrixCSC{Tv,Ti}` - Adjacency/weight matrix; `(i,j)`.
* `path_length::Int` - Maximum number of vertices to record. 
* `p, q::Float64` - Node2vec return-parameter *p* and in-out parameter *q*.

Algorithm  
1. Draw the first hop with `generate_first_move` (unbiased).  
2. For each subsequent hop call `generate_biased_move`, which scales edge
   weights by 1 / *p* or 1 / *q* according to the node2vec rules.  
3. Stop early if the current vertex has no out-neighbours.

Return value  
`Vector{Ti}` of length `path_length`, padded with zeros if the walk ends
prematurely. `Ti` is the index type used by the sparse matrix.
"""
function generate_path(start_node::Int,sparse_matrix::SparseMatrixCSC{Tv,Ti},path_length::Int,p::Float64,q::Float64) where {Tv,Ti}
    path::Vector{Ti} = fill(zero(Ti), path_length)
    path[1] = start_node
    curr_node::Ti = start_node
    next_node = generate_first_move(curr_node,sparse_matrix)
    next_node === nothing && return path
    path[2] = next_node
    prev_node::Ti, curr_node = curr_node, next_node
    for indx in 3:path_length
        next_node = generate_biased_move(curr_node,prev_node,sparse_matrix,p,q)
        next_node === nothing && break
        path[indx] = next_node
        prev_node, curr_node = curr_node, next_node
    end
    return path
end


"""
    generate_corpus(n_path, path_length, A, p, q) -> Matrix{Ti}

Create a collection of **node2vec walks**.

rguments  
* `start_node::Int` - 1-based vertex ID where the walk begins.
* `sparse_matrix::SparseMatrixCSC{Tv,Ti}` - Adjacency/weight matrix; `(i,j)`.
* `path_length::Int` - Maximum number of vertices to record. 
* `p, q::Float64` - Node2vec return-parameter *p* and in-out parameter *q*.


# Returns
`Matrix{Ti}` of size `(n_path, path_length)` where each row is one walk.  
`Ti` is the index type of `A` (`Int` on 64-bit Julia).
"""
function generate_corpus(n_path::Int,path_length::Int,sparse_matrix::SparseMatrixCSC{Tv,Ti},p::Float64,q::Float64) where {Tv,Ti}
    corpus::Matrix{Ti} = fill(zero(Ti),(n_path,path_length))
    N::Int = size(sparse_matrix,1)
    for path in 1:n_path
        start_node = rand(1:N)
        corpus[path,:] = generate_path(start_node, sparse_matrix, path_length, p, q)
    end
    return corpus
end

