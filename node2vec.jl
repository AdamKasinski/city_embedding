using Distributions
include("../OSMXGraph.jl/src/OSMXGraph.jl")
using .OSMXGraph
using OSMToolset
using Graphs

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
only for testing - OSMXGraph already returns sparse matrix  
"""
function generate_sparse_matrix(graph; is_weighted = false) 
    num_of_nodes = nv(graph)
    cols = Int[]
    rows = Int[]
    vals = Float64[]
    for node in 1:num_of_nodes
        nrbs = outneighbors(graph,node)
        n_nrbs = length(nrbs)
        append!(cols,fill(node,n_nrbs))
        append!(rows,nrbs)
        if !is_weighted
            append!(vals,ones(Float64,n_nrbs))
        end
    end
    return sparse(rows,cols,vals,num_of_nodes,num_of_nodes)
end

@inline function get_neighbors_info(sparse_matrix,node)
    ind = sparse_matrix.colptr[node]: sparse_matrix.colptr[node+1]-1
    return @view(sparse_matrix.rowval[ind]), @view(sparse_matrix.nzval[ind])
end


function generate_first_move(start_node,sparse_matrix)
    neighbs, probs = get_neighbors_info(sparse_matrix, start_node)
    isempty(neighbs) && return nothing    
    return sample(neighbs,Weights(probs))
end


function generate_biased_move(curr_node, prev_node, sparse_matrix, p, q)
    neighbs, probs = get_neighbors_info(sparse_matrix, curr_node)
    isempty(neighbs) && return nothing
    
    p_q_probs = Vector{Float64}(undef, length(probs))
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

function generate_path(start_node,sparse_matrix,path_length,p,q)
    path = zeros(Int64,path_length)
    path[1] = start_node
    curr_node = start_node
    next_node = generate_first_move(curr_node,sparse_matrix)
    next_node === nothing && return path
    path[2] = next_node
    prev_node, curr_node = curr_node, next_node
    for indx in 3:path_length
        next_node = generate_biased_move(curr_node,prev_node,sparse_matrix,p,q)
        next_node === nothing && break
        path[indx] = next_node
        prev_node, curr_node = curr_node, next_node
    end
    return path
end

function generate_corpus(n_path,path_length,sparse_matrix,p,q)
    corpus = zeros(Int64,(n_path,path_length))
    N = size(sparse_matrix,1)
    for path in 1:n_path
        start_node = rand(1:N)
        corpus[path,:] = generate_path(start_node, sparse_matrix, path_length, p, q)
    end
    return corpus
end

