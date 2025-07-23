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


function non_first_move_table()
    pass
end

function generate_move()
    pass
end

function generate_one_path()
    pass
end

function generate_n_paths()
    pass
end

