#include "Roadmap.h"

// Inverts the inner node graph. Nodes become edges and edges become nodes.
Roadmap::Roadmap(SharedEnvironment* env,const std::string& fixed_roadmap_path){
    fixed_roadmap = load_fixed_roadmap(env, fixed_roadmap_path);
    graph = RoadmapGraph(env, fixed_roadmap);

    activated_roadmap = std::vector<std::vector<int>>(env->map.size(), std::vector<int>(4, 0));
}

void Roadmap::update_roadmap(SharedEnvironment* env, std::vector<int>& node_activation) {
    //Reset activated roadmap
    for (int i = 0; i < activated_roadmap.size(); i++) {
        for (int j = 0; j < activated_roadmap[i].size(); j++) {
            activated_roadmap[i][j] = 0;
        }
    }
    // Activate cost based on node_activation vector
    for (int node_id = 0; node_id < graph.nNodes; node_id++) {
        if (node_activation[node_id] == 1) {
            for (int loc : graph.node_to_cost_locs[node_id]) {
                int cost_dir = (graph.node_direction[node_id] + 2) % 4; // cost for step is applied in opposite direction of node direction
                activated_roadmap[loc][cost_dir] = 20;
            }
        }
    }   
    updated_last_step = true;         
}





// std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
// std::string fname = (source_dir / ".." / "roadmap" / "warehouse_8x6.roadmap").string();

std::vector<std::vector<int>> load_fixed_roadmap(SharedEnvironment* env,const std::string fname) {
	std::vector<std::vector<int>> fixed_roadmap(env->map.size(), std::vector(4,0));
    std::cout << "env Map size: " << env->map.size() << std::endl;

	
	
	std::ifstream myfile ((fname).c_str());
    if (!myfile.is_open())
    {
        cout << "Roadmap file " << fname << " does not exist. " << std::endl;
        exit(-1);
    }

	std::string line;
	// Skip the first lines
	std::getline(myfile, line);
	std::getline(myfile, line);
	std::getline(myfile, line);
	std::getline(myfile, line);
	
	// Read first map
	for (int row = 0; row < env->rows; row++) {
		std::getline(myfile, line);
		for (int col = 0; col < env->cols; col++) {
			char map_char = line[col];
			int loc_id = row * env->cols + col;
			
			if (map_char == '>'){
				fixed_roadmap[loc_id][2] = 20;
			}
			else if (map_char == '<'){
				fixed_roadmap[loc_id][0] = 20;
			}
		}
	}
	
	// Skip empty line between maps
	std::getline(myfile, line);
	
	// Read second map
	for (int row = 0; row < env->rows; row++) {
		std::getline(myfile, line);
		for (int col = 0; col < env->cols; col++) {
			char map_char = line[col];
			int loc_id = row * env->cols + col;
			
			if (map_char == '^'){
				fixed_roadmap[loc_id][1] = 20;
			}
			else if (map_char == 'v'){
				fixed_roadmap[loc_id][3] = 20;
			}
		}
	}

	std::cout << "Roadmap loaded." << std::endl;
    return fixed_roadmap;
}


 