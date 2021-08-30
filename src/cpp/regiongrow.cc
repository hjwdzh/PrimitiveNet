#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>

extern "C" {

void RegionGrowing(int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask, float score_thres) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[j * numF + i];
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;
		}
	}

	std::vector<int> mask(numV, -2);
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		int is_boundary = 0;
		for (auto& info : v_neighbors[i]) {
			if (info.second == 1) {
				is_boundary += 1;
			}
		}
		if (is_boundary >= v_neighbors[i].size() * score_thres) {
			mask[i] = -1;
			num_boundary += 1;
		}
	}

	if (output_mask) {
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;
			b += output_mask[i];
		}
	}

	int num_labels = 0;
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
				}
			}
			num_labels += 1;
		}
	}

	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}
}

int GetParent(std::vector<int>& parent, int j) {
	if (j == parent[j])
		return j;
	parent[j] = GetParent(parent, parent[j]);
	return parent[j];
}

void RegionGrowingMerge(float* boundary, int* F, int numV, int numF, int* face_labels, int* mask, float score_thres) {
	std::map<std::pair<int, int>, std::vector<int> > edge_to_faces;
	std::map<std::pair<int, int>, float> edge_scores;
	std::vector<float> face_scores(numF, 0);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];
			if (v0 > v1)
				std::swap(v0, v1);
			float score = boundary[j * numF + i];
			auto k = std::make_pair(v0, v1);
			edge_scores[k] = score;
			if (edge_to_faces.count(k)) {
				edge_to_faces[k].push_back(i);
			} else {
				std::vector<int> faces;
				faces.push_back(i);
				edge_to_faces[k] = faces;
			}
			face_scores[i] += score;
		}
	}

	std::vector<int> parents(numF), visited(numF, 0);
	for (int i = 0; i < parents.size(); ++i) {
		parents[i] = i;
	}

	std::map<std::pair<int, int>, float > face_edge_scores;
	for (auto& p : edge_to_faces) {
		auto e = p.first;
		float k = edge_scores[e] * 2;
		for (auto& f1 : p.second) {
			for (auto& f2 : p.second) {
				if (f1 < f2) {
					auto k = std::make_pair(f1, f2);
					float edge = edge_scores[e];
					float s = face_scores[f1] + face_scores[f2] - edge * 2;
					face_edge_scores[k] = s;
				}
			}
		}
	}
	for (float thres = 0; thres <= score_thres; thres += 0.1) {
		bool update = true;
		while (update) {
			update = false;
			std::map<std::pair<int, int>, std::pair<float, int> > merged_face_edge_scores;
			std::vector<std::pair<int, int> > removed_keys;
			for (auto& p : face_edge_scores) {
				auto e = p.first;
				int v1 = GetParent(parents, e.first);
				int v2 = GetParent(parents, e.second);
				if (v1 == v2) {
					removed_keys.push_back(e);
					continue;
				}
				if (v1 > v2) {
					std::swap(v1, v2);
				}
				auto it = merged_face_edge_scores.find(std::make_pair(v1, v2));
				if (it != merged_face_edge_scores.end()) {
					it->second.first += p.second;
					it->second.second += 1;
				} else {
					merged_face_edge_scores[std::make_pair(v1, v2)] = std::make_pair(p.second, 1);
				}
			}
			for (auto& k : removed_keys)
				face_edge_scores.erase(k);

			memset(visited.data(), 0, sizeof(int) * numF);
			std::vector<std::pair<float, std::pair<int, int> > > edge_set;
			for (auto& info : merged_face_edge_scores) {
				float s = info.second.first / info.second.second;
				edge_set.push_back(std::make_pair(s, info.first));
			}
			std::sort(edge_set.begin(), edge_set.end());
			for (auto& e : edge_set) {
				if (e.first > thres)
					break;
				int v1 = e.second.first;
				int v2 = e.second.second;
				if (visited[v1] || visited[v2]) {
					continue;
				}
				parents[v2] = v1;
				visited[v1] = 1;
				visited[v2] = 1;
				update = true;
			}
		}
	}
	int numgroup = 0;
	for (auto& v : visited)
		v = -1;
	for (int i = 0; i < visited.size(); ++i) {
		int p = GetParent(parents, i);
		if (visited[p] == -1) {
			visited[p] = numgroup++;
		}
		face_labels[i] = visited[p];
	}

	std::vector<int> group_count(numgroup, 0);
	for (int i = 0; i < visited.size(); ++i) {
		group_count[face_labels[i]] += 1;
	}
	std::vector<int> group_remap(numgroup, 0);
	numgroup = 0;
	for (int i = 0; i < group_remap.size(); ++i) {
		if (group_count[i] < 5) {
			group_remap[i] = -100;
		} else {
			group_remap[i] = numgroup++;
		}
	}
	for (int i = 0; i < visited.size(); ++i) {
		face_labels[i] = group_remap[face_labels[i]];
	}
}

void BoundaryRegionGrowing(int* boundary, int* F, int numV, int numF, int* face_labels, int* output_mask) {
	std::vector<std::unordered_map<int, int> > v_neighbors(numV);
	for (int i = 0; i < numF; ++i) {
		for (int j = 0; j < 3; ++j) {
			int b = boundary[j * numF + i];
			int v0 = F[i * 3 + j];
			int v1 = F[i * 3 + (j + 1) % 3];
			v_neighbors[v0][v1] = b;
			v_neighbors[v1][v0] = b;
		}
	}

	std::vector<int> mask(numV, -2);
	int num_boundary = 0;
	for (int i = 0; i < numV; ++i) {
		if (boundary[i]) {
			num_boundary += 1;
			mask[i] = -1;
		}
	}
	printf("Num boundary %d of %d\n", num_boundary, numV);

	if (output_mask) {
		int b = 0;
		for (int i = 0; i < numV; ++i) {
			output_mask[i] = (mask[i] == -1) ? 1 : 0;
			b += output_mask[i];
		}
		printf("Num boundary %d\n", b);
	}

	int num_labels = 0;
	for (int i = 0; i < mask.size(); ++i) {
		if (mask[i] == -2) {
			std::queue<int> q;
			q.push(i);
			mask[i] = num_labels;
			while (!q.empty()) {
				int v = q.front();
				q.pop();
				for (auto& nv : v_neighbors[v]) {
					if (nv.second == 0 && mask[nv.first] == -2) {
						mask[nv.first] = num_labels;
						q.push(nv.first);
					}
				}
			}
			num_labels += 1;
		}
	}

	printf("Num label %d\n", num_labels);
	for (int i = 0; i < numF; ++i) {
		int label = -1;
		for (int j = 0; j < 3; ++j) {
			if (mask[F[i * 3 + j]] >= 0) {
				label = mask[F[i * 3 + j]];
				break;
			}
		}
		face_labels[i] = label;
	}
}
};
