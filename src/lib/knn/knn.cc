#include <algorithm>
#include <unordered_map>
#include <vector>

extern "C" {

void KNNGraph(const float* src_points, const float* tar_points,
	int num_src, int num_tar, int k, float resolution, int* tar_indices) {
	//printf("Step1...\n");
	auto make_key1 = [&](float x) {
		return (int)(x / resolution);
	};
	int stride = 1024;
	auto make_key3 = [&](float x, float y, float z) {
		auto k1 = make_key1(x);
		auto k2 = make_key1(y);
		auto k3 = make_key1(z);
		return ((k1 * stride) + k2) * stride + k3;
	};

	//printf("Step2...\n");
	std::unordered_map<int, std::vector<int> > voxel_indices;
	for (int i = 0; i < num_src; ++i) {
		auto k = make_key3(src_points[i * 3],
			src_points[i * 3 + 1],
			src_points[i * 3 + 2]);
		auto it = voxel_indices.find(k);
		if (it != voxel_indices.end())
			it->second.push_back(i);
		else {
			std::vector<int> v;
			v.push_back(i);
			voxel_indices[k] = v;
		}
	}

	//printf("Step3...\n");
	for (int i = 0; i < num_tar; ++i) {
		std::vector<std::pair<double, int> > collection;
		for (int x = -1; x < 2; ++x) {
			for (int y = -1; y < 2; ++y) {
				for (int z = -1; z < 2; ++z) {
					auto k = make_key3(tar_points[i * 3] + x * resolution,
						tar_points[i * 3 + 1] + y * resolution,
						tar_points[i * 3 + 2] + z * resolution);
					auto it = voxel_indices.find(k);
					if (it != voxel_indices.end()) {
						for (auto& src_idx : it->second) {
							float dx = src_points[src_idx * 3] - tar_points[i * 3];
							float dy = src_points[src_idx * 3 + 1] - tar_points[i * 3 + 1];
							float dz = src_points[src_idx * 3 + 2] - tar_points[i * 3 + 2];
							collection.push_back(std::make_pair(dx * dx + dy * dy + dz * dz, src_idx));
						}
					}
				}
			}
		}
		if (collection.size() > k)
			std::nth_element(collection.begin(), collection.begin() + k, collection.end());
		for (int j = 0; j < k; ++j) {
			if (j < collection.size())
				tar_indices[i * k + j] = collection[j].second;
			else if (collection.size() == 0)
				tar_indices[i * k + j] = 0;
			else
				tar_indices[i * k + j] = collection[0].second;
		}
	}
	//printf("Step4...\n");
}

};