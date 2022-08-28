#pragma once

#include <vector>
#include <array>
#include <thread>
#include <mutex>

#include "imgui.h"

#include "Walnut/Layer.h"
#include "Walnut/Image.h"

#include "nn.h"


class MlTool : public Walnut::Layer, public NeuralNetwork {
public:
	MlTool() : NeuralNetwork({2, 3, 1}) {}
	~MlTool() = default;


	virtual void OnUIRender() override;
	inline void invokeMenuPresence() {
		if (ImGui::BeginMenu("Tools")) {
			ImGui::MenuItem("MlModel", NULL, &this->s_tool_enable);
			ImGui::EndMenu();
		}
	}

protected:
	static void trainThread();

private:
	std::vector<float> progress;
	std::vector<size_t> topo_edit{this->topology};
	
	int
		activ_f_idx{ 0 },
		layers_drag{(int)this->topology.size()};
	bool
		s_tool_enable{ true },
		s_regen_avail{ false },
		s_show_weights{ false };

	inline static constexpr char* func_names[]{
		"Sigmoid", "Hyperbolic Tangent", "ReLU"
	};


};