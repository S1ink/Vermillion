#pragma once

#include <vector>
#include <array>
#include <thread>
#include <mutex>

#include "imgui.h"
#include "implot/implot.h"

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
	static void trainThread(MlTool*);

private:
	std::vector<float> epochs_avg, epochs_high, epochs_low, epoch_last;
	std::vector<size_t> topo_edit{this->topology};
	IOFunc rfunc;
	DataSet dataset;
	std::thread trainer;
	std::mutex train_mutex;
	
	ImPlotColormap primary_theme{ImPlotColormap_Cool}, secondary_theme{ImPlotColormap_Hot};
	int
		activ_f_idx{ 0 },
		section_idx{ 0 },
		layers_drag{(int)this->topology.size()},
		data_size{10},
		epochs{0}
	;
	float
		epoch_ms{ 0.f }
	;
	bool
		s_tool_enable{ true },
		s_show_network{ false },
		s_show_graphs{ false },

		s_regen_avail{ false },

		s_training_thread{ false },
		s_training_loop{ false }
	;

	inline static constexpr char* func_names[]{
		"Sigmoid", "Hyperbolic Tangent", "ReLU"
	};


};