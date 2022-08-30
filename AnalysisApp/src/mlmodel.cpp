#include "mlmodel.h"

#include <fstream>
#include <iostream>

#include "Walnut/Random.h"

#include "util.h"


inline ImVec2 operator-(ImVec2 a, ImVec2 b) { return { a.x - b.x, a.y - b.y }; }


void MlTool::OnUIRender() {
	if (this->s_tool_enable) {
		if (ImGui::Begin("Neural Network Toolbox", &this->s_tool_enable)) {
			
			if (ImGui::Button(this->s_show_network ? "Close Network" : "View Network")) { this->s_show_network = !this->s_show_network; }
			ImGui::GetStyle().ItemSpacing.x = 4;
			ImGui::SameLine();
			if (ImGui::Button(this->s_show_graphs ? "Close Training Progress" : "View Training Progress")) { this->s_show_graphs = !this->s_show_graphs; }
			ImGui::GetStyle().ItemSpacing.x = 16;
			ImGui::SameLine();
			ImPlot::ColormapButton("Primary Color", ImVec2{}, this->primary_theme);
			ImGui::GetStyle().ItemSpacing.x = 4;
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			if(ImGui::Combo("##theme1", &this->primary_theme, ImPlot::GetColormapName(0))) { ImPlot::BustColorCache(); }
			ImGui::GetStyle().ItemSpacing.x = 16;
			ImGui::SameLine();
			ImPlot::ColormapButton("Secondary Theme", ImVec2{}, this->secondary_theme);
			ImGui::GetStyle().ItemSpacing.x = 4;
			ImGui::SameLine();
			ImGui::SetNextItemWidth(100);
			if(ImGui::Combo("##theme2", &this->secondary_theme, ImPlot::GetColormapName(0))) { ImPlot::BustColorCache(); }
			ImGui::GetStyle().ItemSpacing.x = 8;

			if (ImGui::BeginChild("sections", ImVec2{ 180, 0 }, true)) {

				ImGui::BeginChild("sect_network", ImVec2{180, 180});
				if (ImGui::Selectable("##n", this->section_idx == 0, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 0; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::TextColored({1, 0.6, 0, 1}, "Network");
				ImGui::EndChild();

				ImGui::Separator();

				ImGui::BeginChild("sect_dataset", ImVec2{ 180, 180 });
				if (ImGui::Selectable("##d", this->section_idx == 1, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 1; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::TextColored({ 1, 0.6, 0, 1 }, "DataSet");
				ImGui::EndChild();

				ImGui::Separator();

				ImGui::BeginChild("sect_training", ImVec2{ 180, 180 });
				if (ImGui::Selectable("##t", this->section_idx == 2, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 2; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::BeginGroup();
				ImGui::TextColored({ 1, 0.6, 0, 1 }, "Training");
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 10, 10 });
				ImGui::Spacing();
				ImGui::PopStyleVar();
				ImGui::Bullet(); ImGui::Text("Time per epoch:");
				ImGui::Indent();
				ImGui::Text("%.3f ms", this->epoch_ms);
				ImGui::Unindent();
				ImGui::Bullet(); ImGui::Text("Epochs: %d", this->epochs);
				ImGui::EndGroup();
				ImGui::EndChild();

				ImGui::Separator();

			} ImGui::EndChild();
			ImGui::SameLine();
			if (ImGui::BeginChild("interactive")) {

				ImGui::Separator();

				switch (this->section_idx) {
				case 0: {

					
				}
				case 1: {
					
				}
				case 2: {


				} }

				// set learning rate
				ImGui::PushItemWidth(70);
				ImGui::Text("Learning Rate: "); ImGui::SameLine();
				ImGui::DragFloat("##lr", &this->learning_rate, 0.0001f, 0.f, 1.f, "%.3f");

				// change activation func
				ImGui::Text("Activation Function:"); ImGui::SameLine();
				ImGui::PushItemWidth(200);
				if (ImGui::Combo("##a_func", &activ_f_idx, func_names, 3)) {
					this->setActivationFunc((ActivationFunc)this->activ_f_idx);
				}
				ImGui::PopItemWidth();

				ImGui::Separator();

				// set topo
				ImGui::PushItemWidth(30);
				ImGui::Text("Layers: "); ImGui::SameLine();
				if (ImGui::DragInt("##layers", &this->layers_drag, 0.04, 2, 25)) {
					this->topo_edit.resize(this->layers_drag, 1);
				}
				ImGui::PopItemWidth();

				ImGui::PushItemWidth(50);
				ImGui::PushID(0);
				int z = this->topo_edit[0];
				if (ImGui::VSliderInt("##l", { 25, 100 }, &z, 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp)) { this->topo_edit[0] = z; }
				ImGui::PopID();
				for (size_t i = 1; i < this->topo_edit.size(); i++) {
					ImGui::SameLine();
					ImGui::PushID(i);
					z = this->topo_edit[i];
					if (ImGui::VSliderInt("##l", { 25, 100 }, &z, 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp)) { this->topo_edit[i] = z; };
					ImGui::PopID();
				}
				ImGui::PopItemWidth();

				bool eq = (this->topo_edit == this->topology);
				if (eq) { ImGui::BeginDisabled(); }
				if (ImGui::Button("Regenerate Network")) {
					this->topology = this->topo_edit;
					this->regenerate();
				}
				if (eq) { ImGui::EndDisabled(); }

				ImGui::Separator();

				/*if (ImGui::Button("Test")) {
					std::cout << "\n\n";
					for (size_t i = 0; i < this->weights.size(); i++) {
						std::cout << this->weights[i]->rows() << " x " << this->weights[i]->cols() << '\n';
						for (size_t f = 0; f < this->weights[i]->rows() * this->weights[i]->cols(); f++) {
							std::cout << this->weights[i]->data()[f] << ' ';
						}
						std::cout << std::endl;
					}
				}*/




				// make inference
				// gen dataset - export/load
				if (ImGui::Button("Regenerate Function")) { this->genFunc(this->rfunc); }
				ImGui::SameLine();
				if (ImGui::Button("Output Function")) { this->rfunc.serializeFunc(std::cout); }
				ImGui::SameLine();
				if (ImGui::Button("View Structure")) { this->rfunc.serializeStructure(std::cout); }
				if (ImGui::Button("Regenerate DataSet")) {
					if (!this->compatibleFunc(this->rfunc)) {
						this->genFunc(this->rfunc);
					}
					this->genData(this->dataset, this->data_size, this->rfunc);
				}
				ImGui::SameLine();
				ImGui::Text("Sets:");
				ImGui::SameLine();
				ImGui::SetNextItemWidth(50);
				ImGui::DragInt("##dataset_sz", &this->data_size, 1, 1, 1000);
				if (ImGui::Button("Export DataSet") && this->dataset.first.size() > 0) {
					std::string f;
					if (saveFile(f)) {
						std::ofstream out(f);
						exportData(this->dataset, out);
					}
				} ImGui::SameLine();
				if (ImGui::Button("View DataSet") && this->dataset.first.size() > 0) {
					exportData(this->dataset, std::cout);
				}


				// view progress
				if (ImGui::Button("Clear Epoch History")) {
					this->epochs_avg.clear();
					this->epochs_high.clear();
					this->epochs_low.clear();
					this->epochs = 0;
				}
				
				// train - inc/run (x)times
				bool disable = this->s_training_thread || this->trainer.joinable();
				if (disable) { ImGui::BeginDisabled(); }
				if (ImGui::Button("Train (Single Run)")) {
					this->trainer = std::thread(trainThread, this);
					this->s_training_thread = true;
				}
				if (disable) { ImGui::EndDisabled(); }
				if (ImGui::Button(this->s_training_loop ? "Stop Continuous" : "Start Continuous")) {
					if (this->s_training_loop = !this->s_training_loop) {
						this->trainer = std::thread(trainThread, this);
						this->s_training_thread = true;
					}
				}
				if (!this->s_training_thread && this->trainer.joinable()) {
					this->trainer.join();
				}

				if (ImGui::Button("Save Model")) {
					std::string f;
					if (saveFile(f)) {
						std::ofstream o(f);
						this->export_weights(o);
						o.close();
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Load Model")) {
					std::string f;
					if (openFile(f)) {
						std::ifstream i(f);
						Weights_t buff;
						this->parse_weights(i, buff);
						this->regenerate(std::move(buff));
						this->topo_edit = this->topology;
						this->layers_drag = this->topology.size();
					}
				}
			} ImGui::EndChild();

			// viewing windows

			if (this->s_show_network) {
				ImGui::SetNextWindowBgAlpha(0.5);
				if (ImGui::Begin("Network Matrix", &this->s_show_network, ImGuiWindowFlags_AlwaysAutoResize)) {
					
					ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, { 0, 0 });
					static int x, y;
					for (int i = 0; i < this->weights.size() + this->topology.size(); i++) {
						if (i % 2) {	// weight
							x = this->weights[i / 2]->cols();
							y = this->weights[i / 2]->rows();
							ImGui::PushID((i & 0xff) | ((x & 0xff) << 8) | ((y & 0xff) << 16));
							ImPlot::PushColormap(this->primary_theme);
							if (ImPlot::BeginPlot("##w", { (x * 50.f), (y * 50.f) }, ImPlotFlags_CanvasOnly)) {
								ImPlot::SetupAxes(NULL, NULL, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations);
								ImPlot::SetupAxesLimits(0, x, 0, y);
								ImPlot::PlotHeatmap("##h", this->weights[i / 2]->data(), y, x, 0, 0, "%.1f", { 0, 0 }, ImPlotPoint{ (float)x, (float)y });
								ImPlot::EndPlot();
							}
							ImPlot::PopColormap();
						} else {	// neuron layer
							x = 1;
							y = this->topology[i / 2];
							ImGui::PushID((i & 0xff) | ((x & 0xff) << 8) | ((y & 0xff) << 16));
							ImPlot::PushColormap(this->secondary_theme);
							if (ImPlot::BeginPlot("##n", { (x * 50.f), (y * 50.f) }, ImPlotFlags_CanvasOnly)) {
								ImPlot::SetupAxes(NULL, NULL, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations);
								ImPlot::SetupAxesLimits(0, x, 0, y);
								ImPlot::PlotHeatmap("##h", this->neurons_matx[i / 2]->data(), y, x, 0, 0, "%.1f", { 0, 0 }, ImPlotPoint{ (float)x, (float)y });
								ImPlot::EndPlot();
							}
							ImPlot::PopColormap();
						}
						ImGui::PopID();
						ImGui::SameLine();
					}
					ImPlot::PopStyleVar();
				} ImGui::End();
			}
			if (this->s_show_graphs) {
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0, 0 });
				if (ImGui::Begin("Training Progress", &this->s_show_graphs)) {
					static ImVec2 sz;
					sz = ImGui::GetContentRegionAvail();
					static ImPlotScale scale_x{ ImPlotScale_Log10 }, scale_y{ ImPlotScale_Log10 };
					ImGui::CheckboxFlags("X-Axis Log10 Scale", &scale_x, ImPlotScale_Log10);
					ImGui::SameLine();
					ImGui::CheckboxFlags("Y-Axis Log10 Scale", &scale_y, ImPlotScale_Log10);
					if (ImPlot::BeginSubplots("Training Progress", 1, 2, {sz.x, sz.y - ImGui::GetTextLineHeight()})) {
						if (ImPlot::BeginPlot("H/L/Avg MSE per Epoch", {-1, 0}, ImPlotFlags_Crosshairs)) {
							ImPlot::SetupAxes("Epoch", "MSE", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
							ImPlot::SetupAxisScale(ImAxis_X1, scale_x);
							ImPlot::SetupAxisScale(ImAxis_Y1, scale_y);
							ImPlot::SetupLegend(ImPlotLocation_NorthEast);
							if (this->train_mutex.try_lock()) {
								ImPlot::SetNextLineStyle({ 0.7, 0.7, 0, 1 });
								ImPlot::PlotLine("High MSE", this->epochs_high.data(), this->epochs_high.size());
								ImPlot::SetNextLineStyle({ 0, 0.7, 0.3, 1 });
								ImPlot::PlotLine("Avg MSE", this->epochs_avg.data(), this->epochs_avg.size());
								ImPlot::SetNextLineStyle({ 0, 0.4, 0.7, 1 });
								ImPlot::PlotLine("Low MSE", this->epochs_low.data(), this->epochs_low.size());
								this->train_mutex.unlock();
							}
							ImPlot::EndPlot();
						}
						if (ImPlot::BeginPlot("Last Epoch MSE Trend", { -1, 0 }, ImPlotFlags_Crosshairs)) {
							ImPlot::SetupAxes("DataPoint", "MSE", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
							//ImPlot::SetupAxisScale(ImAxis_X1, scale_x);
							//ImPlot::SetupAxisScale(ImAxis_Y1, scale_y);
							ImPlot::SetNextLineStyle({ 0.8, 0.4, 0, 1 });
							ImPlot::PlotLine("MSE Trend", this->epoch_last.data(), this->epoch_last.size());
							ImPlot::EndPlot();
						}						
						ImPlot::EndSubplots();
					}
				} ImGui::End();
				ImGui::PopStyleVar();
			}

		} ImGui::End();
	}
}

//template<typename t>
float vec_itr_avg(std::vector<float>::iterator s, std::vector<float>::iterator e) {
	float sum{0};
	size_t count{ 0 };
	for (; s < e; s++) {
		sum += *s;
		count++;
	}
	return sum / count;
}
//Prereq: low is higher than the lowest expected val, high is lower than the highest expected value
void vec_itr_hla(std::vector<float>::iterator start, std::vector<float>::iterator end, float& high, float& low, float& avg) {
	avg = 0;
	size_t count{ 0 };
	for (; start < end; start++) {
		avg += *start;
		if (*start > high) { high = *start; }
		if (*start < low) { low = *start; }
		count++;
	}
	avg /= count;
}

void MlTool::trainThread(MlTool* t) {
	if (t->dataset.first.size() > 0) {
		using hrc = std::chrono::high_resolution_clock;
		hrc::time_point ref;
		if (!t->s_training_loop) {
			std::lock_guard<std::mutex> l{ t->train_mutex };
			ref = hrc::now();
			t->epoch_last.clear();
			t->train_graph(t->dataset, t->epoch_last);
			t->epochs_avg.push_back(0);
			t->epochs_high.push_back(0);
			t->epochs_low.push_back(10000.f);
			vec_itr_hla(t->epoch_last.begin(), t->epoch_last.end(), t->epochs_high.back(), t->epochs_low.back(), t->epochs_avg.back());
			t->epoch_ms = (hrc::now() - ref).count() / 1e6;
			t->epochs++;
		} else {
			while (t->s_training_loop) {
				ref = hrc::now();
				std::lock_guard<std::mutex> l{ t->train_mutex };
				t->epoch_last.clear();
				t->train_graph(t->dataset, t->epoch_last);
				t->epochs_avg.push_back(0);
				t->epochs_high.push_back(-1.f);
				t->epochs_low.push_back(10000.f);
				vec_itr_hla(t->epoch_last.begin(), t->epoch_last.end(), t->epochs_high.back(), t->epochs_low.back(), t->epochs_avg.back());
				t->epoch_ms = (hrc::now() - ref).count() / 1e6;
				t->epochs++;
			}
		}
	}
	t->s_training_thread = false;
}