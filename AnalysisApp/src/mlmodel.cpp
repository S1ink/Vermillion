#include "mlmodel.h"

#include <fstream>
#include <iostream>

#include "implot/implot.h"

#include "Walnut/Random.h"

#include "util.h"


void MlTool::OnUIRender() {
	if (this->s_tool_enable) {
		ImGui::Begin("Neural Network Toolbox");

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
			if(ImGui::VSliderInt("##l", {25, 100}, &z, 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp)) { this->topo_edit[0] = z; }
			ImGui::PopID();
			for (size_t i = 1; i < this->topo_edit.size(); i++) {
				ImGui::SameLine();
				ImGui::PushID(i);
				z = this->topo_edit[i];
				if (ImGui::VSliderInt("##l", {25, 100}, &z, 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp)) { this->topo_edit[i] = z; };
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

			ImGui::Checkbox("Show Weights Heatmap", &this->s_show_weights);
			if (this->s_show_weights) {
				ImGui::Begin("Weights");
				{
					ImPlot::PushColormap(ImPlotColormap_Hot);
					ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, {0, 0});
					int x, y;
					for (int i = 0; i < this->weights.size(); i++) {
						x = this->weights[i]->rows();
						y = this->weights[i]->cols();
						ImGui::PushID((i & 0xff) | ((x & 0xff) << 8) | ((y & 0xff) << 16));
						
						if (ImPlot::BeginPlot("##w", { (x * 50.f), (y * 50.f) }, ImPlotFlags_CanvasOnly)) {
							ImPlot::SetupAxes(NULL, NULL, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations);
							ImPlot::SetupAxesLimits(0, x, 0, y);
							ImPlot::PlotHeatmap("##h", this->weights[i]->data(), y, x, -1, 1, "%.1f", {0, 0}, ImPlotPoint{(float)x, (float)y});
							ImPlot::EndPlot();
						}
						ImGui::PopID();
						ImGui::SameLine();
					}
					ImPlot::PopColormap();
					ImPlot::PopStyleVar();
				}
				ImGui::End();
			}
			if (ImGui::Button("Test")) {
				std::cout << "\n\n";
				for (size_t i = 0; i < this->weights.size(); i++) {
					std::cout << this->weights[i]->rows() << " x " << this->weights[i]->cols() << '\n';
					for (size_t f = 0; f < this->weights[i]->rows() * this->weights[i]->cols(); f++) {
						std::cout << this->weights[i]->data()[f] << ' ';
					}
					std::cout << std::endl;
				}
			}

			// make inference
			// gen dataset - export/load
			
			// view progress
			if (ImGui::Button("Add Item")) { this->progress.push_back(Walnut::Random::Float()); }
			ImGui::SameLine();
			if (ImGui::Button("Clear Plot")) { this->progress.clear(); }
			if (ImPlot::BeginPlot("Training Progress")) {
				ImPlot::PlotLine("Random Data", this->progress.data(), this->progress.size());
				ImPlot::EndPlot();
			}
			// train - inc/run (x)times

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
		ImGui::End();
	}
}