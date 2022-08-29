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
					ImPlot::PushColormap(ImPlotColormap_Cool);
					ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, {0, 0});
					int x, y;
					for (int i = 0; i < this->weights.size(); i++) {
						x = this->weights[i]->cols();
						y = this->weights[i]->rows();
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
			if (ImGui::Button("Add Item")) { this->progress.push_back(Walnut::Random::Float()); }
			ImGui::SameLine();
			if (ImGui::Button("Clear Plot")) {
				this->progress.clear();
				this->epochs = 0;
			}
			ImGui::Text("Epoch Train Time (ms): %.3f", this->epoch_ms);
			ImGui::SameLine();
			ImGui::Text("Epochs: %d", this->epochs);
			if (ImPlot::BeginPlot("Training Progress", ImVec2{-50, 0})) {
				ImPlot::SetupAxes("DataPoint", "MSE", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
				ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
				//ImPlot::SetupAxesLimits(0, this->dataset.first.size(), 0, )
				if (this->train_mutex.try_lock()) {
					ImPlot::PlotLine("Random Data", this->progress.data(), this->progress.size());
					this->train_mutex.unlock();
				}
				ImPlot::EndPlot();
			}
			// train - inc/run (x)times
			bool disable = this->s_training_thread || this->trainer.joinable();
			if(disable) { ImGui::BeginDisabled(); }
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
		ImGui::End();
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

void MlTool::trainThread(MlTool* t) {
	if (t->dataset.first.size() > 0) {
		using hrc = std::chrono::high_resolution_clock;
		hrc::time_point ref;
		if (!t->s_training_loop) {
			std::lock_guard<std::mutex> l{ t->train_mutex };
			ref = hrc::now();
			if (t->epochs > 0) {
				size_t sz = t->dataset.first.size();
				float avg = vec_itr_avg(t->progress.begin() + t->epochs - 1, t->progress.begin() + sz + t->epochs - 1);
				t->progress.resize(t->epochs - 1);
				t->progress.push_back(avg);
			}
			t->train_graph(t->dataset, t->progress);
			t->epoch_ms = (hrc::now() - ref).count() / 1e6;
			t->epochs++;
		} else {
			while (t->s_training_loop) {
				ref = hrc::now();
				std::lock_guard<std::mutex> l{ t->train_mutex };
				if (t->epochs > 0) {
					size_t sz = t->dataset.first.size();
					float avg = vec_itr_avg(t->progress.begin() + t->epochs - 1, t->progress.begin() + sz + t->epochs - 1);
					t->progress.resize(t->epochs - 1);
					t->progress.push_back(avg);
				}
				t->train_graph(t->dataset, t->progress);
				t->epoch_ms = (hrc::now() - ref).count() / 1e6;
				t->epochs++;
			}
		}
	}
	t->s_training_thread = false;
}