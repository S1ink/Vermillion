#include "mlmodel.h"

#include <fstream>
#include <iostream>

#include "Walnut/Random.h"

#include "util.h"


inline ImVec2 operator-(ImVec2 a, ImVec2 b) { return { a.x - b.x, a.y - b.y }; }

template<typename num_t>
inline int sgn(num_t val) { return (num_t(0) < val) - (val < num_t(0)); }

template<typename num_t>
inline num_t dist(num_t x1, num_t y1, num_t x2, num_t y2) {
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}


void MlTool::OnUIRender() {
	if (this->s_tool_enable) {
		if (ImGui::Begin("Neural Network Toolbox", &this->s_tool_enable)) {
			
			if (ImGui::Button(this->s_show_network ? "Close Network" : "View Network")) { this->s_show_network = !this->s_show_network; }
			ImGui::SameLine(0, 4);
			if (ImGui::Button(this->s_show_graphs ? "Close Training Progress" : "View Training Progress")) { this->s_show_graphs = !this->s_show_graphs; }
			ImGui::SameLine(0, 4);
			if (ImGui::Button(this->s_show_console ? "Close Log" : "Open Log")) { this->s_show_console = !this->s_show_console; }
			ImGui::SameLine(0, 24);
			ImPlot::ColormapButton("Primary Theme", ImVec2{}, this->primary_theme);
			ImGui::SameLine(0, 4);
			ImGui::SetNextItemWidth(100);
			if(ImGui::Combo("##theme1", &this->primary_theme, ImPlot::GetColormapName(0))) { ImPlot::BustColorCache(); }
			ImGui::SameLine(0, 16);
			ImPlot::ColormapButton("Secondary Theme", ImVec2{}, this->secondary_theme);
			ImGui::SameLine(0, 4);
			ImGui::SetNextItemWidth(100);
			if(ImGui::Combo("##theme2", &this->secondary_theme, ImPlot::GetColormapName(0))) { ImPlot::BustColorCache(); }

			if (ImGui::BeginChild("sections", ImVec2{ 180, 0 }, true/*, ImGuiWindowFlags_AlwaysAutoResize*/)) {	// old: 180xAuto

				ImGui::BeginChild("sect_network", ImVec2{ 180, 180 }, false/*, ImGuiWindowFlags_AlwaysAutoResize*/);	// old: 180x180
				if (ImGui::Selectable("##n", this->section_idx == 0, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 0; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::BeginGroup();
				if (this->section_idx != 0) { ImGui::BeginDisabled(); }
				{
					ImGui::Text("Network");
					ImGui::Spacing(); ImGui::Spacing();
					ImGui::Bullet(); ImGui::Text("Layers:  %d", this->topology.size());
					ImGui::Spacing(); ImGui::Spacing();
					ImGui::Bullet(); ImGui::Text("Inputs:  %d", this->inputs());
					ImGui::Bullet(); ImGui::Text("Outputs:  %d", this->outputs());
				}
				if (this->section_idx != 0) { ImGui::EndDisabled(); }
				ImGui::EndGroup();
				ImGui::EndChild();

				ImGui::Separator();

				ImGui::BeginChild("sect_dataset", ImVec2{180, 180}, false/*, ImGuiWindowFlags_AlwaysAutoResize*/);	// '''
				if (ImGui::Selectable("##d", this->section_idx == 1, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 1; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::BeginGroup();
				if (this->section_idx != 1) { ImGui::BeginDisabled(); }
				{
					static bool valid_set;
					valid_set = this->dataset.first.size() > 0;
					ImGui::Text("DataSet");
					ImGui::Spacing(); ImGui::Spacing();
					ImGui::AlignTextToFramePadding();
					ImGui::Bullet(); ImGui::Text("Valid Set:"); ImGui::SameLine();
					ImGui::BeginDisabled(); ImGui::Checkbox("##data_valid", &valid_set); ImGui::EndDisabled();
					ImGui::Bullet();
					ImGui::TextColored(
						(this->compatibleDataSet(this->dataset) ? ImVec4{0, 1, 0, 1} : ImVec4{1, 0, 0, 1}),
						"DataSet I/O: %d/%d", (valid_set ? this->dataset.first[0]->size() : 0), (valid_set ? this->dataset.second[0]->size() : 0)
					);
					ImGui::Bullet(); ImGui::Text("Sets: %d", this->dataset.first.size());
				}
				if (this->section_idx != 1) { ImGui::EndDisabled(); }		
				ImGui::EndGroup();
				ImGui::EndChild();

				ImGui::Separator();

				ImGui::BeginChild("sect_training", ImVec2{180, 180}, false/*, ImGuiWindowFlags_AlwaysAutoResize*/);	// '''
				if (ImGui::Selectable("##t", this->section_idx == 2, ImGuiSelectableFlags_None, ImGui::GetContentRegionAvail())) { this->section_idx = 2; }
				ImGui::SetCursorPos(ImGui::GetStyle().ItemInnerSpacing);
				ImGui::BeginGroup();
				if (this->section_idx != 2) { ImGui::BeginDisabled(); }
				{
					ImGui::Text("Training");
					ImGui::Spacing(); ImGui::Spacing();
					ImGui::Bullet(); ImGui::Text("Time per epoch:");
					ImGui::Indent(); ImGui::Text("%.3f ms", this->epoch_ms); ImGui::Unindent();
					ImGui::Bullet(); ImGui::Text("Epochs:");
					ImGui::Indent(); ImGui::Text("%d", this->epochs); ImGui::Unindent();
					ImGui::Bullet(); ImGui::Text("Current avg MSE:");
					ImGui::Indent(); ImGui::Text("%g", this->epochs_avg.size() > 0 ? this->epochs_avg.back() : 0); ImGui::Unindent();
				}
				if (this->section_idx != 2) { ImGui::EndDisabled(); }				
				ImGui::EndGroup();
				ImGui::EndChild();

				ImGui::Separator();

			} ImGui::EndChild();
			ImGui::SameLine();
			if (ImGui::BeginChild("interactive")) {
				ImGui::Separator();
				switch (this->section_idx) {
				case 0: {
/* NETWORK OPTIONS*/
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
					ImGui::SameLine(0, 4);
					if (ImGui::Button("Save Model")) {
						std::string f;
						if (saveFile(f)) {
							std::ofstream o(f);
							this->export_weights(o);
							o.close();
						}
					}
					ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 8, 25 });
					if (ImGui::Button("Dump Network")) {
						this->dump(this->console_log);
						this->s_show_console = true;
					}
					ImGui::PopStyleVar();

					if (ImGui::Button("Remix Weights")) { this->remix(); }
					ImGui::SameLine(0, 4);
					bool eq = (this->topo_edit == this->topology);
					if (eq) { ImGui::BeginDisabled(); }
					if (ImGui::Button("Resize Network")) {
						this->topology = this->topo_edit;
						this->regenerate();
					}
					if (eq) { ImGui::EndDisabled(); }
					ImGui::Separator();

					ImGui::AlignTextToFramePadding();
					ImGui::BulletText("Activation:"); ImGui::SameLine();
					ImGui::SetNextItemWidth(180);
					if (ImGui::Combo("##a_func_combo", &activ_f_idx, func_names, 4)) { this->setActivationFunc((ActivationFunc)this->activ_f_idx); }
					
					ImGui::AlignTextToFramePadding();
					ImGui::BulletText("Learning Rate:"); ImGui::SameLine();
					ImGui::SetNextItemWidth(60);
					ImGui::DragFloat("##lr_drag", &this->learning_rate, 0.0001f, 0.f, 1.f, "%.3f");

					ImGui::AlignTextToFramePadding();
					ImGui::BulletText("Regularization Strategy:"); ImGui::SameLine();
					ImGui::SetNextItemWidth(200);
					if (ImGui::Combo("##reg_func_combo", &reg_f_idx, regl_names, 4)) { this->setRegularization((Regularization)this->reg_f_idx); }
					
					if (this->reg_f_idx == Regularization::NONE) { ImGui::BeginDisabled(); }
					ImGui::AlignTextToFramePadding();
					ImGui::BulletText("Regularization Rate:"); ImGui::SameLine();
					ImGui::SetNextItemWidth(60);
					ImGui::DragFloat("##rr_drag", &this->reg_rate, 0.0001f, 0.f, 1.f, "%.3f");
					if (this->reg_f_idx == Regularization::NONE) { ImGui::EndDisabled(); }

					ImGui::AlignTextToFramePadding();
					ImGui::BulletText("Layers:"); ImGui::SameLine();
					ImGui::SetNextItemWidth(30);
					if (ImGui::DragInt("##layers_drag", &this->layers_drag, 0.04, 2, 25)) { this->topo_edit.resize(this->layers_drag, 1); }
					ImGui::Spacing();

//					ImGui::Indent();
					ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
					if (ImGui::TreeNode("Topology")) {
						ImGui::Spacing();
						static int z;
						for (size_t i = 0; i < this->topo_edit.size(); i++) {
							ImGui::PushID(i);
							z = this->topo_edit[i];
							if (ImGui::VSliderInt("##l", { 25, 150 }, &z, 1, 20, "%d", ImGuiSliderFlags_AlwaysClamp)) { this->topo_edit[i] = z; };
							ImGui::PopID();
							ImGui::SameLine(0, 2);
						}
						ImGui::TreePop();
					}
//					ImGui::Unindent();
					break;
				}
				case 1: {
/*DATASET OPTIONS*/
					// make inference
					// gen dataset - export/load
					ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 8, 25 });
					if (ImGui::Button("Regenerate Function")) { this->genFunc(this->rfunc); }
					ImGui::SameLine(0, 32);
					if (ImGui::Button("Output Function")) { this->rfunc.serializeFunc(this->console_log); this->s_show_console = true; }
					ImGui::SameLine(0, 4);
					if (ImGui::Button("View Structure")) { this->rfunc.serializeStructure(this->console_log); this->s_show_console = true; }
					ImGui::PopStyleVar();
					if (ImGui::Button("Regenerate Dataset")) {
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
					if (ImGui::Button("Import Dataset")) {
						std::string f;
						if (openFile(f)) {
							std::ifstream in(f);
							importData(this->dataset, in);
						}
					}
					ImGui::SameLine(0, 4);
					if (ImGui::Button("Export Dataset") && this->dataset.first.size() > 0) {
						std::string f;
						if (saveFile(f)) {
							std::ofstream out(f);
							exportData(this->dataset, out);
						}
					} ImGui::SameLine(0, 32);
					if (ImGui::Button("View Dataset") && this->dataset.first.size() > 0) {
						exportData(this->dataset, this->console_log);
						this->s_show_console = true;
					}

					break;
				}
				case 2: {
/*TRAINING OPTIONS*/
					// view progress
					if (ImGui::Button("Clear Epoch History")) {
						this->epochs_avg.clear();
						this->epochs_high.clear();
						this->epochs_low.clear();
						this->epochs = 0;
					}

				// add check for dataset size and network i/o size
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

				} }

			} ImGui::EndChild();

			// viewing windows

			static ImVec2 sz;
// NETWORK PLOTS
			if (this->s_show_network) {
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0, 0 });
				if (ImGui::Begin("Network Matrix", &this->s_show_network/*, ImGuiWindowFlags_AlwaysAutoResize*/)) {

					constexpr static float unit_sz{ 50.f }, layer_spacing{ 2.f }, dash_len{ 0.1 }, dash_space{0.02};
					constexpr static int pt_len{ 512 };
					static float line_x[pt_len], line_y[pt_len], start_x, start_y, val, clamp, x, y;
					static int r, c, layer_display{ 0 };
					static bool
						net_matx_view{ false },	// true for net, false for matx
						net_show_biases{ false },
						animate{ false };

					sz = ImGui::GetContentRegionAvail();
					sz.y -= (ImGui::GetStyle().ItemSpacing.y * 2 - ImGui::GetItemRectSize().y);
					x = (net_matx_view ? (this->topology.size() * (1 + layer_spacing) - layer_spacing) : (this->computeViewWidth() + (0.1 * this->weights.size()))) + 2;
					y = this->computeViewHeight() + 2 + !net_matx_view;
					sz.y = sz.x * y / x;

					ImGui::Spacing(); ImGui::Spacing(); ImGui::SameLine();	// account for no padding
					if (ImGui::Button(net_matx_view ? "Show Matrix View" : "Show Connection View")) { net_matx_view = !net_matx_view; }
					if (net_matx_view) {
						ImGui::SameLine(0, 32);
						if (ImGui::Button(net_show_biases ? "Show Neuron Values" : "Show Biases")) { net_show_biases = !net_show_biases; }
						ImGui::SameLine();
						ImGui::Checkbox("Animate", &animate);
					} else {
						ImGui::SameLine(0, 32);
						ImGui::AlignTextToFramePadding();
						ImGui::Text("Layer Display:");
						ImGui::SameLine();
						ImGui::SetNextItemWidth(150);
						ImGui::Combo("##ldselect", &layer_display, "Neuron Values\0Cache Values\0Deltas");
					}
		// PLOT BLOCK
					ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, { 0, 0 });
					if (ImPlot::BeginPlot("##network", sz, ImPlotFlags_NoMouseText | ImPlotFlags_Equal)) {
						ImPlot::SetupAxes(NULL, NULL, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
						ImPlot::SetupAxesLimits(0, x, 0, y);
						ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, x);
						ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0, y);
						Scalar_t* dtbeg;
						if (net_matx_view) {
							static int offset_b;
							static float offset;
							offset_b = (offset_b + 1) % ((int)round(ImGui::GetIO().Framerate) + 1);
							offset = (float)offset_b / (int)round(ImGui::GetIO().Framerate) * (dash_len + dash_space);
							for (size_t i = 0; i < this->topology.size(); i++) {
								r = this->topology[i];
								c = 1;
								start_x = (i * (layer_spacing + 1)) + 1;
								start_y = ((y - 2 - r) / 2.f) + 1;
								if (i != this->topology.size() - 1) {
									ImPlot::PushColormap(this->primary_theme);
									line_x[0] = start_x + 0.75;
									line_x[pt_len - 1] = start_x + layer_spacing + 1.25;
									for (size_t rw = 0; rw < this->weights[i]->rows() - 1; rw++) {
										line_y[0] = start_y + (this->topology[i] - rw) - 0.5;
										for (size_t cw = 0; cw < this->weights[i]->cols() - (i != this->topology.size() - 2); cw++) {
											ImGui::PushID((i & 0xff) | ((cw & 0xff) << 8) | ((rw & 0xff) << 16));
											val = this->weights[i]->coeff(rw, cw);
											clamp = (val > 5.f ? 5.f : (val < -5.f ? -5.f : val));
											ImVec4 color = ImPlot::SampleColormap(clamp / 10.f + 0.5f);
											ImPlot::SetNextLineStyle(
												color,
												fabs(clamp) * 2.f
											);
											//ImPlot::GetPlotSize();	// ^^^
											line_y[pt_len - 1] =  0.5 * (y + this->topology[i + 1] - 1) - cw;
											float d = dist(line_x[0], line_y[0], line_x[pt_len - 1], line_y[pt_len - 1]);
											static int len;
											static float t;
											bool s_anim = animate;	// static state
											t = offset / d;
											len = 0;
											for (int k = 1; k < pt_len - 1; k++) {
												static float u, w1, w2, w3, w4;
												if (s_anim) {
													if (!(k % 3)) {
														line_x[k] = NAN;
														line_y[k] = NAN;
														continue;
													} else {
														t += (!(k % 3 - 1) * dash_space + !(k % 3 - 2) * dash_len) / d;
													}
													if (t > 1.f) {
														t = 1.f;
														line_x[k + 1] = line_x[pt_len - 1];
														line_y[k + 1] = line_y[pt_len - 1];
														len = k + 2;
													}
												} else {
													t = k / (float)(pt_len - 1);	// equal increments
													len = pt_len;
												}
												u = 1 - t;
												w1 = u * u * u;
												w2 = 3 * u * u * t;
												w3 = 3 * u * t * t;
												w4 = t * t * t;
												line_x[k] = w1 * line_x[0] + w2 * (line_x[0] + 1.f) + w3 * (line_x[pt_len - 1] - 1.f) + w4 * (line_x[pt_len - 1]);
												line_y[k] = w1 * line_y[0] + w2 * line_y[0] + w3 * line_y[pt_len - 1] + w4 * line_y[pt_len - 1];
												if (t == 1.f) { break; }
											}
											ImPlot::PlotLine("##wl", line_x, line_y, len ? len : pt_len);
											ImGui::PopID();
										}
									}
									ImPlot::PopColormap();
								}
								dtbeg = (net_show_biases && i > 0 && i != this->topology.size() - 1) ?
									(this->weights[i - 1]->row(this->topology[i - 1]).data()) : this->neurons_matx[i]->data();
								ImGui::PushID((i & 0xff) | ((c & 0xff) << 8) | ((r & 0xff) << 16));
								ImPlot::PushColormap(this->secondary_theme);
								ImPlot::PlotHeatmap("##nl", dtbeg, r, c, -1, 1, "%.1f", { start_x, start_y }, { start_x + c, start_y + r });
								ImPlot::PopColormap();
								ImGui::PopID();
							}
						} else {
							start_x = 1;	// set to padding size
							for (size_t i = 0; i < this->topology.size(); i++) {
								r = this->neurons_matx[i]->cols();
								c = 1;
								start_y = ((y - 2 - r) / 2.f) + 1;
								switch (layer_display) {
									case 0: dtbeg = this->neurons_matx[i]->data(); break;
									case 1: dtbeg = this->cache_matx[i]->data(); break;
									case 2: dtbeg = this->errors[i]->data(); break;
								}
								ImGui::PushID((i & 0xff) | ((c & 0xff) << 8) | ((r & 0xff) << 16));
								ImPlot::PushColormap(this->secondary_theme);
								ImPlot::PlotHeatmap("##nl", dtbeg, r, c, -1, 1, "%.1f", { start_x, start_y }, { start_x + c, start_y + r });
								ImPlot::PopColormap();
								ImGui::PopID();
								if (i != this->topology.size() - 1) {
									r = this->weights[i]->rows();
									c = this->weights[i]->cols();
									start_x += 1.05;
									start_y = ((y - 2 - r) / 2.f) + 1;
									ImGui::PushID((i & 0xff) | ((c & 0xff) << 8) | ((r & 0xff) << 16));
									ImPlot::PushColormap(this->primary_theme);
									ImPlot::PlotHeatmap("##nw", this->weights[i]->data(), r, c, -1, 1, "%.1f", { start_x, start_y }, { start_x + c, start_y + r });
									ImPlot::PopColormap();
									ImGui::PopID();
									start_x += (c + 0.05);
								}
							}
						}
						ImPlot::EndPlot();
					}
					ImPlot::PopStyleVar();
				} ImGui::End();
				ImGui::PopStyleVar();
			}
// GRAPH PLOTS
			if (this->s_show_graphs) {
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0, 0 });
				if (ImGui::Begin("Training Progress", &this->s_show_graphs)) {
					static ImPlotScale scale_x{ ImPlotScale_Log10 }, scale_y{ ImPlotScale_Log10 };
					sz = ImGui::GetContentRegionAvail();
					ImGui::Spacing();
					ImGui::Spacing(); ImGui::SameLine();
					ImGui::CheckboxFlags("X-Axis Log10 Scale", &scale_x, ImPlotScale_Log10);
					ImGui::SameLine();
					ImGui::CheckboxFlags("Y-Axis Log10 Scale", &scale_y, ImPlotScale_Log10);
					if (ImPlot::BeginSubplots("Training Progress", 1, 2, {sz.x, sz.y - ImGui::GetStyle().ItemSpacing.y * 2 - ImGui::GetItemRectSize().y})) {
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
// CONSOLE
			if (this->s_show_console) {
				//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0, 0 });
				if (ImGui::Begin("Log", &this->s_show_console)) {
					//ImGui::PopStyleVar();
					if (ImGui::Button("Clear Log")) { std::ostringstream().swap(this->console_log); }
					if (ImGui::BeginChild("##console", { 0, 0 }, true, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
						ImGui::TextWrapped("%s", this->console_log.str().c_str());
					} ImGui::EndChild();
				} else {
					//ImGui::PopStyleVar();
				} ImGui::End();
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

size_t MlTool::computeViewWidth() const {
	size_t ret = this->topology.size();
	for (size_t i = 0; i < this->topology.size() - 1; i++) {
		ret += this->weights[i]->cols();
	}
	return ret;
}