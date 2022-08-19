#include "vision.h"

#include <iostream>

#include "util.h"


inline bool operator!=(ImVec2 a, ImVec2 b) { return a.x != b.x || a.y != b.y; }
template<typename ta, typename tb>
inline bool operator!=(cv::Size_<ta> a, cv::Size_<tb> b) { return a.width != b.width || a.height != b.height; }
inline bool operator>>(ImVec2 a, ImVec2 b) { return a.x > b.x || a.y > b.y; }
inline bool operator<<(ImVec2 a, ImVec2 b) { return a.x < b.x || a.y < b.y; }
template<typename ta, typename tb>
inline bool operator>>(cv::Size_<ta> a, cv::Size_<tb> b) { return a.width > b.width || a.height > b.height; }

inline float noNaN(float v) { return isnan(v) ? 0.f : v; }
inline double noNaN(double v) { return isnan(v) ? 0.0 : v; }


void VisionTool::OnUIRender() {
	if (this->s_tool_enable) {
		ImGui::Begin("Vision Options"); {
			if (ImGui::Button(this->s_capturing ? "Stop Capture" : "Start Capture")) {
				if (s_capturing = !s_capturing) {
					this->s_playing = false;
					this->cap_src.open(0);
					this->cap_src.set(cv::CAP_PROP_FRAME_WIDTH, 640);
					this->cap_src.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
					this->cap_src.set(cv::CAP_PROP_FPS, 60);
					this->view_size = { 0, 0 };
					std::cout <<
						"Cap mode: " << this->cap_src.get(cv::CAP_PROP_MODE) <<
						"\nFrame height: " << this->cap_src.get(cv::CAP_PROP_FRAME_HEIGHT) <<
						"\nFrame width: " << this->cap_src.get(cv::CAP_PROP_FRAME_WIDTH) <<
						"\nFPS: " << this->cap_src.get(cv::CAP_PROP_FPS) <<
						"\nFrame count: " << this->cap_src.get(cv::CAP_PROP_FRAME_COUNT) <<
						"\nFrame number: " << this->cap_src.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
				}/* else {
					this->cap_src.release();
				}*/
			} ImGui::SameLine();
			if (ImGui::Button("Grab Next")) {
				this->s_capturing = false;
				this->s_grabnext = true;
			}
			if (ImGui::Button("Load Video")) {
				if (openFile(this->vfile)) {
					this->s_capturing = false;
					this->cap_src.open(this->vfile);
					this->view_size = { 0, 0 };
					this->s_grabnext = true;
					std::cout <<
						"Cap mode: " << this->cap_src.get(cv::CAP_PROP_MODE) <<
						"\nFrame height: " << this->cap_src.get(cv::CAP_PROP_FRAME_HEIGHT) <<
						"\nFrame width: " << this->cap_src.get(cv::CAP_PROP_FRAME_WIDTH) <<
						"\nFPS: " << this->cap_src.get(cv::CAP_PROP_FPS) <<
						"\nFrame count: " << this->cap_src.get(cv::CAP_PROP_FRAME_COUNT) <<
						"\nFrame number: " << this->cap_src.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
				}
			}
			if (this->vfile.empty()) { ImGui::BeginDisabled(); }
			if (ImGui::Button(this->s_playing ? "Pause" : "Play")) {
				if (this->s_playing = !this->s_playing) {
					this->s_capturing = false;
					if (this->cap_src.get(cv::CAP_PROP_MODE) == -1) {
						this->cap_src.open(this->vfile);
						this->view_size = { 0, 0 };
					}
				}
			}
			if (this->vfile.empty()) { ImGui::EndDisabled(); }

			ImGui::BeginChild("settings", { 0, 0 }, true);
			if (ImGui::Checkbox("Enable Upscale", &this->s_upscale_viewport)) { this->view_size = { 0 }; }
			ImGui::Checkbox("Loop Video", &this->s_loopmode);
			ImGui::Checkbox("Limit Framerate", &this->s_limit_frate);
			ImGui::Checkbox("Display FPS", &this->s_fps_display);
			ImGui::Spacing();
			ImGui::Checkbox("Enable Processing", &this->s_enable_proc);
			ImGui::SliderFloat("Alpha", &this->alpha, 0.f, 1.f, "%.3f", 1);
			ImGui::SliderFloat("Beta", &this->beta, 0.f, 1.f, "%.3f", 1);
			ImGui::SliderFloat("Gamma", &this->gamma, 0.f, 100.f, "%.3f", 1);
			ImGui::EndChild();
		} ImGui::End();
		ImGui::Begin("Viewport"); {
			//std::lock_guard<std::mutex> io_lock{ this->framebuff_mutex };
			if (this->view_size.imvec != ImGui::GetContentRegionAvail() ||
				this->raw_size.cvsize.aspectRatio() != this->frame_size.cvsize.aspectRatio()
			) {
				this->view_size.imvec = ImGui::GetContentRegionAvail();
				this->frame_size.x = (this->raw_size.x > this->view_size.x || this->s_upscale_viewport ? this->view_size.x : this->raw_size.x);
				this->frame_size.y = noNaN(this->frame_size.x / (float)this->raw_size.cvsize.aspectRatio());
				this->resizeViewPortBuffers(this->frame_size);
				if (!this->rgba_out.empty()) {
					cv::resize(this->rgba_out, this->viewport_matbuff, this->frame_size.cvsize);
				}
			} else if(this->cap_src.isOpened()) {
				cv::resize(this->rgba_out, this->viewport_matbuff, this->frame_size.cvsize);
			}
			if (!this->viewport_matbuff.empty()) {
				this->viewport_gmem->SetData(this->viewport_matbuff.data);
				ImGui::Image(this->viewport_gmem->GetDescriptorSet(), this->frame_size.imvec);
			}
		} ImGui::End();

	}
}
void VisionTool::imgPipeline(VisionTool* that, const bool& state) {
	using hrc = std::chrono::high_resolution_clock;
	hrc::time_point ref;
	//bool keep_current = false;
	for (;state;) {
		if (that->cap_src.isOpened()) {
			if (that->s_playing) {
				if (that->s_loopmode && that->cap_src.get(cv::CAP_PROP_POS_FRAMES) == that->cap_src.get(cv::CAP_PROP_FRAME_COUNT)) {
					that->cap_src.set(cv::CAP_PROP_POS_FRAMES, 0);
				}
				if (!that->cap_src.grab()) { goto wait; }
				if (that->s_limit_frate) {
					std::this_thread::sleep_for(std::chrono::microseconds((uint)(1e6 / that->cap_src.get(cv::CAP_PROP_FPS))) - (hrc::now() - ref));
				}
			}
			else if (that->s_capturing && that->cap_src.grab()) {}
			else if (that->s_grabnext && that->cap_src.grab()) { that->s_grabnext = false; }
			//else if (that->s_enable_proc) { keep_current = true; }
			else { goto wait; }

			that->fps = 1e9 / (hrc::now() - ref).count();
			ref = hrc::now();

			//if (!keep_current) {
				that->cap_src.retrieve(that->raw_frame);
			//	keep_current = false;
				if (that->raw_frame.size() != that->raw_size.cvsize) {
					that->raw_size.cvsize = that->raw_frame.size();
					that->resizeProcBuffers(that->raw_size);
				}
			//} else {
			//	std::this_thread::sleep_for(std::chrono::milliseconds(20) - (hrc::now() - ref));
			//}
			
			if(that->s_enable_proc) {
				cv::split(that->raw_frame, that->channels);
				cv::addWeighted(that->channels[0], that->alpha, that->channels[1], that->beta, that->gamma, that->binary);
				cv::subtract(that->channels[2], that->binary, that->binary);

				cv::aruco::detectMarkers(that->raw_frame, that->markers_dict, that->corners, that->ids);
				cv::findContours(that->binary, that->contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

				cv::cvtColor(that->binary, that->proc, cv::COLOR_GRAY2BGR);
				cv::aruco::drawDetectedMarkers(that->proc, that->corners, that->ids);
				cv::drawContours(that->proc, that->contours, -1, { 0, 100, 255 }, 2);
				cv::cvtColor(that->proc, that->rgba_out, cv::COLOR_BGR2RGBA);

			} else {
				cv::cvtColor(that->raw_frame, that->rgba_out, cv::COLOR_BGR2RGBA);
			}
			if (that->s_fps_display) {
				cv::putText(that->rgba_out, "FPS: " + std::to_string(that->fps), { 10, 20 }, cv::FONT_HERSHEY_DUPLEX, 0.5, { 0, 255, 0, 255 });
			}
			continue;
		}
		wait:
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		//std::cout << "FTime: " << static_cast<std::chrono::duration<double>>(hrc::now() - ref).count() << std::endl;
	}
}

//void AppLayer::OnUIRender() {
//	//ImGui::ShowDemoWindow();
//	//ImGui::End();
//	if (this->vision_w) {
//
//		ImGui::Begin("Processing");
//		if (ImGui::Button(this->capturing ? "Stop Capture" : "Start Capture")) {
//			if (capturing = !capturing) {
//				this->vsrc.open(0);
//				this->vsrc.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//				this->vsrc.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//				this->vsrc.set(cv::CAP_PROP_FPS, 60);
//				this->stream_enable = true;
//				this->viewsize = { 0, 0 };
//			}
//			else {
//				this->vsrc.release();
//			}
//		}
//		ImGui::SameLine();
//		if (ImGui::Button("Load Video")) { openFile(this->vfile); }
//		ImGui::Text("Selected file: "); ImGui::SameLine(); ImGui::Text(this->vfile.c_str());
//		if (this->vfile.empty()) {
//			ImGui::BeginDisabled();
//		}
//		if (ImGui::Button("Run Video")) {
//			capturing = false;
//			if (!this->vsrc.open(this->vfile)) {
//				std::cout << "Error: failed to load video file\n";
//			}
//			else {
//				this->viewsize = { 0, 0 };
//				stream_enable = true;
//			}
//		}
//		if (ImGui::Button("Stop Video")) {
//			this->stream_enable = false;
//		}
//		if (this->vfile.empty()) {
//			ImGui::EndDisabled();
//		}
//		
//		ImGui::SliderFloat("Alpha", &this->alpha, 0.f, 1.f, "%.3f", 1);
//		ImGui::SliderFloat("Beta", &this->beta, 0.f, 1.f, "%.3f", 1);
//		ImGui::SliderFloat("Gamma", &this->gamma, 0.f, 100.f, "%.3f", 1);
//		if (ImGui::InputInt("ArUco ID", &this->id)) {
//			if (this->id > 6) { this->id = 6; }
//			if (this->id < 0) { this->id = 0; }
//			cv::aruco::drawMarker(this->markers_dict, this->id, 180, this->marker);
//			cv::cvtColor(this->marker, this->aruco_buff, cv::COLOR_GRAY2RGBA);
//			this->aruco_view->SetData(this->aruco_buff.data);
//		}
//		ImGui::Image(this->aruco_view->GetDescriptorSet(), { 180, 180 });
//		if (ImGui::Button("Save ArUco")) {
//			this->sfile = ("aruco_4x4_" + std::to_string(this->id) + ".png");
//			if (saveFile(this->sfile)) {
//				cv::imwrite(this->sfile, this->marker);
//			}
//		}
//		if (ImGui::Button("Save GridBoard")) {
//			cv::Mat board;
//			this->markers_export->draw({ 816, 1056 }, board, 96);
//			std::string f{ "aruco_board_4x4_5x7.png" };
//			if (saveFile(f)) {
//				cv::imwrite(f, board);
//			}
//		}
//		ImGui::End();
//
//		ImGui::Begin("Video");
//
//		if (this->stream_enable) {
//			if (this->vsrc.isOpened() && this->vsrc.read(this->cap)) {
//
//				if (this->viewsize != ImGui::GetContentRegionAvail()) {
//					this->viewsize = ImGui::GetContentRegionAvail();
//
//					this->framesize = (
//						this->cap.size().aspectRatio() > this->viewsize.x / this->viewsize.y ?
//						(cv::Size2d{ this->viewsize.x, this->viewsize.x / this->cap.size().aspectRatio() }) :
//						(this->cap.size().aspectRatio() < this->viewsize.x / this->viewsize.y ?
//							(cv::Size2d{ this->viewsize.y * this->cap.size().aspectRatio(), this->viewsize.y }) :
//							(cv::Size2f{ this->viewsize.x, this->viewsize.y })
//							));
//					this->resizeImageBuffers(this->framesize);
//				}
//
//				cv::resize(this->cap, this->proc, this->framesize);
//				cv::split(this->proc, this->channels);
//
//				cv::addWeighted(this->channels[0], this->alpha, this->channels[1], this->beta, this->gamma, this->binary);
//				cv::subtract(this->channels[2], this->binary, this->binary);
//
//				cv::aruco::detectMarkers(this->proc, this->markers_dict, this->corners, this->ids);
//
//				cv::cvtColor(this->binary, this->proc, cv::COLOR_GRAY2BGR);
//				cv::aruco::drawDetectedMarkers(this->proc, this->corners, this->ids);
//				cv::cvtColor(this->proc, this->rgba, cv::COLOR_BGR2RGBA);
//
//				this->stream_framebuff->SetData(this->rgba.data);
//			}
//			ImGui::Image(
//				this->stream_framebuff->GetDescriptorSet(),
//				{ (float)this->stream_framebuff->GetWidth(), (float)this->stream_framebuff->GetHeight() }
//			);
//			
//		}
//
//		ImGui::End();
//	}
//	if (this->data_w) {
//		ImGui::Begin("Data");
//		ImGui::End();
//	}
//	if (this->nn_w) {
//		ImGui::Begin("Training");
//		ImGui::End();
//	}
//	if (this->demo_w) {
//		ImGui::ShowDemoWindow();
//	}
//	
//
//}
//void AppLayer::menuFunctionality() {
//	if (ImGui::BeginMenu("File")) {
//		if (ImGui::MenuItem("Save")) {
//			std::cout << "Saved!\n";
//		}
//		ImGui::EndMenu();
//	}
//	if (ImGui::BeginMenu("Tools")) {
//		ImGui::MenuItem("Vision", NULL, &this->vision_w);
//		ImGui::MenuItem("Data", NULL, &this->data_w);
//		ImGui::MenuItem("Train", NULL, &this->nn_w);
//		ImGui::MenuItem("Demo", NULL, &this->demo_w);
//		ImGui::EndMenu();
//	}
//}
//
//
//void AppLayer::resizeImageBuffers(cv::Size nsize) {
//	this->proc = cv::Mat(nsize, CV_8UC3);
//	this->binary = cv::Mat(nsize, CV_8UC1);
//	this->rgba = cv::Mat(nsize, CV_8UC4);
//	this->channels = {
//		cv::Mat(nsize, CV_8UC1),
//		cv::Mat(nsize, CV_8UC1),
//		cv::Mat(nsize, CV_8UC1)
//	};
//	this->stream_framebuff->Resize(nsize.width, nsize.height);
//}